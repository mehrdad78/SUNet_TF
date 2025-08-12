import os, time, random, yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from skimage.morphology import binary_dilation
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

# --- your project imports ---
from model.SUNet import SUNet_model
from data_RGB import get_training_data, get_validation_data
import utils
from utils import network_parameters

# =========================
# Config (you can tweak)
# =========================
# Boundary-weight settings
K_RINGS   = 2
STROKE_W  = 3.0
RING_W    = (3.0, 2.0, 1.0)
NORM_MEAN_ONE = True

# Metrics settings
COMPUTE_TRAIN_ROC = False     # set True to also compute AUROC/AUPRC on training each epoch
VAL_AUROC_SUBSAMPLE = 0       # cap pixels per epoch (e.g., 200_000). 0 = no cap
TRAIN_AUROC_SUBSAMPLE = 0     # same cap for training collection

# =========================
# Repro
# =========================
torch.backends.cudnn.benchmark = True
random.seed(42); np.random.seed(42)
torch.manual_seed(42); torch.cuda.manual_seed_all(42)

# =========================
# Load config
# =========================
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']; OPT = opt['OPTIM']

# =========================
# Build model
# =========================
print('==> Build the model')
model_restored = SUNet_model(opt)
p_number = network_parameters(model_restored)
model_restored.cuda()
mode = opt['MODEL']['MODE']

# Dirs
model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
utils.mkdir(model_dir)
train_dir = Train['TRAIN_DIR']; val_dir = Train['VAL_DIR']

# GPUs
gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
if len(device_ids) > 1:
    model_restored = nn.DataParallel(model_restored, device_ids=device_ids)

# Log
log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

# Plot subfolders
plots_root = os.path.join(log_dir, 'plots')
loss_dir   = os.path.join(plots_root, 'loss')
mse_dir    = os.path.join(plots_root, 'mse')
roc_dir    = os.path.join(plots_root, 'roc')
pr_dir     = os.path.join(plots_root, 'pr')
overlay_dir= os.path.join(plots_root, 'overlay')
for d in [plots_root, loss_dir, mse_dir, roc_dir, pr_dir, overlay_dir]:
    os.makedirs(d, exist_ok=True)

# =========================
# Optim/Sched
# =========================
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, OPT['EPOCHS'] - warmup_epochs, eta_min=float(OPT['LR_MIN'])
)
scheduler = GradualWarmupScheduler(
    optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine
)
scheduler.step()

# Resume
if Train['RESUME']:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restored, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)
    for _ in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------')

# =========================
# Dataloaders
# =========================
print('==> Loading datasets')
train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS']})
train_loader  = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                           shuffle=True, num_workers=0, drop_last=False)
val_dataset   = get_validation_data(val_dir, {'patch_size': Train['VAL_PS']})
val_loader    = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
                           num_workers=0, drop_last=False)

# =========================
# Info
# =========================
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Train patches size: {Train['TRAIN_PS']}x{Train['TRAIN_PS']}
    Val patches size:   {Train['VAL_PS']}x{Train['VAL_PS']}
    Model parameters:   {p_number}
    Start/End epochs:   {start_epoch}~{OPT['EPOCHS']}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}
    GPU:                {'GPU' + str(device_ids)}''')
print('------------------------------------------------------------------')

# =========================
# Loss & helpers
# =========================
def charbonnier_loss(pred, target, weight=None, eps=1e-3):
    diff = pred - target
    l = torch.sqrt(diff * diff + eps * eps)
    if weight is None:
        return l.mean()
    return (l * weight).sum() / weight.sum().clamp(min=1e-8)

def background_adjacent_to_foreground(binary_image, k, footprint=None):
    if footprint is None:
        footprint = np.ones((3, 3), dtype=bool)  # 8-neighborhood
    prev = (binary_image > 0).astype(np.uint8)
    neigh_masks = []
    for _ in range(k):  # exactly k rings
        dil = binary_dilation(prev.astype(bool), footprint=footprint).astype(np.uint8)
        ring = (dil - prev).astype(bool)
        neigh_masks.append(ring)
        prev = dil
    return neigh_masks

def make_weight_matrix(binary_image, masks, stroke_w=STROKE_W, masks_w=RING_W, bg_min=0.0):
    h, w = binary_image.shape
    weights = np.zeros((h, w), dtype=np.float32)
    if bg_min > 0.0:
        weights[:] = float(bg_min)
    fg = (binary_image == 1)
    weights[fg] = float(stroke_w)
    for i, mask in enumerate(masks):
        wv = masks_w[i] if i < len(masks_w) else masks_w[-1]
        weights[mask] = float(wv)
    return weights

def make_weights_from_numpy(target_t, k=K_RINGS, stroke_w=STROKE_W, ring_w=RING_W,
                            normalize_to_mean_one=NORM_MEAN_ONE, bg_min=0.0):
    assert target_t.dim() == 4 and target_t.size(1) == 1, "expect (B,1,H,W)"
    device = target_t.device
    tgt_np = target_t.detach().cpu().numpy()
    if tgt_np.max() <= 1.0:
        bin_batch = (tgt_np > 0.5).astype(np.uint8)
    else:
        bin_batch = (tgt_np > 127).astype(np.uint8)
    weights_list = []
    for b in range(bin_batch.shape[0]):
        bin_img = bin_batch[b, 0]
        masks = background_adjacent_to_foreground(bin_img, k)
        w_np = make_weight_matrix(bin_img, masks, stroke_w=float(stroke_w), masks_w=list(ring_w)).astype(np.float32)
        if bg_min > 0.0:
            w_np[w_np == 0] = bg_min
        weights_list.append(w_np[None, None, ...])
    w_np_batch = np.concatenate(weights_list, axis=0)
    w = torch.from_numpy(w_np_batch).to(device=device, dtype=target_t.dtype)
    if float(w.sum()) == 0.0:
        w.fill_(1.0)
    if normalize_to_mean_one:
        w = w / w.mean().clamp(min=1e-8)
    return w

def get_last_trainable_leaf(model):
    last_name, last_mod = None, None
    for name, m in (model.named_modules()):
        if sum(1 for _ in m.children()) == 0 and any(p.requires_grad for p in m.parameters(recurse=False)):
            last_name, last_mod = name, m
    return last_name, last_mod

# =========================
# Train!
# =========================
print('==> Training start: ')
total_start_time = time.time()

loss_history = []
val_loss_history = []
val_epoch_list = []

mse_train_history = []
mse_val_history = []
mse_val_weighted_history = []

auroc_val_history = []
auprc_val_history = []

auroc_train_history = []
auprc_train_history = []

# --- Best metrics tracking ---
best_auroc = -1.0
best_auprc = -1.0
best_auroc_epoch, best_auprc_epoch = None, None
best_auroc_path, best_auprc_path = None, None


def _collect_scores(y_score, y_true, buf_scores, buf_trues, cap, collected_count):
    """Append scores/labels with an optional global cap to limit memory."""
    if cap <= 0:
        buf_scores.append(y_score); buf_trues.append(y_true)
        return collected_count + y_score.size
    remaining = cap - collected_count
    if remaining <= 0:
        return collected_count
    if y_score.size > remaining:
        idx = np.random.choice(y_score.size, remaining, replace=False)
        buf_scores.append(y_score[idx]); buf_trues.append(y_true[idx])
        return cap
    else:
        buf_scores.append(y_score); buf_trues.append(y_true)
        return collected_count + y_score.size

for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0.0

    # --- Train ---
    model_restored.train()
    train_mse_running = 0.0
    train_mse_batches = 0

    # (optional) train AUROC/AUPRC collectors
    train_probs_list = []; train_targets_list = []; train_collected = 0

    for i, data in enumerate(tqdm(train_loader), 0):
        # zero grad
        for p in model_restored.parameters():
            p.grad = None

        target = data[0].cuda()
        input_  = data[1].cuda()

        if target.shape[1] == 3:
            target = 0.2989 * target[:, 0:1] + 0.5870 * target[:, 1:2] + 0.1140 * target[:, 2:3]

        restored = model_restored(input_)

        # boundary weights + loss
        weights = make_weights_from_numpy(target, k=K_RINGS, stroke_w=STROKE_W, ring_w=RING_W)
        loss = charbonnier_loss(restored, target, weight=weights, eps=1e-3)

        # MSE metric (no grad)
        with torch.no_grad():
            prob = torch.sigmoid(restored)
            mse_b = torch.mean((prob - target) ** 2).item()
            train_mse_running += mse_b
            train_mse_batches += 1

            if COMPUTE_TRAIN_ROC:
                p = prob.detach().cpu().numpy().ravel()
                t = target.detach().cpu().numpy().ravel().astype(np.uint8)
                train_collected = _collect_scores(p, t, train_probs_list, train_targets_list,
                                                  TRAIN_AUROC_SUBSAMPLE, train_collected)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Per-epoch train MSE
    mse_train_epoch = train_mse_running / max(1, train_mse_batches)
    mse_train_history.append(mse_train_epoch)
    writer.add_scalar('train/mse', mse_train_epoch, epoch)

    # Optional train AUROC/AUPRC
    if COMPUTE_TRAIN_ROC and len(train_targets_list):
        y_score_tr = np.concatenate(train_probs_list); y_true_tr = np.concatenate(train_targets_list)
        if np.unique(y_true_tr).size == 2:
            auroc_tr = roc_auc_score(y_true_tr, y_score_tr)
            auprc_tr = average_precision_score(y_true_tr, y_score_tr)
            auroc_train_history.append(auroc_tr); auprc_train_history.append(auprc_tr)
            writer.add_scalar('train/auroc', auroc_tr, epoch)
            writer.add_scalar('train/auprc', auprc_tr, epoch)
        else:
            auroc_train_history.append(np.nan); auprc_train_history.append(np.nan)

    # --- Validation ---
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model_restored.eval()
        val_mse_running = 0.0
        val_mse_batches = 0
        val_mse_weighted_running = 0.0
        val_epoch_loss = 0.0

        # AUROC/AUPRC collectors
        val_probs_list = []; val_targets_list = []; val_collected = 0

        for ii, data_val in enumerate(val_loader, 0):
            target = data_val[0].cuda()
            input_  = data_val[1].cuda()

            if target.shape[1] == 3:
                target = 0.2989 * target[:, 0:1] + 0.5870 * target[:, 1:2] + 0.1140 * target[:, 2:3]

            with torch.no_grad():
                restored = model_restored(input_)
                prob = torch.sigmoid(restored)

                # plain MSE
                mse_b = torch.mean((prob - target) ** 2).item()
                val_mse_running += mse_b
                val_mse_batches += 1

                # boundary-weighted MSE + val loss
                val_weights = make_weights_from_numpy(target, k=K_RINGS, stroke_w=STROKE_W, ring_w=RING_W)
                se = (prob - target) ** 2
                mse_b_w = (se * val_weights).sum().item() / max(1e-8, val_weights.sum().item())
                val_mse_weighted_running += mse_b_w

                val_loss = charbonnier_loss(restored, target, weight=val_weights, eps=1e-3)
                val_epoch_loss += val_loss.item()

                # collect for AUROC/AUPRC
                p = prob.detach().cpu().numpy().ravel()
                t = target.detach().cpu().numpy().ravel().astype(np.uint8)
                val_collected = _collect_scores(p, t, val_probs_list, val_targets_list,
                                                VAL_AUROC_SUBSAMPLE, val_collected)

        # aggregate & log per-epoch
        val_mse_epoch = val_mse_running / max(1, val_mse_batches)
        mse_val_history.append(val_mse_epoch)
        writer.add_scalar('val/mse', val_mse_epoch, epoch)

        val_mse_weighted_epoch = val_mse_weighted_running / max(1, val_mse_batches)
        mse_val_weighted_history.append(val_mse_weighted_epoch)
        writer.add_scalar('val/mse_weighted', val_mse_weighted_epoch, epoch)

        val_loss_history.append(val_epoch_loss / max(1, len(val_loader)))
        val_epoch_list.append(epoch)

        # --- AUROC / AUPRC + ROC/PR plots ---
        y_score = np.concatenate(val_probs_list) if len(val_probs_list) else np.array([])
        y_true  = np.concatenate(val_targets_list) if len(val_targets_list) else np.array([])
        if y_true.size > 0 and np.unique(y_true).size == 2:
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
            auroc_val_history.append(auroc); auprc_val_history.append(auprc)
            writer.add_scalar('val/auroc', auroc, epoch)
            writer.add_scalar('val/auprc', auprc, epoch)

            fpr, tpr, _ = roc_curve(y_true, y_score)
            prec, rec, _ = precision_recall_curve(y_true, y_score)

            # ROC curve image
            plt.figure(figsize=(6,6))
            plt.plot(fpr, tpr, label=f'AUROC={auroc:.4f}')
            plt.plot([0,1], [0,1], '--', linewidth=1)
            plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC (epoch {epoch})')
            plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(roc_dir, f'roc_epoch_{epoch:03d}.png')); plt.close()

            # PR curve image
            plt.figure(figsize=(6,6))
            plt.plot(rec, prec, label=f'AP={auprc:.4f}')
            plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR (epoch {epoch})')
            plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(pr_dir, f'pr_epoch_{epoch:03d}.png')); plt.close()
        else:
            auroc_val_history.append(np.nan); auprc_val_history.append(np.nan)
            print(f"[val] AUROC/AUPRC undefined (mask had one class) at epoch {epoch}")

        # --- Overlay plot (all metrics) ---
        # Left y-axis: AUROC & AUPRC (0-1); Right y-axis: val loss & val MSE
        xs = val_epoch_list
        plt.figure(figsize=(9,6))
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(xs, auroc_val_history, marker='o', label='Val AUROC')
        ax1.plot(xs, auprc_val_history, marker='o', label='Val AUPRC')
        ax1.set_ylim(0, 1.0); ax1.set_ylabel('AUROC / AUPRC')
        ax2.plot(xs, [val_loss_history[i] for i in range(len(xs))], marker='s', label='Val Loss')
        ax2.plot(xs, [mse_val_history[i]  for i in range(len(xs))], marker='s', label='Val MSE')
        ax2.set_ylabel('Loss / MSE')
        ax1.set_xlabel('Epoch'); ax1.set_title('Overlay: AUROC/AUPRC vs Loss/MSE (Validation)')
        # Build combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        ax1.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(overlay_dir, f'overlay_epoch_{epoch:03d}.png')); plt.close()

    # --- Scheduler & checkpoints ---
    scheduler.step()
    # save "latest" (already there above)

# >>> NEW: also save each epoch from 5 to 15
    if 5 <= epoch <= 15:
        net = model_restored.module if hasattr(model_restored, "module") else model_restored
        torch.save({'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()},
               os.path.join(model_dir, f"model_epoch_{epoch:02d}.pth"))

    # save "latest" (only tensors)
    torch.save({
        'epoch': epoch,
        'state_dict': (model_restored.module if hasattr(model_restored, "module") else model_restored).state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(model_dir, "model_latest.pth"))

    # Logs
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(
        epoch, time.time() - epoch_start_time, epoch_loss, scheduler.get_last_lr()[0]
    ))
    print("------------------------------------------------------------------")

    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

    # Collect train loss for plotting
    loss_history.append(epoch_loss / max(1, len(train_loader)))

total_finish_time = (time.time() - total_start_time)
print('Total training time: {:.1f} hours'.format(total_finish_time / 3600.0))
writer.close()

# =========================
# Final time-series plots
# =========================
epochs_train = list(range(1, len(loss_history) + 1))

# Train vs Val MSE (time series)
plt.figure(figsize=(10, 6))
plt.plot(epochs_train[:len(mse_train_history)], mse_train_history, marker='o', label='Train MSE')
plt.plot(val_epoch_list, mse_val_history, marker='o', label='Val MSE')
plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.title('Train/Val MSE per Epoch')
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(mse_dir, 'mse_curves.png')); plt.close()

# Weighted Val MSE (time series)
plt.figure(figsize=(10, 6))
plt.plot(val_epoch_list, mse_val_weighted_history, marker='o', label='Val MSE (Weighted)')
plt.xlabel('Epoch'); plt.ylabel('MSE (Weighted)'); plt.title('Boundary-Weighted Val MSE per Epoch')
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(mse_dir, 'mse_weighted_curve.png')); plt.close()

# Train & Val Loss (time series)
plt.figure(figsize=(10, 6))
plt.plot(epochs_train, loss_history, marker='o', label='Training Loss')
if len(val_loss_history):
    plt.plot(val_epoch_list, val_loss_history, marker='o', label='Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training and Validation Loss per Epoch')
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(loss_dir, 'train_val_loss.png')); plt.close()

# Val AUROC/AUPRC (time series)
plt.figure(figsize=(10, 6))
plt.plot(val_epoch_list, auroc_val_history, marker='o', label='Val AUROC')
plt.plot(val_epoch_list, auprc_val_history, marker='o', label='Val AUPRC')
plt.ylim(0,1); plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('Val AUROC & AUPRC per Epoch')
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(roc_dir, 'val_auroc_auprc_curves.png')); plt.close()

# Optional: Train AUROC/AUPRC time series
if COMPUTE_TRAIN_ROC and len(auroc_train_history):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_train[:len(auroc_train_history)], auroc_train_history, marker='o', label='Train AUROC')
    plt.plot(epochs_train[:len(auprc_train_history)], auprc_train_history, marker='o', label='Train AUPRC')
    plt.ylim(0,1); plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('Train AUROC & AUPRC per Epoch')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(roc_dir, 'train_auroc_auprc_curves.png')); plt.close()


print("\n==================== Best checkpoints ====================")
if best_auroc_epoch is not None:
    print(f"Best AUROC : {best_auroc:.6f} at epoch {best_auroc_epoch} -> {best_auroc_path}")
else:
    print("Best AUROC : (not available; AUROC was undefined for all val epochs)")

if best_auprc_epoch is not None:
    print(f"Best AUPRC : {best_auprc:.6f} at epoch {best_auprc_epoch} -> {best_auprc_path}")
else:
    print("Best AUPRC : (not available; AUPRC was undefined for all val epochs)")
print("==========================================================\n")


# =========================
# Save numeric logs (CSV-like)
# =========================
metrics_txt = os.path.join(log_dir, 'metrics_per_epoch.txt')
with open(metrics_txt, 'w') as f:
    f.write('Epoch\tTrain_Loss\tVal_Loss\tTrain_MSE\tVal_MSE\tVal_MSEw\tVal_AUROC\tVal_AUPRC\n')
    for ep_idx, ep in enumerate(epochs_train):
        vloss = ''
        vmse  = ''
        vmsew = ''
        vauroc = ''
        vauprc = ''
        if ep in val_epoch_list:
            i = val_epoch_list.index(ep)
            vloss = f'{val_loss_history[i]:.6f}'
            vmse  = f'{mse_val_history[i]:.6f}'
            vmsew = f'{mse_val_weighted_history[i]:.6f}'
            vauroc = f'{auroc_val_history[i]:.6f}' if not np.isnan(auroc_val_history[i]) else ''
            vauprc = f'{auprc_val_history[i]:.6f}' if not np.isnan(auprc_val_history[i]) else ''
        f.write(f'{ep}\t{loss_history[ep_idx]:.6f}\t{vloss}\t{mse_train_history[ep_idx]:.6f}\t{vmse}\t{vmsew}\t{vauroc}\t{vauprc}\n')
