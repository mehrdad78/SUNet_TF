import os
import time
import random
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from skimage.morphology import binary_dilation
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

# project imports
from model.SUNet import SUNet_model
from data_RGB import get_training_data, get_validation_data
import utils
from utils import network_parameters


SPLIT_COLOR = {'train':'tab:blue','val':'tab:red','test':'tab:green'}
# optional: markers & linestyles so different metrics remain distinguishable
MARK = {'auroc':'o', 'auprc':'x', 'loss':'^', 'mse':'s', 'mse_w':'d'}
STYLE = {'train':'-', 'val':'--', 'test':':'}

# Boundary-weight settings
K_RINGS = 3
STROKE_W = 5.0
RING_W = (3.0, 2.0, 1.0)
NORM_MEAN_ONE = True

# ROC/PR collectors (subsample pixels to save RAM; 0 = no cap)
TRAIN_AUROC_SUBSAMPLE = 200_000
VAL_AUROC_SUBSAMPLE = 0
TEST_AUROC_SUBSAMPLE = 0

# Compute train ROC/PR too?
COMPUTE_TRAIN_ROC = True

# Validate every epoch? (use your YAML if you prefer)
FORCE_VAL_EVERY_EPOCH = True

# =========================
# Repro
# =========================
torch.backends.cudnn.benchmark = True
SEED=85

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =========================
# Load YAML
# =========================
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']

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
train_dir = Train['TRAIN_DIR']
val_dir = Train['VAL_DIR']

# GPUs
gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
if len(device_ids) > 1:
    model_restored = nn.DataParallel(model_restored, device_ids=device_ids)

# Logs
log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

# Plot folders
plots_root = os.path.join(log_dir, 'plots')
os.makedirs(plots_root, exist_ok=True)
overlay_tr_d = os.path.join(plots_root, 'overlay', 'train')
os.makedirs(overlay_tr_d, exist_ok=True)
overlay_v_d = os.path.join(plots_root, 'overlay', 'val')
os.makedirs(overlay_v_d,  exist_ok=True)
roc_tr_dir = os.path.join(plots_root, 'roc', 'train')
os.makedirs(roc_tr_dir,   exist_ok=True)
pr_tr_dir = os.path.join(plots_root, 'pr', 'train')
os.makedirs(pr_tr_dir,    exist_ok=True)
roc_val_dir = os.path.join(plots_root, 'roc', 'val')
os.makedirs(roc_val_dir,  exist_ok=True)
pr_val_dir = os.path.join(plots_root, 'pr', 'val')
os.makedirs(pr_val_dir,   exist_ok=True)
mse_dir = os.path.join(plots_root, 'mse')
os.makedirs(mse_dir,      exist_ok=True)
loss_dir = os.path.join(plots_root, 'loss')
os.makedirs(loss_dir,     exist_ok=True)
overlay_tv_d = os.path.join(plots_root, 'overlay', 'train_val')
os.makedirs(overlay_tv_d, exist_ok=True)
# NEW: combined Train+Val+Test overlay
overlay_tvt_d = os.path.join(plots_root, 'overlay', 'train_val_test')
os.makedirs(overlay_tvt_d, exist_ok=True)
weights_dbg_dir = os.path.join(plots_root, 'weights_debug')
os.makedirs(weights_dbg_dir, exist_ok=True)


# =========================
# Optimizer / Scheduler
# =========================
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_restored.parameters(),
                       lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, OPT['EPOCHS'] - warmup_epochs, eta_min=float(OPT['LR_MIN'])
)
scheduler = GradualWarmupScheduler(
    optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine
)
scheduler.step()

# Resume
if Train.get('RESUME', False):
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
# Data
# =========================
print('==> Loading datasets')
train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS']})
train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                          shuffle=True, num_workers=0, drop_last=False)
val_dataset = get_validation_data(val_dir, {'patch_size': Train['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
                        num_workers=0, drop_last=False)
# Optional TEST split (from YAML)
test_dir = Train.get('TEST_DIR', None)
test_loader = None
if test_dir and os.path.isdir(test_dir):
    test_dataset = get_validation_data(test_dir, {'patch_size': Train['VAL_PS']})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
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
def mse_loss(pred, target, weight=None):
    diff = (pred - target) ** 2
    if weight is None:
        return diff.mean()
    return (diff * weight).sum() / weight.sum().clamp(min=1e-8)


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


def make_weight_matrix(binary_image, masks, stroke_w=STROKE_W, masks_w=RING_W, bg_min=0.5):
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
    
    bin_batch = (tgt_np > 0.5).astype(np.uint8)

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

# =========================
# Weighting debug: compute & plot step-by-step
# =========================
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl

def _to_np01(t):
    """(B,1,H,W) -> (H,W) float32 in [0,1] for visualization."""
    a = t.detach().cpu().numpy()
    a = a[0,0] if a.ndim == 4 else a
    a = a.astype(np.float32)
    if a.max() > 1.0: a = a / 255.0
    return a

def compute_weighting_steps(target_t, k=K_RINGS, stroke_w=STROKE_W, ring_w=RING_W,
                            normalize_to_mean_one=NORM_MEAN_ONE, bg_min=0.0):

    assert target_t.dim() == 4 and target_t.size(1) == 1
    # Work on a single example for plotting
    tgt = target_t[0:1]

    # Build binary + rings exactly like training
    tgt_np = tgt.detach().cpu().numpy()
    
    bin_img = (tgt_np[0,0] > 0.5).astype(np.uint8)


    rings = background_adjacent_to_foreground(bin_img, k)   # list of bool masks

    # Raw (unnormalized) weights
    w_raw = make_weight_matrix(bin_img, rings, stroke_w=float(stroke_w), masks_w=list(ring_w)).astype(np.float32)
    if bg_min > 0.0:
        w_raw[w_raw == 0] = bg_min

    # Normalized weights (what your loss actually uses if NORM_MEAN_ONE=True)
    w_norm = w_raw.copy()
    if normalize_to_mean_one:
        m = w_norm.mean() if w_norm.size > 0 else 1.0
        w_norm = w_norm / max(1e-8, m)

    steps = {
        "binary": bin_img.astype(np.float32),
        "rings": rings,                # list of HxW bools
        "weights_raw": w_raw,          # float32
        "weights_norm": w_norm         # float32
    }
    return steps

def plot_weighting_steps(steps, save_path, title="Weighting process", cmap='hot', show_raw=True):
    """
    Makes a compact grid:
      [binary] [ring1] [ring2] ... [ringK] [weights_norm (or raw)]
    """
    rings = steps["rings"]
    K = len(rings)
    cols = 2 + K  # binary + rings + final weights
    fig, axes = plt.subplots(1, cols, figsize=(3.5*cols, 3.6))

    # 1) binary
    axes[0].imshow(steps["binary"], vmin=0, vmax=1, cmap='gray')
    axes[0].set_title("Binary mask")
    axes[0].axis('off')

    # 2) rings
    for i, r in enumerate(rings):
        axes[1+i].imshow(r.astype(np.float32), vmin=0, vmax=1, cmap='inferno')
        axes[1+i].set_title(f"Ring {i+1}")
        axes[1+i].axis('off')

    # 3) final weights
    W = steps["weights_raw"] if show_raw else steps["weights_norm"]
    im = axes[-1].imshow(W, cmap=cmap, interpolation='nearest')
    axes[-1].set_title("Weights (norm=1)" if not show_raw else "Weights (raw)")
    axes[-1].axis('off')

    # Colorbar only for the weight map
    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04).set_label('Weight value')

    fig.suptitle(title, y=0.95, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)



def _collect_scores(y_score, y_true, buf_scores, buf_trues, cap, collected_count):
    """Append scores/labels with an optional global cap to limit memory."""
    if cap <= 0:
        buf_scores.append(y_score)
        buf_trues.append(y_true)
        return collected_count + y_score.size
    remaining = cap - collected_count
    if remaining <= 0:
        return cap
    if y_score.size > remaining:
        idx = np.random.choice(y_score.size, remaining, replace=False)
        buf_scores.append(y_score[idx])
        buf_trues.append(y_true[idx])
        return cap
    else:
        buf_scores.append(y_score)
        buf_trues.append(y_true)
        return collected_count + y_score.size
    

# =========================
# Histories & best trackers
# =========================
loss_hist_tr = []
mse_hist_tr = []
mseW_hist_tr = []
auroc_hist_tr = []
auprc_hist_tr = []

loss_hist_val = []
mse_hist_val = []
mseW_hist_val = []
auroc_hist_val = []
auprc_hist_val = []
val_epoch_list = []

mse_hist_test = []
mseW_hist_test = []
auroc_hist_test = []
auprc_hist_test = []
test_epoch_list = []

best_auroc = -1.0
best_auprc = -1.0
best_auroc_epoch = best_auprc_epoch = None
best_auroc_path = best_auprc_path = None

# =========================
# Training
# =========================
print('==> Training start: ')
total_start_time = time.time()
VAL_AFTER = 1 if FORCE_VAL_EVERY_EPOCH else max(1, int(Train.get('VAL_AFTER_EVERY', 1)))

for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0.0

    # --- Train ---
    model_restored.train()
    tr_mse_sum = 0.0
    tr_mseW_sum = 0.0
    tr_batches = 0

    tr_probs_list, tr_targets_list = [], []
    tr_collected = 0
    tr_pos_total = tr_neg_total = 0
    tr_mixed = tr_skipped = 0

    for i, data in enumerate(tqdm(train_loader), 0):
        for p in model_restored.parameters():
            p.grad = None

        target = data[0].cuda()
        input_  = data[1].cuda()

        # if masks are RGB, convert; otherwise keep (B,1,H,W)
        if target.shape[1] == 3:
            target = 0.2989 * target[:, 0:1] + 0.5870 * target[:, 1:2] + 0.1140 * target[:, 2:3]

        logits = model_restored(input_)              # raw model output
        prob   = torch.sigmoid(logits)               # for metrics

        # weights & losses
        weights = make_weights_from_numpy(target, k=K_RINGS, stroke_w=STROKE_W, ring_w=RING_W)
        # NOTE: using logits in Charbonnier is fine with eps
        loss = charbonnier_loss(logits, target, weight=weights, eps=1e-3)
        N_SAMPLES_PER_EPOCH = 3
        if i < N_SAMPLES_PER_EPOCH and (epoch in (start_epoch, start_epoch+1) or epoch % 10 == 0):
            steps = compute_weighting_steps(target, k=K_RINGS, stroke_w=STROKE_W, ring_w=RING_W,
                                    normalize_to_mean_one=NORM_MEAN_ONE, bg_min=0.0)
            out_path = os.path.join(weights_dbg_dir, f"train_e{epoch:03d}_b{i:04d}.png")
            plot_weighting_steps(steps, out_path, title=f"Train — epoch {epoch}, batch {i}")
        # Train MSE & weighted MSE (no grad)
        with torch.no_grad():
            se = (logits - target) ** 2
            tr_mse_sum  += se.mean().item()
            tr_mseW_sum += (se * weights).sum().item() / max(1e-8, weights.sum().item())
            tr_batches  += 1

            if COMPUTE_TRAIN_ROC:
                p = prob.detach().cpu().numpy().ravel()
                t = target.detach().cpu().numpy().ravel()
                t = (t > 0.5).astype(np.uint8) if t.max() <= 1.0 else (t > 127).astype(np.uint8)
                pos = int(t.sum()); neg = int(t.size - pos)
                tr_pos_total += pos; tr_neg_total += neg
                if pos > 0 and neg > 0:
                    tr_mixed += 1
                    tr_collected = _collect_scores(p, t, tr_probs_list, tr_targets_list,
                                                   TRAIN_AUROC_SUBSAMPLE, tr_collected)
                else:
                    tr_skipped += 1

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Aggregate train metrics (per epoch)
    train_loss_epoch = epoch_loss / max(1, len(train_loader))
    mse_tr_epoch  = tr_mse_sum  / max(1, tr_batches)
    mseW_tr_epoch = tr_mseW_sum / max(1, tr_batches)
    loss_hist_tr.append(train_loss_epoch)
    mse_hist_tr.append(mse_tr_epoch)
    mseW_hist_tr.append(mseW_tr_epoch)

    writer.add_scalar('train/loss_epoch', train_loss_epoch, epoch)
    writer.add_scalar('train/mse', mse_tr_epoch, epoch)
    writer.add_scalar('train/mse_weighted', mseW_tr_epoch, epoch)

    # Train AUROC/AUPRC per epoch
    if COMPUTE_TRAIN_ROC and len(tr_targets_list):
        y_score_tr = np.concatenate(tr_probs_list)
        y_true_tr  = np.concatenate(tr_targets_list)
        if np.unique(y_true_tr).size == 2:
            auroc_tr = roc_auc_score(y_true_tr, y_score_tr)
            auprc_tr = average_precision_score(y_true_tr, y_score_tr)
            auroc_hist_tr.append(auroc_tr)
            auprc_hist_tr.append(auprc_tr)
            writer.add_scalar('train/auroc', auroc_tr, epoch)
            writer.add_scalar('train/auprc', auprc_tr, epoch)

            # ROC/PR plots (train)
            fpr, tpr, _ = roc_curve(y_true_tr, y_score_tr)
            prec, rec, _ = precision_recall_curve(y_true_tr, y_score_tr)

            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, label=f'AUROC={auroc_tr:.4f}', color='tab:blue')
            plt.plot([0, 1], [0, 1], '--', linewidth=1, color='gray')
            plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'Train ROC (epoch {epoch})')
            plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(roc_tr_dir, f'roc_train_epoch_{epoch:03d}.png'))
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.plot(rec, prec, label=f'AP={auprc_tr:.4f}', color='tab:orange')
            plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'Train PR (epoch {epoch})')
            plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(pr_tr_dir, f'pr_train_epoch_{epoch:03d}.png'))
            plt.close()
        else:
            auroc_hist_tr.append(np.nan)
            auprc_hist_tr.append(np.nan)
            print(f"[train] AUROC/AUPRC undefined (no mixed-class batches) at epoch {epoch}")
    else:
        auroc_hist_tr.append(np.nan)
        auprc_hist_tr.append(np.nan)

    print(f"[train@{epoch}] pos={tr_pos_total}, neg={tr_neg_total}, mixed_batches={tr_mixed}, skipped={tr_skipped}")

    # --- Validation ---
    if epoch % VAL_AFTER == 0:
        model_restored.eval()
        val_mse_sum = 0.0
        val_mseW_sum = 0.0
        val_batches = 0
        val_epoch_loss = 0.0

        val_probs_list, val_targets_list = [], []
        val_collected = 0
        pos_total = neg_total = 0
        mixed_items = skipped_single = 0

        with torch.no_grad():
            for data_val in val_loader:
                target = data_val[0].cuda()
                input_  = data_val[1].cuda()

                if target.shape[1] == 3:
                    target = 0.2989 * target[:, 0:1] + 0.5870 * target[:, 1:2] + 0.1140 * target[:, 2:3]

                logits = model_restored(input_)
                prob   = torch.sigmoid(logits)

                # MSE, MSE weighted, and val loss
                se = (logits - target) ** 2
                val_mse_sum += se.mean().item()

                weights = make_weights_from_numpy(target, k=K_RINGS, stroke_w=STROKE_W, ring_w=RING_W)
                if data_val is val_loader.dataset and val_batches == 1:
                    steps_val = compute_weighting_steps(target, k=K_RINGS, stroke_w=STROKE_W, ring_w=RING_W,
                                        normalize_to_mean_one=NORM_MEAN_ONE, bg_min=0.0)
                    out_path_v = os.path.join(weights_dbg_dir, f"val_e{epoch:03d}.png")
                    plot_weighting_steps(steps_val, out_path_v, title=f"Val — epoch {epoch}")

                val_mseW_sum += (se * weights).sum().item() / max(1e-8, weights.sum().item())

                val_loss = charbonnier_loss(logits, target, weight=weights, eps=1e-3)
                val_epoch_loss += val_loss.item()
                val_batches += 1

                # collect for AUROC/AUPRC if both classes present
                t_np  = target.detach().cpu().numpy().ravel()
                t_bin = (t_np > 0.5).astype(np.uint8) if t_np.max() <= 1.0 else (t_np > 127).astype(np.uint8)
                p_np  = prob.detach().cpu().numpy().ravel()

                pos = int(t_bin.sum()); neg = int(t_bin.size - pos)
                pos_total += pos; neg_total += neg

                if pos > 0 and neg > 0:
                    mixed_items += 1
                    val_collected = _collect_scores(p_np, t_bin, val_probs_list, val_targets_list,
                                                    VAL_AUROC_SUBSAMPLE, val_collected)
                else:
                    skipped_single += 1

        # Aggregate val metrics
        val_mse_epoch  = val_mse_sum  / max(1, val_batches)
        val_mseW_epoch = val_mseW_sum / max(1, val_batches)
        val_loss_epoch = val_epoch_loss / max(1, val_batches)

        mse_hist_val.append(val_mse_epoch)
        mseW_hist_val.append(val_mseW_epoch)
        loss_hist_val.append(val_loss_epoch)
        val_epoch_list.append(epoch)

        writer.add_scalar('val/mse', val_mse_epoch, epoch)
        writer.add_scalar('val/mse_weighted', val_mseW_epoch, epoch)
        writer.add_scalar('val/loss_epoch', val_loss_epoch, epoch)

        # AUROC / AUPRC
        y_score = np.concatenate(val_probs_list) if len(val_probs_list) else np.array([])
        y_true  = np.concatenate(val_targets_list) if len(val_targets_list) else np.array([])
        have_two = (y_true.size > 0 and np.unique(y_true).size == 2)

        print(f"[val@{epoch}] pos={pos_total}, neg={neg_total}, mixed_items={mixed_items}, skipped={skipped_single}")

        if have_two:
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
            auroc_hist_val.append(auroc)
            auprc_hist_val.append(auprc)
            writer.add_scalar('val/auroc', auroc, epoch)
            writer.add_scalar('val/auprc', auprc, epoch)

            fpr, tpr, _ = roc_curve(y_true, y_score)
            prec, rec, _ = precision_recall_curve(y_true, y_score)

            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, label=f'AUROC={auroc:.4f}', color='tab:blue')
            plt.plot([0, 1], [0, 1], '--', linewidth=1, color='gray')
            plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'Val ROC (epoch {epoch})')
            plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(roc_val_dir, f'roc_val_epoch_{epoch:03d}.png'))
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.plot(rec, prec, label=f'AP={auprc:.4f}', color='tab:orange')
            plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'Val PR (epoch {epoch})')
            plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(pr_val_dir, f'pr_val_epoch_{epoch:03d}.png'))
            plt.close()

            # Save best-by-AUROC / AUPRC (VAL)
            net = model_restored.module if hasattr(model_restored, "module") else model_restored
            if auroc > best_auroc:
                best_auroc = auroc
                best_auroc_epoch = epoch
                best_auroc_path = os.path.join(model_dir, f"model_best_auroc_e{epoch:03d}.pth")
                
            if auprc > best_auprc:
                best_auprc = auprc
                best_auprc_epoch = epoch
                best_auprc_path = os.path.join(model_dir, f"model_best_auprc_e{epoch:03d}.pth")
               
        else:
            auroc_hist_val.append(np.nan)
            auprc_hist_val.append(np.nan)
            print(f"[val] AUROC/AUPRC undefined (no mixed-class masks collected) at epoch {epoch}")

        # --- TEST (same cadence as VAL) ---
        if test_loader is not None:
            model_restored.eval()
            test_mse_sum = 0.0; test_mseW_sum = 0.0; test_batches = 0
            probs_list = []; tgts_list = []
            pos_total_t = neg_total_t = 0; mixed_items_t = skipped_single_t = 0
            collected_t = 0
            with torch.no_grad():
                for data_test in test_loader:
                    target = data_test[0].cuda()
                    input_  = data_test[1].cuda()
                    if target.shape[1] == 3:
                        target = 0.2989 * target[:, 0:1] + 0.5870 * target[:, 1:2] + 0.1140 * target[:, 2:3]
                    logits = model_restored(input_)
                    prob   = torch.sigmoid(logits)
                    se = (logits - target) ** 2
                    test_mse_sum  += se.mean().item()
                    w = make_weights_from_numpy(target, k=K_RINGS, stroke_w=STROKE_W, ring_w=RING_W)
                    test_mseW_sum += (se * w).sum().item() / max(1e-8, w.sum().item())
                    test_batches  += 1

                    t_np  = target.detach().cpu().numpy().ravel()
                    t_bin = (t_np > 0.5).astype(np.uint8) if t_np.max() <= 1.0 else (t_np > 127).astype(np.uint8)
                    p_np  = prob.detach().cpu().numpy().ravel()
                    pos = int(t_bin.sum()); neg = int(t_bin.size - pos)
                    pos_total_t += pos; neg_total_t += neg
                    if pos > 0 and neg > 0:
                        mixed_items_t += 1
                        collected_t = _collect_scores(p_np, t_bin, probs_list, tgts_list,
                                                      TEST_AUROC_SUBSAMPLE, collected_t)
                    else:
                        skipped_single_t += 1

            test_mse_epoch  = test_mse_sum  / max(1, test_batches)
            test_mseW_epoch = test_mseW_sum / max(1, test_batches)
            mse_hist_test.append(test_mse_epoch)
            mseW_hist_test.append(test_mseW_epoch)
            test_epoch_list.append(epoch)
            writer.add_scalar('test/mse', test_mse_epoch, epoch)
            writer.add_scalar('test/mse_weighted', test_mseW_epoch, epoch)

            if len(tgts_list):
                y_true_t  = np.concatenate(tgts_list)
                y_score_t = np.concatenate(probs_list)
                if np.unique(y_true_t).size == 2:
                    auroc_t = roc_auc_score(y_true_t, y_score_t)
                    auprc_t = average_precision_score(y_true_t, y_score_t)
                    auroc_hist_test.append(auroc_t); auprc_hist_test.append(auprc_t)
                    writer.add_scalar('test/auroc', auroc_t, epoch)
                    writer.add_scalar('test/auprc', auprc_t, epoch)
                else:
                    auroc_hist_test.append(np.nan); auprc_hist_test.append(np.nan)
                    print(f"[test] AUROC/AUPRC undefined (one-class) at epoch {epoch}")
            else:
                auroc_hist_test.append(np.nan); auprc_hist_test.append(np.nan)

    # =========================
    # Per-epoch OVERLAY plots
    # =========================
    # TRAIN overlay (up to this epoch)
    xs_tr = list(range(1, len(loss_hist_tr) + 1))
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca(); ax2 = ax1.twinx()

    # AUROC/AUPRC on left (0..1)
    ax1.plot(xs_tr, auroc_hist_tr, marker='o', color='tab:blue',   label='Train AUROC')
    ax1.plot(xs_tr, auprc_hist_tr, marker='o', color='tab:orange', label='Train AUPRC')
    ax1.set_ylim(0, 1.0); ax1.set_ylabel('AUROC / AUPRC')

    # Loss/MSE on right
    ax2.plot(xs_tr, loss_hist_tr, marker='^', color='tab:red',    label='Train Loss', linestyle='-')
    ax2.plot(xs_tr, mse_hist_tr,  marker='s', color='tab:green',  label='Train MSE')
    ax2.plot(xs_tr, mseW_hist_tr, marker='d', color='tab:purple', label='Train MSE (Weighted)')
    ax2.set_ylabel('Loss / MSE')

    ax1.set_xlabel('Epoch'); ax1.set_title('TRAIN Overlay (epoch {})'.format(epoch))
    h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='best'); ax1.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(overlay_tr_d, f'overlay_train_up_to_epoch_{epoch:03d}.png'))
    plt.close()

    # VAL overlay (only for validated epochs)
    xs_val = val_epoch_list
    if len(xs_val) > 0:
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca(); ax2 = ax1.twinx()
        ax1.plot(xs_val, auroc_hist_val, marker='o', color='tab:blue',   label='Val AUROC')
        ax1.plot(xs_val, auprc_hist_val, marker='o', color='tab:orange', label='Val AUPRC')
        ax1.set_ylim(0, 1.0); ax1.set_ylabel('AUROC / AUPRC')
        ax2.plot(xs_val, mse_hist_val,  marker='s', color='tab:green',  label='Val MSE')
        ax2.plot(xs_val, mseW_hist_val, marker='d', color='tab:purple', label='Val MSE (Weighted)')
        # also put train loss on same axis for epoch alignment
        tr_loss_for_val = [loss_hist_tr[e-1] for e in xs_val]
        ax2.plot(xs_val, tr_loss_for_val, marker='^', color='tab:red', linestyle='--', label='Train Loss')
        ax2.set_ylabel('Loss / MSE')
        ax1.set_xlabel('Epoch'); ax1.set_title('VAL Overlay (epoch {})'.format(epoch))
        h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='best'); ax1.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(overlay_v_d, f'overlay_val_up_to_epoch_{epoch:03d}.png'))
        plt.close()

    # === Combined TRAIN+VAL overlay (all metrics) ===
    if len(xs_tr) > 0 and len(xs_val) > 0:
        plt.figure(figsize=(12, 7))
        ax1 = plt.gca(); ax2 = ax1.twinx()
        # Left axis: AUROC/AUPRC (0..1)
        ax1.plot(xs_tr,  auroc_hist_tr, marker='o', color='tab:blue',   label='Train AUROC')
        ax1.plot(xs_val, auroc_hist_val, marker='o', color='tab:blue',   linestyle='--', label='Val AUROC')
        ax1.plot(xs_tr,  auprc_hist_tr, marker='o', color='tab:orange', label='Train AUPRC')
        ax1.plot(xs_val, auprc_hist_val, marker='o', color='tab:orange', linestyle='--', label='Val AUPRC')
        ax1.set_ylim(0, 1.0); ax1.set_ylabel('AUROC / AUPRC')
        # Right axis: Loss / MSE / Weighted MSE
        ax2.plot(xs_tr,  loss_hist_tr, marker='^', color='tab:red',    label='Train Loss')
        ax2.plot(xs_val, loss_hist_val, marker='^', color='tab:red',    linestyle='--', label='Val Loss')
        ax2.plot(xs_tr,  mse_hist_tr,  marker='s', color='tab:green',  label='Train MSE')
        ax2.plot(xs_val, mse_hist_val, marker='s', color='tab:green',  linestyle='--', label='Val MSE')
        ax2.plot(xs_tr,  mseW_hist_tr, marker='d', color='tab:purple', label='Train MSE (Weighted)')
        ax2.plot(xs_val, mseW_hist_val, marker='d', color='tab:purple', linestyle='--', label='Val MSE (Weighted)')
        ax2.set_ylabel('Loss / MSE')
        ax1.set_xlabel('Epoch'); ax1.set_title(f'Train + Val Overlay (up to epoch {epoch})')
        h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='best'); ax1.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(overlay_tv_d, f'overlay_train_val_up_to_epoch_{epoch:03d}.png'))
        plt.close()

    # === Combined TRAIN+VAL+TEST overlay (metrics) ===
    # === Split-by-goodness overlays (TRAIN+VAL+TEST) ===
    xs_tr = list(range(1, len(loss_hist_tr) + 1))
    xs_val = val_epoch_list
    xs_te = test_epoch_list

    if len(xs_tr) > 0 and len(xs_val) > 0 and len(xs_te) > 0:
        C_TR = 'tab:blue'   # train
        C_VA = 'tab:red'    # val
        C_TE = 'tab:green'  # test

        # -------- High-is-good: AUROC & AUPRC --------
        plt.figure(figsize=(12, 7))
        # Train (blue)
        plt.plot(xs_tr, auroc_hist_tr,  marker='o', linestyle='-',  color=C_TR, label='Train AUROC')
        plt.plot(xs_tr, auprc_hist_tr,  marker='s', linestyle='--', color=C_TR, label='Train AUPRC')
        # Val (red)
        plt.plot(xs_val, auroc_hist_val, marker='o', linestyle='-',  color=C_VA, label='Val AUROC')
        plt.plot(xs_val, auprc_hist_val, marker='s', linestyle='--', color=C_VA, label='Val AUPRC')
        # Test (green)
        plt.plot(xs_te, auroc_hist_test, marker='o', linestyle='-',  color=C_TE, label='Test AUROC')
        plt.plot(xs_te, auprc_hist_test, marker='s', linestyle='--', color=C_TE, label='Test AUPRC')

        plt.ylim(0, 1.0)
        plt.xlabel('Epoch')
        plt.ylabel('Score (higher is better)')
        plt.title(f'AUROC & AUPRC (Train/Val/Test) — up to epoch {epoch}')
        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(overlay_tvt_d, f'high_metrics_up_to_epoch_{epoch:03d}.png'))
        plt.close()

        # -------- Low-is-good: Loss, MSE, Weighted MSE --------
        plt.figure(figsize=(12, 7))
        # Train (blue)
        plt.plot(xs_tr, loss_hist_tr,  marker='^', linestyle='-',  color=C_TR, label='Train Loss')
        plt.plot(xs_tr, mse_hist_tr,   marker='d', linestyle='-.', color=C_TR, label='Train MSE')
        plt.plot(xs_tr, mseW_hist_tr,  marker='x', linestyle=':',  color=C_TR, label='Train MSE (W)')
        # Val (red)
        plt.plot(xs_val, loss_hist_val,  marker='^', linestyle='-',  color=C_VA, label='Val Loss')
        plt.plot(xs_val, mse_hist_val,   marker='d', linestyle='-.', color=C_VA, label='Val MSE')
        plt.plot(xs_val, mseW_hist_val,  marker='x', linestyle=':',  color=C_VA, label='Val MSE (W)')
        # Test (green) — typically we don’t track test *loss*, so only MSEs:
        plt.plot(xs_te, mse_hist_test,   marker='d', linestyle='-.', color=C_TE, label='Test MSE')
        plt.plot(xs_te, mseW_hist_test,  marker='x', linestyle=':',  color=C_TE, label='Test MSE (W)')

        plt.xlabel('Epoch')
        plt.ylabel('Loss / Error (lower is better)')
        plt.title(f'Loss, MSE, Weighted MSE (Train/Val/Test) — up to epoch {epoch}')
        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(overlay_tvt_d, f'low_metrics_up_to_epoch_{epoch:03d}.png'))
        plt.close()


    # =========================
    # Scheduler & checkpoints
    # =========================
    scheduler.step()

    # save "latest"
    torch.save({
        'epoch': epoch,
        'state_dict': (model_restored.module if hasattr(model_restored, "module") else model_restored).state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(model_dir, "model_latest.pth"))


    
    # Console log per epoch
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(
        epoch, time.time() - epoch_start_time, epoch_loss, scheduler.get_last_lr()[0]
    ))
    if len(val_epoch_list) and val_epoch_list[-1] == epoch:
        i = len(val_epoch_list) - 1
        v_mse, v_mseW = mse_hist_val[i], mseW_hist_val[i]
        v_auroc, v_auprc = auroc_hist_val[i], auprc_hist_val[i]
        print(f"[val@{epoch}] MSE={v_mse:.6f}  MSEw={v_mseW:.6f}  AUROC={v_auroc:.6f}  AUPRC={v_auprc:.6f}")
    print("------------------------------------------------------------------")

# =========================
# Wrap up
# =========================
total_finish_time = (time.time() - total_start_time)
print('Total training time: {:.1f} hours'.format(total_finish_time / 3600.0))
writer.close()

# =========================
# Final time-series plots (optional summary)
# =========================
epochs_tr = list(range(1, len(loss_hist_tr) + 1))

# Train & Val Loss curves
plt.figure(figsize=(10, 6))
plt.plot(epochs_tr, loss_hist_tr, marker='o', label='Train Loss', color='tab:red')
if len(loss_hist_val):
    plt.plot(val_epoch_list, loss_hist_val, marker='o', label='Val Loss', color='tab:pink')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss per Epoch')
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(loss_dir, 'train_val_loss.png'))
plt.close()

# =========================
# Save numeric logs
# =========================
# --- Collect all metrics into a DataFrame ---
records = []

for ep in epochs_tr:
    idx = ep - 1
    record = {
        "Epoch": ep,
        "Train_Loss": loss_hist_tr[idx],
        "Train_MSE": mse_hist_tr[idx],
        #"Train_MSEw": mseW_hist_tr[idx],
        "Train_AUROC": auroc_hist_tr[idx] if not np.isnan(auroc_hist_tr[idx]) else None,
        "Train_AUPRC": auprc_hist_tr[idx] if not np.isnan(auprc_hist_tr[idx]) else None,
        "Val_Loss": None, "Val_MSE": None, "Val_MSEw": None,
        "Val_AUROC": None, "Val_AUPRC": None,
        "Test_MSE": None, "Test_MSEw": None, "Test_AUROC": None, "Test_AUPRC": None,
    }

    # Fill VAL if available
    if ep in val_epoch_list:
        i = val_epoch_list.index(ep)
        record.update({
            "Val_Loss":  loss_hist_val[i],
            "Val_MSE":   mse_hist_val[i],
            #"Val_MSEw":  mseW_hist_val[i],
            "Val_AUROC": auroc_hist_val[i] if not np.isnan(auroc_hist_val[i]) else None,
            "Val_AUPRC": auprc_hist_val[i] if not np.isnan(auprc_hist_val[i]) else None,
        })

    # Fill TEST if available
    if ep in test_epoch_list:
        j = test_epoch_list.index(ep)
        record.update({
            "Test_MSE":   mse_hist_test[j],
            #"Test_MSEw":  mseW_hist_test[j],
            "Test_AUROC": auroc_hist_test[j] if not np.isnan(auroc_hist_test[j]) else None,
            "Test_AUPRC": auprc_hist_test[j] if not np.isnan(auprc_hist_test[j]) else None,
        })

    records.append(record)

df = pd.DataFrame(records)

# --- Save to CSV ---
csv_path = os.path.join(log_dir, "metrics_per_epoch.csv")
df.to_csv(csv_path, index=False, float_format="%.6f")

print(f"✅ Metrics saved to {csv_path}")
# =========================
# Print best checkpoints (by VAL)
# =========================
print("\n==================== Best checkpoints (by VAL) ====================")
if best_auroc_epoch is not None:
    print(f"Best AUROC : {best_auroc:.6f} at epoch {best_auroc_epoch} -> {best_auroc_path}")
else:
    print("Best AUROC : (not available; AUROC was undefined for all val epochs)")
if best_auprc_epoch is not None:
    print(f"Best AUPRC : {best_auprc:.6f} at epoch {best_auprc_epoch} -> {best_auprc_path}")
else:
    print("Best AUPRC : (not available; AUPRC was undefined for all val epochs)")
print("==========================================================\n")
