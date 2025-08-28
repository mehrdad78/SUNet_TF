import os
import time
import random
import yaml
import numpy as np
import torch.nn.functional as F

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

# =========================
# Settings you can tweak
# =========================
# ---- color/style palette (split-based colors) ----
SPLIT_COLOR = {'train': 'tab:blue', 'val': 'tab:red', 'test': 'tab:green'}
# optional: markers & linestyles so different metrics remain distinguishable
MARK = {'auroc': 'o', 'auprc': 'x', 'loss': '^', 'mse': 's', 'mse_w': 'd'}
STYLE = {'train': '-', 'val': '--', 'test': ':'}

# Boundary-weight settings
K_RINGS = 2
STROKE_W = 5.0
RING_W   = (4.0, 2.0)  # or (3.0, 2.5)

NORM_MEAN_ONE = False
FG_IS_WHITE = False 
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
SEED = 85

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
    test_dataset = get_validation_data(
        test_dir, {'patch_size': Train['VAL_PS']})
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
# ========= GPU dilation + weights (Torch) =========


def _dilate_torch(bin_mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    bin_mask: (H, W) bool/0-1 tensor on any device
    returns: (H, W) bool tensor after one binary dilation step with square kernel
    """
    x = bin_mask.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    pad = kernel_size // 2
    weight = torch.ones((1, 1, kernel_size, kernel_size),
                        device=bin_mask.device)
    y = F.conv2d(x, weight, padding=pad)
    return (y > 0).squeeze(0).squeeze(0)  # bool


@torch.no_grad()
def background_adjacent_to_foreground_torch(binary_image_np: np.ndarray, k: int,
                                            kernel_size: int = 3, device: str = "cuda"):
    """
    binary_image_np: 0/1 numpy (H,W). 1=foreground, 0=background
    returns: list of k boolean torch masks (rings) on device
    """
    dev = device if (torch.cuda.is_available()
                     and device.startswith("cuda")) else "cpu"
    prev = torch.from_numpy(binary_image_np.astype(np.uint8)).to(dev).bool()
    rings = []
    for _ in range(k):                 # exactly k rings (matches your old behavior)
        dil = _dilate_torch(prev, kernel_size=kernel_size)
        ring = (dil & (~prev))         # newly added boundary band
        rings.append(ring)
        prev = dil
    return rings


@torch.no_grad()
def make_weights_from_torch(target_t: torch.Tensor,
                            k: int = 2,
                            stroke_w: float = 3.0,
                            ring_w=(3.0, 2.0, 1.0),
                            normalize_to_mean_one: bool = True,
                            bg_min: float = 0.5,
                            kernel_size: int = 3,
                            device: str = "cuda") -> torch.Tensor:
    """
    target_t: (B,1,H,W) tensor in [0,1] or [0,255]
    returns: (B,1,H,W) weights tensor on same device/dtype as target_t
    """
    assert target_t.dim() == 4 and target_t.size(1) == 1, "expect (B,1,H,W)"
    dev = target_t.device
    dtype = target_t.dtype

    # binarize per-sample (GPU-friendly; only the mask gen uses a short CPU hop for rings)
    # We keep the same thresholds you used. 
  # <-- set False if strokes are black on white bg

# ...
    if target_t.size(1) != 1:
        raise ValueError("make_weights_from_torch expects (B,1,H,W)")

    bin_batch = _binarize_mask(target_t)  

    B, _, H, W = target_t.shape
    weights = torch.full((B, 1, H, W), fill_value=bg_min,
                         dtype=torch.float32, device=dev)

    for b in range(B):
        # tiny hop to CPU for ring building input
        bin_img = bin_batch[b, 0].to(torch.uint8).cpu().numpy()
        rings = background_adjacent_to_foreground_torch(
            bin_img, k=k, kernel_size=kernel_size, device="cuda" if dev.type == "cuda" else "cpu"
        )  # rings are bool tensors on dev

        # foreground weight
        fg = bin_batch[b, 0]  # bool on dev
        weights[b, 0][fg] = float(stroke_w)

        # ring weights
        for i, r in enumerate(rings):
            wv = ring_w[i] if i < len(ring_w) else ring_w[-1]
            weights[b, 0][r] = float(wv)

    # avoid division by zero
    if normalize_to_mean_one:
        m = weights.mean().clamp(min=1e-8)
        weights = weights / m

    return weights.to(dtype=dtype)
# ========= Debug plotting of weighting process =========
def _ensure_dir(d): 
    os.makedirs(d, exist_ok=True)

@torch.no_grad()
def save_weighting_debug(target_t: torch.Tensor, k: int, out_dir: str, tag: str,
                         ring_w=(3.0,2.0,1.0), kernel_size=3):
    """
    Saves: (a) FG mask, (b) each ring mask, (c) raw + normalized heatmap
    target_t: (1,1,H,W) tensor
    """
    _ensure_dir(out_dir)
    dev = target_t.device

    # binarize (0/255 → bool)
    bin_img = _binarize_mask(target_t[:1])[0,0]  # (H,W) bool on dev
    H, W = bin_img.shape

    # --- rings ---
    bin_np = bin_img.to(torch.uint8).cpu().numpy()
    rings = background_adjacent_to_foreground_torch(
        bin_np, k=k, kernel_size=kernel_size, device="cuda" if dev.type=="cuda" else "cpu"
    )

    # plot foreground
    plt.figure(figsize=(4,4))
    plt.imshow(bin_img.cpu().numpy(), cmap='gray')
    plt.title(f'Foreground (tag={tag})'); plt.axis('off')
    plt.savefig(os.path.join(out_dir, f'{tag}_fg.png')); plt.close()

    # plot rings one by one
    for i, r in enumerate(rings, 1):
        plt.figure(figsize=(4,4))
        plt.imshow(r.cpu().numpy(), cmap='gray')
        plt.title(f'Ring {i}/{k}'); plt.axis('off')
        plt.savefig(os.path.join(out_dir, f'{tag}_ring_{i}.png')); plt.close()

    # build weight map
    weights = torch.zeros((H,W), dtype=torch.float32, device=dev)
    weights[bin_img] = float(STROKE_W)
    for i, r in enumerate(rings):
        wv = ring_w[i] if i < len(ring_w) else ring_w[-1]
        weights[r] = float(wv)

    # raw
    plt.figure(figsize=(5,5))
    plt.imshow(weights.cpu().numpy(), cmap='hot')
    plt.colorbar(label='Raw Weight Value')
    plt.title(f'Raw Weights (tag={tag})'); plt.axis('off')
    plt.savefig(os.path.join(out_dir, f'{tag}_weights_heatmap_raw.png')); plt.close()

    # normalized
    norm_w = weights / weights[weights > 0].mean().clamp(min=1e-8) if NORM_MEAN_ONE else weights
    plt.figure(figsize=(5,5))
    plt.imshow(norm_w.cpu().numpy(), cmap='hot')
    plt.colorbar(label='Weight Value')
    plt.title(f'Normalized Weights (tag={tag})'); plt.axis('off')
    plt.savefig(os.path.join(out_dir, f'{tag}_weights_heatmap.png')); plt.close()



def _binarize_mask(t: torch.Tensor) -> torch.Tensor:
    """
    Assumes white background (1.0 or 255), black foreground (0.0).
    Returns: bool mask, True = foreground (black stroke).
    """
    if t.dtype.is_floating_point:
        if t.max() <= 1.0 + 1e-6:
            return (t < 0.5)      # black ~0 → stroke
        else:
            return (t == 0.0)     # 0/255 scale
    else:
        return (t == 0)



@torch.no_grad()
def save_rings_debug(target_t: torch.Tensor, k: int, out_dir: str, tag: str,
                     kernel_size: int = 3):
    """
    Save and plot foreground and every dilation ring separately.
    target_t: (1,1,H,W) tensor
    """
    os.makedirs(out_dir, exist_ok=True)
    dev = target_t.device

    # binarize once
    bin_img = _binarize_mask(target_t[:1])[0,0]
    bin_np  = bin_img.to(torch.uint8).cpu().numpy()

    # compute rings
    rings = background_adjacent_to_foreground_torch(
        bin_np, k=k, kernel_size=kernel_size,
        device="cuda" if dev.type=="cuda" else "cpu"
    )

    # foreground
    plt.figure(figsize=(4,4))
    plt.imshow(bin_img.cpu().numpy(), cmap='gray', interpolation='nearest')
    plt.title(f'Foreground (tag={tag})'); plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{tag}_fg.png'))
    plt.close()

    # each ring individually
    for i, r in enumerate(rings, 1):
        plt.figure(figsize=(4,4))
        plt.imshow(r.cpu().numpy(), cmap='gray', interpolation='nearest')
        plt.title(f'Ring {i}/{k} (tag={tag})')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{tag}_ring_{i}.png'))
        plt.close()


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
VAL_AFTER = 1 if FORCE_VAL_EVERY_EPOCH else max(
    1, int(Train.get('VAL_AFTER_EVERY', 1)))

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
        input_ = data[1].cuda()

      
        # if masks are RGB, convert; otherwise keep (B,1,H,W)
        if target.shape[1] == 3:
            target = 0.2989 * target[:, 0:1] + 0.5870 * \
                target[:, 1:2] + 0.1140 * target[:, 2:3]
            
        print("RAW target range:", float(data[0].min()), float(data[0].max()), data[0].dtype)

        if target.dtype.is_floating_point and target.max() <= 1.0 + 1e-6:
    # already [0,1], do nothing
            pass
        else:
    # convert [0,255] → [0,1]
            target = target / 255.0

        target = 1.0 - target
        

        if i == 0:  # only first batch per epoch
            debug_dir = os.path.join(plots_root, 'weights_debug', 'train')
            print("target range:", float(target.min()), float(target.max()), target.dtype)
            fg = _binarize_mask(target[:1])
            print("fg ratio:", fg.float().mean().item())

            save_weighting_debug(target[:1], k=K_RINGS, out_dir=debug_dir, tag=f'epoch_{epoch:03d}_train')
        if i == 0:  # only first batch per epoch
            debug_dir = os.path.join(plots_root, 'rings_debug', 'train')
            save_rings_debug(target[:1], k=K_RINGS, out_dir=debug_dir,
                     tag=f'epoch_{epoch:03d}_train')


        logits = model_restored(input_)              # raw model output
        prob = torch.sigmoid(logits)               # for metrics

        # weights & losses
        weights = make_weights_from_torch(
            target, k=K_RINGS, stroke_w=STROKE_W, ring_w=RING_W,
            normalize_to_mean_one=NORM_MEAN_ONE, bg_min=0.0, kernel_size=3
        )
        # NOTE: using logits in Charbonnier is fine with eps
        loss = charbonnier_loss(logits, target, weight=weights, eps=1e-3)

        # Train MSE & weighted MSE (no grad)
        with torch.no_grad():
            se = (prob - target) ** 2
            tr_mse_sum += se.mean().item()
            tr_mseW_sum += (se * weights).sum().item() / \
                max(1e-8, weights.sum().item())
            tr_batches += 1

            if COMPUTE_TRAIN_ROC:
                p = prob.detach().cpu().numpy().ravel()
                t = target.detach().cpu().numpy().ravel()
                t = (t > 0.5).astype(np.uint8) if t.max(
                ) <= 1.0 else (t > 127).astype(np.uint8)
                pos = int(t.sum())
                neg = int(t.size - pos)
                tr_pos_total += pos
                tr_neg_total += neg
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
    mse_tr_epoch = tr_mse_sum / max(1, tr_batches)
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
        y_true_tr = np.concatenate(tr_targets_list)
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
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title(f'Train ROC (epoch {epoch})')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(
                roc_tr_dir, f'roc_train_epoch_{epoch:03d}.png'))
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.plot(rec, prec, label=f'AP={auprc_tr:.4f}', color='tab:orange')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Train PR (epoch {epoch})')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(
                pr_tr_dir, f'pr_train_epoch_{epoch:03d}.png'))
            plt.close()
        else:
            auroc_hist_tr.append(np.nan)
            auprc_hist_tr.append(np.nan)
            print(
                f"[train] AUROC/AUPRC undefined (no mixed-class batches) at epoch {epoch}")
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
                input_ = data_val[1].cuda()

                if target.shape[1] == 3:
                    target = 0.2989 * target[:, 0:1] + 0.5870 * \
                        target[:, 1:2] + 0.1140 * target[:, 2:3]

                logits = model_restored(input_)
                prob = torch.sigmoid(logits)

                # MSE, MSE weighted, and val loss
                se = (prob - target) ** 2
                val_mse_sum += se.mean().item()

                weights = make_weights_from_torch(
                    target, k=K_RINGS, stroke_w=STROKE_W, ring_w=RING_W,
                    normalize_to_mean_one=NORM_MEAN_ONE, bg_min=0.0, kernel_size=3
                )
                val_mseW_sum += (se * weights).sum().item() / \
                    max(1e-8, weights.sum().item())

                val_loss = charbonnier_loss(
                    logits, target, weight=weights, eps=1e-3)
                val_epoch_loss += val_loss.item()
                val_batches += 1

                # collect for AUROC/AUPRC if both classes present
                t_np = target.detach().cpu().numpy().ravel()
                t_bin = (t_np > 0.5).astype(np.uint8) if t_np.max(
                ) <= 1.0 else (t_np > 127).astype(np.uint8)
                p_np = prob.detach().cpu().numpy().ravel()

                pos = int(t_bin.sum())
                neg = int(t_bin.size - pos)
                pos_total += pos
                neg_total += neg

                if pos > 0 and neg > 0:
                    mixed_items += 1
                    val_collected = _collect_scores(p_np, t_bin, val_probs_list, val_targets_list,
                                                    VAL_AUROC_SUBSAMPLE, val_collected)
                else:
                    skipped_single += 1

        # Aggregate val metrics
        val_mse_epoch = val_mse_sum / max(1, val_batches)
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
        y_score = np.concatenate(val_probs_list) if len(
            val_probs_list) else np.array([])
        y_true = np.concatenate(val_targets_list) if len(
            val_targets_list) else np.array([])
        have_two = (y_true.size > 0 and np.unique(y_true).size == 2)

        print(
            f"[val@{epoch}] pos={pos_total}, neg={neg_total}, mixed_items={mixed_items}, skipped={skipped_single}")

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
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title(f'Val ROC (epoch {epoch})')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(
                roc_val_dir, f'roc_val_epoch_{epoch:03d}.png'))
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.plot(rec, prec, label=f'AP={auprc:.4f}', color='tab:orange')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Val PR (epoch {epoch})')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(
                pr_val_dir, f'pr_val_epoch_{epoch:03d}.png'))
            plt.close()

            # Save best-by-AUROC / AUPRC (VAL)
            net = model_restored.module if hasattr(
                model_restored, "module") else model_restored
            if auroc > best_auroc:
                best_auroc = auroc
                best_auroc_epoch = epoch
                best_auroc_path = os.path.join(
                    model_dir, f"model_best_auroc_e{epoch:03d}.pth")

            if auprc > best_auprc:
                best_auprc = auprc
                best_auprc_epoch = epoch
                best_auprc_path = os.path.join(
                    model_dir, f"model_best_auprc_e{epoch:03d}.pth")

        else:
            auroc_hist_val.append(np.nan)
            auprc_hist_val.append(np.nan)
            print(
                f"[val] AUROC/AUPRC undefined (no mixed-class masks collected) at epoch {epoch}")

        # --- TEST (same cadence as VAL) ---
        if test_loader is not None:
            model_restored.eval()
            test_mse_sum = 0.0
            test_mseW_sum = 0.0
            test_batches = 0
            probs_list = []
            tgts_list = []
            pos_total_t = neg_total_t = 0
            mixed_items_t = skipped_single_t = 0
            collected_t = 0
            with torch.no_grad():
                for data_test in test_loader:
                    target = data_test[0].cuda()
                    input_ = data_test[1].cuda()
                    if target.shape[1] == 3:
                        target = 0.2989 * \
                            target[:, 0:1] + 0.5870 * \
                            target[:, 1:2] + 0.1140 * target[:, 2:3]
                    logits = model_restored(input_)
                    prob = torch.sigmoid(logits)
                    se = (prob - target) ** 2
                    test_mse_sum += se.mean().item()
                    w = make_weights_from_torch(
                        target, k=K_RINGS, stroke_w=STROKE_W, ring_w=RING_W,
                        normalize_to_mean_one=NORM_MEAN_ONE, bg_min=0.0, kernel_size=3
                    )
                    test_mseW_sum += (se * w).sum().item() / \
                        max(1e-8, w.sum().item())
                    test_batches += 1

                    t_np = target.detach().cpu().numpy().ravel()
                    t_bin = (t_np > 0.5).astype(np.uint8) if t_np.max(
                    ) <= 1.0 else (t_np > 127).astype(np.uint8)
                    p_np = prob.detach().cpu().numpy().ravel()
                    pos = int(t_bin.sum())
                    neg = int(t_bin.size - pos)
                    pos_total_t += pos
                    neg_total_t += neg
                    if pos > 0 and neg > 0:
                        mixed_items_t += 1
                        collected_t = _collect_scores(p_np, t_bin, probs_list, tgts_list,
                                                      TEST_AUROC_SUBSAMPLE, collected_t)
                    else:
                        skipped_single_t += 1

            test_mse_epoch = test_mse_sum / max(1, test_batches)
            test_mseW_epoch = test_mseW_sum / max(1, test_batches)
            mse_hist_test.append(test_mse_epoch)
            mseW_hist_test.append(test_mseW_epoch)
            test_epoch_list.append(epoch)
            writer.add_scalar('test/mse', test_mse_epoch, epoch)
            writer.add_scalar('test/mse_weighted', test_mseW_epoch, epoch)

            if len(tgts_list):
                y_true_t = np.concatenate(tgts_list)
                y_score_t = np.concatenate(probs_list)
                if np.unique(y_true_t).size == 2:
                    auroc_t = roc_auc_score(y_true_t, y_score_t)
                    auprc_t = average_precision_score(y_true_t, y_score_t)
                    auroc_hist_test.append(auroc_t)
                    auprc_hist_test.append(auprc_t)
                    writer.add_scalar('test/auroc', auroc_t, epoch)
                    writer.add_scalar('test/auprc', auprc_t, epoch)
                else:
                    auroc_hist_test.append(np.nan)
                    auprc_hist_test.append(np.nan)
                    print(
                        f"[test] AUROC/AUPRC undefined (one-class) at epoch {epoch}")
            else:
                auroc_hist_test.append(np.nan)
                auprc_hist_test.append(np.nan)

    # =========================
    # Per-epoch OVERLAY plots
    # =========================
    # TRAIN overlay (up to this epoch)
    xs_tr = list(range(1, len(loss_hist_tr) + 1))
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # AUROC/AUPRC on left (0..1)
    ax1.plot(xs_tr, auroc_hist_tr, marker='o',
             color='tab:blue',   label='Train AUROC')
    ax1.plot(xs_tr, auprc_hist_tr, marker='o',
             color='tab:orange', label='Train AUPRC')
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('AUROC / AUPRC')

    # Loss/MSE on right
    ax2.plot(xs_tr, loss_hist_tr, marker='^', color='tab:red',
             label='Train Loss', linestyle='-')
    ax2.plot(xs_tr, mse_hist_tr,  marker='s',
             color='tab:green',  label='Train MSE')
    ax2.plot(xs_tr, mseW_hist_tr, marker='d',
             color='tab:purple', label='Train MSE (Weighted)')
    ax2.set_ylabel('Loss / MSE')

    ax1.set_xlabel('Epoch')
    ax1.set_title('TRAIN Overlay (epoch {})'.format(epoch))
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='best')
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(
        overlay_tr_d, f'overlay_train_up_to_epoch_{epoch:03d}.png'))
    plt.close()

    # VAL overlay (only for validated epochs)
    xs_val = val_epoch_list
    if len(xs_val) > 0:
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(xs_val, auroc_hist_val, marker='o',
                 color='tab:blue',   label='Val AUROC')
        ax1.plot(xs_val, auprc_hist_val, marker='o',
                 color='tab:orange', label='Val AUPRC')
        ax1.set_ylim(0, 1.0)
        ax1.set_ylabel('AUROC / AUPRC')
        ax2.plot(xs_val, mse_hist_val,  marker='s',
                 color='tab:green',  label='Val MSE')
        ax2.plot(xs_val, mseW_hist_val, marker='d',
                 color='tab:purple', label='Val MSE (Weighted)')
        # also put train loss on same axis for epoch alignment
        tr_loss_for_val = [loss_hist_tr[e-1] for e in xs_val]
        ax2.plot(xs_val, tr_loss_for_val, marker='^',
                 color='tab:red', linestyle='--', label='Train Loss')
        ax2.set_ylabel('Loss / MSE')
        ax1.set_xlabel('Epoch')
        ax1.set_title('VAL Overlay (epoch {})'.format(epoch))
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='best')
        ax1.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(
            overlay_v_d, f'overlay_val_up_to_epoch_{epoch:03d}.png'))
        plt.close()

    # === Combined TRAIN+VAL overlay (all metrics) ===
    if len(xs_tr) > 0 and len(xs_val) > 0:
        plt.figure(figsize=(12, 7))
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        # Left axis: AUROC/AUPRC (0..1)
        ax1.plot(xs_tr,  auroc_hist_tr, marker='o',
                 color='tab:blue',   label='Train AUROC')
        ax1.plot(xs_val, auroc_hist_val, marker='o',
                 color='tab:blue',   linestyle='--', label='Val AUROC')
        ax1.plot(xs_tr,  auprc_hist_tr, marker='o',
                 color='tab:orange', label='Train AUPRC')
        ax1.plot(xs_val, auprc_hist_val, marker='o',
                 color='tab:orange', linestyle='--', label='Val AUPRC')
        ax1.set_ylim(0, 1.0)
        ax1.set_ylabel('AUROC / AUPRC')
        # Right axis: Loss / MSE / Weighted MSE
        ax2.plot(xs_tr,  loss_hist_tr, marker='^',
                 color='tab:red',    label='Train Loss')
        ax2.plot(xs_val, loss_hist_val, marker='^', color='tab:red',
                 linestyle='--', label='Val Loss')
        ax2.plot(xs_tr,  mse_hist_tr,  marker='s',
                 color='tab:green',  label='Train MSE')
        ax2.plot(xs_val, mse_hist_val, marker='s',
                 color='tab:green',  linestyle='--', label='Val MSE')
        ax2.plot(xs_tr,  mseW_hist_tr, marker='d',
                 color='tab:purple', label='Train MSE (Weighted)')
        ax2.plot(xs_val, mseW_hist_val, marker='d', color='tab:purple',
                 linestyle='--', label='Val MSE (Weighted)')
        ax2.set_ylabel('Loss / MSE')
        ax1.set_xlabel('Epoch')
        ax1.set_title(f'Train + Val Overlay (up to epoch {epoch})')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='best')
        ax1.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(
            overlay_tv_d, f'overlay_train_val_up_to_epoch_{epoch:03d}.png'))
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
        plt.plot(xs_tr, auroc_hist_tr,  marker='o',
                 linestyle='-',  color=C_TR, label='Train AUROC')
        plt.plot(xs_tr, auprc_hist_tr,  marker='s',
                 linestyle='--', color=C_TR, label='Train AUPRC')
        # Val (red)
        plt.plot(xs_val, auroc_hist_val, marker='o',
                 linestyle='-',  color=C_VA, label='Val AUROC')
        plt.plot(xs_val, auprc_hist_val, marker='s',
                 linestyle='--', color=C_VA, label='Val AUPRC')
        # Test (green)
        plt.plot(xs_te, auroc_hist_test, marker='o',
                 linestyle='-',  color=C_TE, label='Test AUROC')
        plt.plot(xs_te, auprc_hist_test, marker='s',
                 linestyle='--', color=C_TE, label='Test AUPRC')

        plt.ylim(0, 1.0)
        plt.xlabel('Epoch')
        plt.ylabel('Score (higher is better)')
        plt.title(f'AUROC & AUPRC (Train/Val/Test) — up to epoch {epoch}')
        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(overlay_tvt_d,
                    f'high_metrics_up_to_epoch_{epoch:03d}.png'))
        plt.close()

        # -------- Low-is-good: Loss, MSE, Weighted MSE --------
        plt.figure(figsize=(12, 7))
        # Train (blue)
        plt.plot(xs_tr, loss_hist_tr,  marker='^',
                 linestyle='-',  color=C_TR, label='Train Loss')
        plt.plot(xs_tr, mse_hist_tr,   marker='d',
                 linestyle='-.', color=C_TR, label='Train MSE')
        plt.plot(xs_tr, mseW_hist_tr,  marker='x', linestyle=':',
                 color=C_TR, label='Train MSE (W)')
        # Val (red)
        plt.plot(xs_val, loss_hist_val,  marker='^',
                 linestyle='-',  color=C_VA, label='Val Loss')
        plt.plot(xs_val, mse_hist_val,   marker='d',
                 linestyle='-.', color=C_VA, label='Val MSE')
        plt.plot(xs_val, mseW_hist_val,  marker='x',
                 linestyle=':',  color=C_VA, label='Val MSE (W)')
        # Test (green) — typically we don’t track test *loss*, so only MSEs:
        plt.plot(xs_te, mse_hist_test,   marker='d',
                 linestyle='-.', color=C_TE, label='Test MSE')
        plt.plot(xs_te, mseW_hist_test,  marker='x',
                 linestyle=':',  color=C_TE, label='Test MSE (W)')

        plt.xlabel('Epoch')
        plt.ylabel('Loss / Error (lower is better)')
        plt.title(
            f'Loss, MSE, Weighted MSE (Train/Val/Test) — up to epoch {epoch}')
        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(overlay_tvt_d,
                    f'low_metrics_up_to_epoch_{epoch:03d}.png'))
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
        epoch, time.time() -
        epoch_start_time, epoch_loss, scheduler.get_last_lr()[0]
    ))
    if len(val_epoch_list) and val_epoch_list[-1] == epoch:
        i = len(val_epoch_list) - 1
        v_mse, v_mseW = mse_hist_val[i], mseW_hist_val[i]
        v_auroc, v_auprc = auroc_hist_val[i], auprc_hist_val[i]
        print(
            f"[val@{epoch}] MSE={v_mse:.6f}  MSEw={v_mseW:.6f}  AUROC={v_auroc:.6f}  AUPRC={v_auprc:.6f}")
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
plt.plot(epochs_tr, loss_hist_tr, marker='o',
         label='Train Loss', color='tab:red')
if len(loss_hist_val):
    plt.plot(val_epoch_list, loss_hist_val, marker='o',
             label='Val Loss', color='tab:pink')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(loss_dir, 'train_val_loss.png'))
plt.close()

# =========================
# Save numeric logs
# =========================
metrics_txt = os.path.join(log_dir, 'metrics_per_epoch.txt')
with open(metrics_txt, 'w') as f:
    f.write('Epoch\tTrain_Loss\tVal_Loss\tTrain_MSE\tTrain_MSEw\tVal_MSE\tVal_MSEw\tTrain_AUROC\tTrain_AUPRC\tVal_AUROC\tVal_AUPRC\tTest_MSE\tTest_MSEw\tTest_AUROC\tTest_AUPRC\n')
    for ep in epochs_tr:
        idx = ep - 1
        vloss = vmse = vmsew = vauroc = vauprc = ''
        if ep in val_epoch_list:
            i = val_epoch_list.index(ep)
            vloss = f'{loss_hist_val[i]:.6f}'
            vmse = f'{mse_hist_val[i]:.6f}'
            vmsew = f'{mseW_hist_val[i]:.6f}'
            vauroc = f'{auroc_hist_val[i]:.6f}' if not np.isnan(
                auroc_hist_val[i]) else ''
            vauprc = f'{auprc_hist_val[i]:.6f}' if not np.isnan(
                auprc_hist_val[i]) else ''
        # test columns (only when that epoch had test eval)
        t_mse = t_msew = t_roc = t_pr = ''
        if ep in test_epoch_list:
            j = test_epoch_list.index(ep)
            t_mse = f'{mse_hist_test[j]:.6f}'
            t_msew = f'{mseW_hist_test[j]:.6f}'
            t_roc = f'{auroc_hist_test[j]:.6f}' if not np.isnan(
                auroc_hist_test[j]) else ''
            t_pr = f'{auprc_hist_test[j]:.6f}' if not np.isnan(
                auprc_hist_test[j]) else ''
        tr_auroc = f'{auroc_hist_tr[idx]:.6f}' if not np.isnan(
            auroc_hist_tr[idx]) else ''
        tr_auprc = f'{auprc_hist_tr[idx]:.6f}' if not np.isnan(
            auprc_hist_tr[idx]) else ''
        f.write(f'{ep}\t{loss_hist_tr[idx]:.6f}\t{vloss}\t{mse_hist_tr[idx]:.6f}\t{mseW_hist_tr[idx]:.6f}\t{vmse}\t{vmsew}\t{tr_auroc}\t{tr_auprc}\t{vauroc}\t{vauprc}\t{t_mse}\t{t_msew}\t{t_roc}\t{t_pr}\n')

# =========================
# Print best checkpoints (by VAL)
# =========================
print("\n==================== Best checkpoints (by VAL) ====================")
if best_auroc_epoch is not None:
    print(
        f"Best AUROC : {best_auroc:.6f} at epoch {best_auroc_epoch} -> {best_auroc_path}")
else:
    print("Best AUROC : (not available; AUROC was undefined for all val epochs)")
if best_auprc_epoch is not None:
    print(
        f"Best AUPRC : {best_auprc:.6f} at epoch {best_auprc_epoch} -> {best_auprc_path}")
else:
    print("Best AUPRC : (not available; AUPRC was undefined for all val epochs)")
print("==========================================================\n")
