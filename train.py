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

# =========================
# Settings you can tweak
# =========================
# ---- color/style palette (split-based colors) ----
SPLIT_COLOR = {'train':'tab:blue','val':'tab:red','test':'tab:green'}
# optional: markers & linestyles so different metrics remain distinguishable
MARK = {'auroc':'o', 'auprc':'x', 'loss':'^', 'mse':'s', 'mse_w':'d'}
STYLE = {'train':'-', 'val':'--', 'test':':'}

# Boundary-weight settings
K_RINGS = 2
STROKE_W = 3.0
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
# === اینجا اضافه کن ===

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
# Directory for weight visualizations
weights_dir = os.path.join(plots_root, 'weights')
os.makedirs(weights_dir, exist_ok=True)


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
# ========= Heatmap hooks for ALL intermediate tensors (2D heatmaps) =========
import os, math, re, gc
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ------------ Configs ------------
ACT_OUT_DIR = os.path.join(model_dir, "activations_heatmaps")
os.makedirs(ACT_OUT_DIR, exist_ok=True)

# کاهش کانال‌ها به 2D: 'mean' یا 'max' یا 'first'
REDUCE_TO_2D = 'mean'
# چند کانال «برگزیده» علاوه بر نقشه‌ی میانگین ذخیره شود؟ (براساس واریانس)
TOPK_CHANNELS = 4
# فقط از اولین batch خروجی بگیر؟
ONLY_FIRST_BATCH = True
# محدودیت تعداد لایه‌ها (0 یعنی بدون محدودیت)
MAX_LAYERS = 0

# چه ماژول‌هایی را مانیتور کنیم؟
MONITORED_TYPES = (
    nn.Conv2d, nn.BatchNorm2d,
    nn.ReLU, nn.LeakyReLU, nn.SiLU, nn.ELU, nn.GELU,
    nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
    nn.Upsample, nn.ConvTranspose2d,
)

# ------------ Helpers ------------
def _sanitize(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.\-]+', '_', name)

def _to_numpy_2d(t: torch.Tensor) -> np.ndarray:
    """
    ورودی می‌تواند 2D/3D/4D باشد:
      - (H,W) -> همان
      - (C,H,W) یا (B,C,H,W) -> کاهش به 2D با REDUCE_TO_2D
    """
    with torch.no_grad():
        t = t.detach()
        # اگر NaN/Inf باشد، صفر کنیم تا رسم خراب نشود
        t = torch.where(torch.isfinite(t), t, torch.zeros_like(t))
        if t.dim() == 2:
            arr = t.float().cpu().numpy()
            return arr
        if t.dim() == 3:
            C, H, W = t.shape
            if REDUCE_TO_2D == 'mean':
                return t.mean(0).float().cpu().numpy()
            elif REDUCE_TO_2D == 'max':
                return t.max(0).values.float().cpu().numpy()
            else:  # 'first'
                return t[0].float().cpu().numpy()
        if t.dim() == 4:
            B, C, H, W = t.shape
            x = t[0]  # اولین نمونه‌ی batch
            if REDUCE_TO_2D == 'mean':
                return x.mean(0).float().cpu().numpy()
            elif REDUCE_TO_2D == 'max':
                return x.max(0).values.float().cpu().numpy()
            else:  # 'first'
                return x[0].float().cpu().numpy()
        # سایر ابعاد را سعی می‌کنیم به آخرین دو بعد کاهش دهیم
        v = t.flatten(start_dim=0, end_dim=t.dim()-3)
        v = v.mean(0)  # میانگین همه ابعاد اضافه
        return v.float().cpu().numpy()

def _save_heatmap(arr2d: np.ndarray, out_path: str, title: str = None):
    plt.figure(figsize=(5,5))
    plt.imshow(arr2d, cmap='hot', interpolation='nearest')  # مثل عکس نمونه
    plt.colorbar(label='Value')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _save_topk_channels(x: torch.Tensor, layer_name: str, step_tag: str):
    """
    علاوه بر نقشه‌ی میانگین، K کانالِ با واریانس بیشتر را هم ذخیره می‌کند (کمک به دیدن جزییات).
    فقط وقتی ابعاد 4D/3D باشد اجرا می‌شود.
    """
    with torch.no_grad():
        if x.dim() == 4:    # (B,C,H,W)
            x0 = x[0]
        elif x.dim() == 3:  # (C,H,W)
            x0 = x
        else:
            return
        C = x0.size(0)
        if C <= 1 or TOPK_CHANNELS <= 0: return
        # واریانس هر کانال
        var = x0.float().flatten(1).var(dim=1)  # [C]
        k = min(TOPK_CHANNELS, C)
        topk = torch.topk(var, k=k).indices.tolist()
        for rank, ch in enumerate(topk):
            arr = x0[ch].detach().float().cpu().numpy()
            outp = os.path.join(ACT_OUT_DIR, f"{_sanitize(layer_name)}__{step_tag}__ch{ch:04d}_r{rank+1}.png")
            _save_heatmap(arr, outp, title=f"{layer_name}  ch={ch} (rank {rank+1})")

# ------------ Hook machinery ------------
_hooks = []
_seen_a_batch = False
_layer_counter = 0

def _hook_factory(layer_name: str):
    def _hook(module, input, output):
        global _seen_a_batch, _layer_counter
        if ONLY_FIRST_BATCH and _seen_a_batch:
            return
        if MAX_LAYERS and _layer_counter >= MAX_LAYERS:
            return

        # بعضی لایه‌ها tuple یا dict خروجی می‌دهند
        out = output
        if isinstance(out, (tuple, list)):
            if len(out) == 0: return
            out = out[0]
        if isinstance(out, dict):
            # اگر کلید معمولی داشت مثل 'out' یا 'attn' برداریم
            out = out.get('out', list(out.values())[0])

        try:
            # ذخیره نقشه‌ی میانگین/انتخابی به 2D
            arr2d = _to_numpy_2d(out)
            step_tag = "step0000000"  # برای forwardهای تک‌مرحله‌ای
            fname = f"{_sanitize(layer_name)}__{step_tag}.png"
            _save_heatmap(arr2d, os.path.join(ACT_OUT_DIR, fname), title=layer_name)

            # ذخیره‌ی کانال‌های برگزیده
            _save_topk_channels(out, layer_name, step_tag)

            _layer_counter += 1
        except Exception as e:
            print(f"[heatmap-hook] skip {layer_name}: {e}")
    return _hook

def register_heatmap_hooks(model: nn.Module):
    for name, m in model.named_modules():
        if isinstance(m, MONITORED_TYPES):
            _hooks.append(m.register_forward_hook(_hook_factory(name)))
    print(f"[heatmap] registered {len(_hooks)} hooks; output dir = {ACT_OUT_DIR}")

def remove_heatmap_hooks():
    global _hooks
    for h in _hooks:
        try: h.remove()
        except: pass
    _hooks = []
    gc.collect()
    print("[heatmap] hooks removed.")
# ============================================================================


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
    # DEBUG: print once per process to verify binarization/normalization
    if not hasattr(make_weights_from_numpy, "_dbg_count"):
        make_weights_from_numpy._dbg_count = 0
    if make_weights_from_numpy._dbg_count < 3:
        make_weights_from_numpy._dbg_count += 1
        with torch.no_grad():
        # reconstruct the internal binarization used
            tgt_np = target_t.detach().cpu().numpy()
            if tgt_np.max() <= 1.0:
                bin_batch = (tgt_np > 0.5).astype(np.uint8)
            else:
                bin_batch = (tgt_np > 127).astype(np.uint8)
            fg_ratio_batch = bin_batch.mean()
            print("[make_weights_from_numpy:DEBUG]",
              f"shape={target_t.shape} dtype={target_t.dtype} max={target_t.max().item():.3f}",
              f"fg_ratio~={fg_ratio_batch:.4f}",
              f"w_min={w.min().item():.4f} w_mean={w.mean().item():.4f} w_max={w.max().item():.4f}",
              f"norm_mean_one={normalize_to_mean_one}")

    return w
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import os

def debug_plot_weighting(target_tensor, save_dir, name="sample",
                         k=K_RINGS, stroke_w=STROKE_W, ring_w=RING_W):
    """
    target_tensor: (1,1,H,W) torch tensor (یک تصویر)
    save_dir: فولدر خروجی (مثلاً weights_dir)
    name: اسم فایل خروجی
    """
    os.makedirs(save_dir, exist_ok=True)

    tgt_np = target_tensor.squeeze().cpu().numpy()
    if tgt_np.max() <= 1.0:
        bin_img = (tgt_np > 0.5).astype(np.uint8)
    else:
        bin_img = (tgt_np > 127).astype(np.uint8)

    # Step 1: ساخت رینگ‌ها
    masks = background_adjacent_to_foreground(bin_img, k)

    # Step 2: وزن نهایی
    w_np = make_weight_matrix(
        bin_img, masks, stroke_w=float(stroke_w), masks_w=list(ring_w)
    ).astype(np.float32)

    # Step 3: پلات کل پروسه
    cols = 3 + len(masks)  # Target, Binarized, Rings..., Final
    fig, axes = plt.subplots(1, cols, figsize=(4*cols, 4))

    axes[0].imshow(tgt_np, cmap="gray")
    axes[0].set_title("Target (raw)")
    axes[0].axis("off")

    axes[1].imshow(bin_img, cmap="gray")
    axes[1].set_title("Binarized")
    axes[1].axis("off")

    for i, ring in enumerate(masks):
        axes[2+i].imshow(ring, cmap="gray")
        axes[2+i].set_title(f"Ring {i+1}")
        axes[2+i].axis("off")

    im = axes[-1].imshow(w_np, cmap="magma")
    axes[-1].set_title("Final Weights")
    axes[-1].axis("off")
    plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # ذخیره در فایل
    out_path = os.path.join(save_dir, f"weight_debug_{name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"✅ Weighting debug plot saved: {out_path}")


sample = next(iter(train_loader))
target = sample[0][0:1].cuda()  # فقط یک تصویر
debug_plot_weighting(target, save_dir=weights_dir, name="train_example")


def _to_gray_if_rgb(t):
    """Convert RGB (B,3,H,W) to grayscale, else return unchanged."""
    if t.size(1) == 1:
        return t
    r, g, b = t[:,0:1], t[:,1:2], t[:,2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

@torch.no_grad()
def plot_some_weight_maps(loader, num_batches=2, max_per_batch=3,
                          k=K_RINGS, stroke_w=STROKE_W, ring_w=RING_W,
                          normalize_to_mean_one=NORM_MEAN_ONE):
    """
    Visualize Input / Target / Weight map for a few samples.
    loader: DataLoader (train_loader or val_loader)
    num_batches: how many batches to visualize
    max_per_batch: how many samples per batch
    """
    shown = 0
    for bi, batch in enumerate(loader):
        if shown >= num_batches:
            break

        target = batch[0].cuda()   # (B,C,H,W)
        input_  = batch[1].cuda()

        # Make single-channel targets (same as in training)
        target_gray = _to_gray_if_rgb(target)
        # If dataset gives [0,255], normalize:
        if target_gray.max() > 1.0:
            target_gray = target_gray / 255.0

        # Build weights from TARGET
        weights = make_weights_from_numpy(
            target_gray, k=k, stroke_w=stroke_w, ring_w=ring_w,
            normalize_to_mean_one=normalize_to_mean_one
        )

        B = min(input_.size(0), max_per_batch)
        for i in range(B):
            inp_vis = _to_gray_if_rgb(input_[i:i+1]).squeeze().cpu().numpy()
            tgt_vis = target_gray[i:i+1].squeeze().cpu().numpy()
            w_vis   = weights[i:i+1].squeeze().cpu().numpy()

            w_min, w_mean, w_max = float(w_vis.min()), float(w_vis.mean()), float(w_vis.max())

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            axes[0].imshow(inp_vis, cmap='gray')
            axes[0].set_title('Input')
            axes[0].axis('off')

            axes[1].imshow(tgt_vis, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title('Target (mask)')
            axes[1].axis('off')

            im2 = axes[2].imshow(w_vis, cmap='magma')
            axes[2].set_title(f'Weights\nmin={w_min:.2f} mean={w_mean:.2f} max={w_max:.2f}')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            plt.tight_layout()
            fname = os.path.join(weights_dir, f'weights_b{bi:03d}_i{i:02d}.png')
            plt.savefig(fname, dpi=150)
            plt.close(fig)

        shown += 1

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

def _fg_ratio(t: torch.Tensor) -> float:
    """Fraction of foreground pixels in target (assumes 0..1; if not, we clamp test below)."""
    with torch.no_grad():
        t_cpu = t.detach().float().cpu()
        if t_cpu.max() > 1.0:  # not normalized
            t_cpu = (t_cpu > 127).float()
        else:
            t_cpu = (t_cpu > 0.5).float()
        return float(t_cpu.mean())

def _tensor_stats(x: torch.Tensor):
    x = x.detach().float().cpu()
    # Replace NaNs with 0 before computing stats
    x_clean = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    return {
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "min": float(x_clean.min().item()) if x.numel() else float('nan'),
        "max": float(x_clean.max().item()) if x.numel() else float('nan'),
        "mean": float(x_clean.mean().item()) if x.numel() else float('nan'),
        "sum": float(x_clean.sum().item()) if x.numel() else float('nan'),
        "finite": bool(torch.isfinite(x).all())
    }


import json
import matplotlib.pyplot as plt

def _save_weight_preview(save_dir, tag_prefix, step, target, weights):
    """Save a small 2-panel preview: target (grayscale) and weights (magma)."""
    os.makedirs(save_dir, exist_ok=True)
    # pick the first sample in batch for a consistent preview
    tgt = target[0, 0].detach().float().cpu().numpy()
    w   = weights[0, 0].detach().float().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(tgt, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Target (grayscale)')
    axes[0].axis('off')

    im = axes[1].imshow(w, cmap='magma')
    axes[1].set_title('Weights')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fname = os.path.join(save_dir, f"{tag_prefix.replace('/','_')}_step{step:07d}.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)


def log_weight_debug(writer, tag_prefix, step, target, weights, save_dir=None, save_preview=True):
    """Logs to TensorBoard; optionally saves JSON stats + preview PNG to checkpoints."""
    fg = _fg_ratio(target)
    wstats = _tensor_stats(weights)
    tstats = _tensor_stats(target)

    # --- TensorBoard scalars ---
    writer.add_scalar(f'{tag_prefix}/target_fg_ratio', fg, step)
    writer.add_scalar(f'{tag_prefix}/weights_mean', wstats["mean"], step)
    writer.add_scalar(f'{tag_prefix}/weights_min',  wstats["min"],  step)
    writer.add_scalar(f'{tag_prefix}/weights_max',  wstats["max"],  step)
    writer.add_scalar(f'{tag_prefix}/weights_sum',  wstats["sum"],  step)

    # --- TensorBoard histograms ---
    writer.add_histogram(f'{tag_prefix}/weights_hist', weights.detach().cpu(), step)
    writer.add_histogram(f'{tag_prefix}/target_hist',  target.detach().cpu(),  step)

    if step % 50 == 0:
        print(f"[{tag_prefix}@{step}] target stats={tstats}  weights stats={wstats}  fg_ratio={fg:.4f}")

    # --- Save to checkpoints folder (JSON + optional preview PNG) ---
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, f"weights_debug_{tag_prefix.replace('/','_')}_step{step:07d}.json")
        with open(json_path, "w") as f:
            json.dump({
                "tag": tag_prefix,
                "step": int(step),
                "target_stats": tstats,
                "weight_stats": wstats,
                "fg_ratio": float(fg),
            }, f, indent=2)

        if save_preview:
            _save_weight_preview(save_dir, tag_prefix, step, target, weights)
import glob
import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_weight_stats_timeseries(model_dir, out_subdir="weights_plots"):
    """
    Scans model_dir for weights_debug_*.json files and plots:
      - weights_mean / min / max / sum over step
      - fg_ratio over step
    Groups by 'tag' (e.g., train/weights vs val/weights).
    """
    json_paths = glob.glob(os.path.join(model_dir, "weights_debug_*_step*.json"))
    if not json_paths:
        print(f"[plot_weight_stats_timeseries] No JSON files found in {model_dir}")
        return

    # Collect per tag
    series = defaultdict(lambda: {"step": [], "w_mean": [], "w_min": [], "w_max": [], "w_sum": [], "fg": []})

    step_re = re.compile(r".*_step(\d+)\.json$")
    for p in json_paths:
        try:
            with open(p, "r") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"Skip {p}: {e}")
            continue

        tag   = obj.get("tag", "unknown")
        stepm = step_re.match(p)
        step  = int(obj.get("step")) if obj.get("step") is not None else (int(stepm.group(1)) if stepm else 0)

        wstats = obj.get("weight_stats", {})
        series[tag]["step"].append(step)
        series[tag]["w_mean"].append(float(wstats.get("mean", float('nan'))))
        series[tag]["w_min"].append(float(wstats.get("min",  float('nan'))))
        series[tag]["w_max"].append(float(wstats.get("max",  float('nan'))))
        series[tag]["w_sum"].append(float(wstats.get("sum",  float('nan'))))
        series[tag]["fg"].append(float(obj.get("fg_ratio", float('nan'))))

    # Output dir
    out_dir = os.path.join(model_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # Plot per tag
    for tag, d in series.items():
        # sort by step
        idx = sorted(range(len(d["step"])), key=lambda i: d["step"][i])
        xs  = [d["step"][i] for i in idx]

        def _plot_one(ys, title, fname, ylabel):
            plt.figure(figsize=(10, 5))
            plt.plot(xs, [ys[i] for i in idx], marker='o')
            plt.xlabel("Step")
            plt.ylabel(ylabel)
            plt.title(f"{title} — {tag}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, fname), dpi=150)
            plt.close()

        _plot_one(d["w_mean"], "Weights Mean", f"{tag.replace('/','_')}_weights_mean.png", "Mean")
        _plot_one(d["w_min"],  "Weights Min",  f"{tag.replace('/','_')}_weights_min.png",  "Min")
        _plot_one(d["w_max"],  "Weights Max",  f"{tag.replace('/','_')}_weights_max.png",  "Max")
        _plot_one(d["w_sum"],  "Weights Sum",  f"{tag.replace('/','_')}_weights_sum.png",  "Sum")
        _plot_one(d["fg"],     "Foreground Ratio", f"{tag.replace('/','_')}_fg_ratio.png", "Foreground Ratio")

    # Combined (all tags on one chart) for fast comparison
    def _plot_combined(key, title, ylabel, fname):
        plt.figure(figsize=(11, 6))
        for tag, d in series.items():
            idx = sorted(range(len(d["step"])), key=lambda i: d["step"][i])
            xs  = [d["step"][i] for i in idx]
            ys  = [d[key][i] for i in idx]
            plt.plot(xs, ys, marker='o', label=tag)
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()

    _plot_combined("w_mean", "Weights Mean (all tags)", "Mean", "combined_weights_mean.png")
    _plot_combined("w_min",  "Weights Min (all tags)",  "Min",  "combined_weights_min.png")
    _plot_combined("w_max",  "Weights Max (all tags)",  "Max",  "combined_weights_max.png")
    _plot_combined("w_sum",  "Weights Sum (all tags)",  "Sum",  "combined_weights_sum.png")
    _plot_combined("fg",     "Foreground Ratio (all tags)", "Foreground Ratio", "combined_fg_ratio.png")

    print(f"✅ Plots written to: {out_dir}")



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
plot_some_weight_maps(train_loader, num_batches=1, max_per_batch=2)
plot_some_weight_maps(val_loader,   num_batches=1, max_per_batch=2)

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
    # یک batch نمونه برای عبور از مدل
 # اجرای forward -> همه هوک‌ها فعال می‌شن

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
        # Log once per N steps to avoid spam (adjust N as you like)
        

        # Safety checks (catch silent failures)
        assert weights.shape == target.shape, f"weights {weights.shape} vs target {target.shape}"
        assert torch.isfinite(weights).all(), "NaN/Inf in weights"
        assert weights.sum() > 0, "weights sum == 0 (all zeros?)"

        # NOTE: using logits in Charbonnier is fine with eps
        loss = charbonnier_loss(logits, target, weight=weights, eps=1e-3)

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
        epoch_loss = epoch_loss+loss.item()

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
            ii = 0
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
                # after weights = ...
                ii += 1

                val_mseW_sum += (se * weights).sum().item() / max(1e-8, weights.sum().item())

                val_loss = charbonnier_loss(logits, target, weight=weights, eps=1e-3)
                val_epoch_loss = val_epoch_loss+val_loss.item()
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
            register_heatmap_hooks(model_restored)
            test_mse_sum = 0.0; test_mseW_sum = 0.0; test_batches = 0
            probs_list = []; tgts_list = []
            pos_total_t = neg_total_t = 0; mixed_items_t = skipped_single_t = 0
            collected_t = 0
            # یک بتچ از train_loader یا val_loader بگیر

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
plot_weight_stats_timeseries(model_dir)
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
