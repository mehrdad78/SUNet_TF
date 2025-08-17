# --- Standard library ---
import os
import time
import random
import yaml

# --- Third-party libraries ---
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.ndimage import distance_transform_edt
from tensorboardX import SummaryWriter
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from skimage.morphology import binary_dilation
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    accuracy_score
)

# --- Local project imports ---
from model.SUNet import SUNet_model
from data_RGB import get_training_data, get_validation_data
import utils
from utils import network_parameters


# =========================
# Config
# =========================
# Boundary-weight settings
K_RINGS   = 2
STROKE_W  = 3.0
RING_W    = (3.0, 2.0, 1.0)  # weights for the successive rings
NORM_MEAN_ONE = True         # normalize weight maps to mean 1

# =========================
# Repro
# =========================
torch.backends.cudnn.benchmark = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# =========================
# Load config
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
val_dir   = Train['VAL_DIR']

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
    """
    binary_image: (H,W) uint8/bool (0/1)
    k: number of dilation steps -> returns exactly k rings
    """
    if footprint is None:
        footprint = np.ones((3, 3), dtype=bool)  # 8-neighborhood
    prev = (binary_image > 0).astype(np.uint8)
    neigh_masks = []
    for _ in range(k):  # k rings
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

def make_weights_from_numpy(
    target_t: torch.Tensor,
    k: int = K_RINGS,
    stroke_w: float = STROKE_W,
    ring_w = RING_W,
    normalize_to_mean_one: bool = NORM_MEAN_ONE,
    bg_min: float = 0.0,
) -> torch.Tensor:
    """target_t: (B,1,H,W) float on GPU; returns (B,1,H,W) float on same device."""
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
        weights_list.append(w_np[None, None, ...])  # (1,1,H,W)

    w_np_batch = np.concatenate(weights_list, axis=0)  # (B,1,H,W)
    w = torch.from_numpy(w_np_batch).to(device=device, dtype=target_t.dtype)

    # Guard empty-foreground patch
    if float(w.sum()) == 0.0:
        w.fill_(1.0)

    if normalize_to_mean_one:
        w = w / w.mean().clamp(min=1e-8)

    return w

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

for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0.0

    # --- Train ---
    model_restored.train()
    train_mse_running = 0.0
    train_mse_batches = 0

    for i, data in enumerate(tqdm(train_loader), 0):
        # zero grad (faster than optimizer.zero_grad())
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

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Per-epoch train MSE
    mse_train_epoch = train_mse_running / max(1, train_mse_batches)
    mse_train_history.append(mse_train_epoch)
    writer.add_scalar('train/mse', mse_train_epoch, epoch)

    # --- Validation ---
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model_restored.eval()
        val_mse_running = 0.0
        val_mse_batches = 0
        val_mse_weighted_running = 0.0
        val_epoch_loss = 0.0

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

        # aggregate & log per-epoch
        val_mse_epoch = val_mse_running / max(1, val_mse_batches)
        mse_val_history.append(val_mse_epoch)
        writer.add_scalar('val/mse', val_mse_epoch, epoch)

        val_mse_weighted_epoch = val_mse_weighted_running / max(1, val_mse_batches)
        mse_val_weighted_history.append(val_mse_weighted_epoch)
        writer.add_scalar('val/mse_weighted', val_mse_weighted_epoch, epoch)

        val_loss_history.append(val_epoch_loss / max(1, len(val_loader)))
        val_epoch_list.append(epoch)

    # --- Scheduler & checkpoints ---
    scheduler.step()

    # save "latest" (single clean block)
    torch.save({
        'epoch': epoch,
        'state_dict': (model_restored.module if hasattr(model_restored, "module") else model_restored).state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, os.path.join(model_dir, "model_latest.pth"))

    # milestone save @ epoch 5 (adjust as you like)
    if epoch == 5 or epoch == 4:
        torch.save({
            'epoch': epoch,
            'state_dict': (model_restored.module if hasattr(model_restored, "module") else model_restored).state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, os.path.join(model_dir, f"model_epoch_{epoch:01d}.pth"))

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
# Plots (Loss + MSE)
# =========================
epochs_train = list(range(1, len(loss_history) + 1))

# Train vs Val MSE
plt.figure(figsize=(10, 6))
plt.plot(epochs_train[:len(mse_train_history)], mse_train_history, marker='o', label='Train MSE')
plt.plot(val_epoch_list, mse_val_history, marker='o', label='Val MSE')
plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.title('Train/Val MSE per Epoch')
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(log_dir, 'mse_curves.png'))
plt.show()

# Weighted Val MSE
plt.figure(figsize=(10, 6))
plt.plot(val_epoch_list, mse_val_weighted_history, marker='o', label='Val MSE (Weighted)')
plt.xlabel('Epoch'); plt.ylabel('MSE (Weighted)'); plt.title('Boundary-Weighted Val MSE per Epoch')
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(log_dir, 'mse_weighted_curve.png'))
plt.show()

# Train & Val Loss per epoch
plt.figure(figsize=(10, 6))
plt.plot(epochs_train, loss_history, marker='o', label='Training Loss', color='blue')

if val_loss_history:
    plt.plot(val_epoch_list, val_loss_history, marker='o', color='red', label='Validation Loss')

plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training and Validation Loss per Epoch')
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(log_dir, 'train_val_loss_with_values.png'))
plt.show()

# =========================
# Save numeric logs
# =========================
loss_txt_path = os.path.join(log_dir, 'train_val_loss_values.txt')
with open(loss_txt_path, 'w') as f:
    f.write('Epoch\tTrain_Loss\tVal_Loss\n')
    for i, ep in enumerate(epochs_train):
        val_loss_str = ''
        if ep in val_epoch_list:
            idx = val_epoch_list.index(ep)
            val_loss_str = f'{val_loss_history[idx]:.6f}'
        f.write(f'{ep}\t{loss_history[i]:.6f}\t{val_loss_str}\n')
