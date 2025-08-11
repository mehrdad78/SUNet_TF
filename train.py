import os
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import utils
import numpy as np
import random
from data_RGB import get_training_data, get_validation_data
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model.SUNet import SUNet_model
from utils import network_parameters
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, accuracy_score
from scipy.ndimage import distance_transform_edt


# Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Load yaml configuration file
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']

# Build Model
print('==> Build the model')
model_restored = SUNet_model(opt)
p_number = network_parameters(model_restored)
model_restored.cuda()

# Training model path direction
mode = opt['MODEL']['MODE']

model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
utils.mkdir(model_dir)
train_dir = Train['TRAIN_DIR']
val_dir = Train['VAL_DIR']

# GPU
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

# Optimizer
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_restored.parameters(),
                       lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

# Scheduler (Strategy)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(
    optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

# Resume (Continue training by a pretrained model)
if Train['RESUME']:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restored, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------')

# Loss
loss_history = []

# DataLoaders
print('==> Loading datasets')
train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS']})
train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                          shuffle=True, num_workers=0, drop_last=False)
val_dataset = get_validation_data(val_dir, {'patch_size': Train['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
                        drop_last=False)

# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
    Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
    Model parameters:   {p_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}
    GPU:                {'GPU' + str(device_ids)}''')
print('------------------------------------------------------------------')

# Start training!
print('==> Training start: ')
best_psnr = 0
best_ssim = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
total_start_time = time.time()
psnr_history = []
ssim_history = []
val_loss_history = []
accuracy_history = []
precision_history = []
val_epoch_list = []

all_val_preds = []
all_val_targets = []
tp_history = []
fp_history = []

def charbonnier_loss(pred, target, weight=None, eps=1e-3):
    diff = pred - target
    l = torch.sqrt(diff * diff + eps * eps)
    if weight is None:
        return l.mean()
    return (l * weight).sum() / weight.sum().clamp(min=1e-8)
def get_last_trainable_leaf(model):
    last_name, last_mod = None, None
    for name, m in (model.named_modules()):
        # leaf = no children; has parameters directly
        if sum(1 for _ in m.children()) == 0 and any(p.requires_grad for p in m.parameters(recurse=False)):
            last_name, last_mod = name, m
    return last_name, last_mod


'''
def charbonnier_loss(pred, target, weight=None, eps=1e-3, reduction='mean'):
    diff = pred - target
    loss = torch.sqrt(diff * diff + eps * eps)
    if weight is not None:
        loss = loss * weight
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss
'''


from skimage.morphology import binary_dilation


def background_adjacent_to_foreground(binary_image, k, footprint=None):
    """
    binary_image: (H,W) uint8/bool (0/1)
    k: number of dilation *steps*
    RETURNS: list of ring masks. 
      - If you want EXACTLY k rings, loop range(k).
      - If you prefer your original behavior (k+1 rings), loop range(k+1).
    """
    if footprint is None:
        footprint = np.ones((3, 3), dtype=bool)  # 8-neighborhood

    prev = (binary_image > 0).astype(np.uint8)
    neigh_masks = []
    for _ in range(k):  # <- change to range(k+1) if you want k+1 rings (your original)
        dil = binary_dilation(prev.astype(bool), footprint=footprint).astype(np.uint8)
        ring = (dil - prev).astype(bool)
        neigh_masks.append(ring)
        prev = dil
    return neigh_masks

def make_weight_matrix(binary_image, masks, stroke_w=4.0, masks_w=(4.0, 1.0, 1.0), bg_min=0.0):
    """
    weights: float32. Foreground gets stroke_w; ring i gets masks_w[i] (or last if i >= len).
    """
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
    k: int = 2,
    stroke_w: float = 3.0,
    ring_w=(3.0, 2.0, 1.0),
    normalize_to_mean_one: bool = True,
    bg_min: float = 0.0,  # set >0.0 if you want background to have tiny weight
) -> torch.Tensor:

    assert target_t.dim() == 4 and target_t.size(1) == 1, "expect (B,1,H,W)"
    device = target_t.device

    # robust binary (works for 0/1 or 0/255 just in case)
    tgt_np = target_t.detach().cpu().numpy()
    if tgt_np.max() <= 1.0:
        bin_batch = (tgt_np > 0.5).astype(np.uint8)
    else:
        bin_batch = (tgt_np > 127).astype(np.uint8)

    weights_list = []
    B = bin_batch.shape[0]
    for b in range(B):
        bin_img = bin_batch[b, 0]  # (H,W)
        masks = background_adjacent_to_foreground(bin_img, k)
        w_np = make_weight_matrix(bin_img, masks, stroke_w=float(stroke_w), masks_w=list(ring_w)).astype(np.float32)
        if bg_min > 0.0:
            w_np[w_np == 0] = bg_min
        weights_list.append(w_np[None, None, ...])  # (1,1,H,W)

    w_np_batch = np.concatenate(weights_list, axis=0)  # (B,1,H,W)
    w = torch.from_numpy(w_np_batch).to(device=device, dtype=target_t.dtype)

    # handle empty-foreground patches
    if float(w.sum()) == 0.0:
        w.fill_(1.0)

    if normalize_to_mean_one:
        w = w / w.mean().clamp(min=1e-8)

    return w

'''
@torch.no_grad()
def dilate_binary_torch(bin_img: torch.Tensor, k: int):

    assert bin_img.ndim == 4 and bin_img.shape[1] == 1, "انتظار (N,1,H,W)"
    dilations = [bin_img]
    for _ in range(k):
     
        d = F.max_pool2d(dilations[-1], kernel_size=3, stride=1, padding=1)
        d = (d > 0.5).float()
        dilations.append(d)
    rings = []
    for i in range(1, len(dilations)):
        ring = (dilations[i] - dilations[i-1]).clamp(min=0.0, max=1.0)
        rings.append(ring)
    return dilations, rings

@torch.no_grad()
def make_weight_matrix_torch(
    target_bin: torch.Tensor,
    k: int = 2,
    stroke_w: float = 3.5,       # 
    ring_weights = (3.0, 2.5),   
    background_w: float = 2.0    
):
  
    device = target_bin.device
    N, C, H, W = target_bin.shape
    weights = torch.full((N, 1, H, W), background_w, device=device, dtype=torch.float32)

    # خود stroke
    fg = (target_bin > 0.5).float()
    weights = torch.where(fg > 0.5, torch.tensor(stroke_w, device=device), weights)

    if k > 0:
        _, rings = dilate_binary_torch(fg, k)
      
        if len(ring_weights) < k:
         
            ring_weights = list(ring_weights) + [ring_weights[-1]] * (k - len(ring_weights))
        else:
            ring_weights = list(ring_weights[:k])

     
        not_fg = (fg < 0.5)
        for i, ring in enumerate(rings):
            rw = torch.tensor(ring_weights[i], device=device)
            weights = torch.where((ring > 0.5) & not_fg, rw, weights)

    return weights
'''
for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restored.train()
    for i, data in enumerate(tqdm(train_loader), 0):
        # Forward propagation
        for param in model_restored.parameters():
            param.grad = None
        target = data[0].cuda()

        input_ = data[1].cuda()

        if target.shape[1] == 3:
            target = 0.2989 * target[:, 0:1] + 0.5870 * \
                target[:, 1:2] + 0.1140 * target[:, 2:3]

        #restored = torch.sigmoid(model_restored(input_))
        restored = model_restored(input_)

            # Stack into (B,1,H,W) and move to GPU
        weights = make_weights_from_numpy(target, k=2, stroke_w=3.0, ring_w=(3.0,2.0,1.0))
        loss = charbonnier_loss(restored, target, weight=weights, eps=1e-3)
        
        # Back propagation
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Evaluation (Validation)
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model_restored.eval()
        psnr_val_rgb = []
        ssim_val_rgb = []
        val_epoch_loss = 0
        epoch_val_preds = []
        epoch_val_targets = []
        for ii, data_val in enumerate(val_loader, 0):
            target = data_val[0].cuda()
                                        # 🔍 بررسی محدوده تارگت برای نرمال‌سازی
            input_ = data_val[1].cuda()

            if target.shape[1] == 3:
                target = 0.2989 * target[:, 0:1] + 0.5870 * \
                    target[:, 1:2] + 0.1140 * target[:, 2:3]

            target_bin = (target > 0.5).float()
            with torch.no_grad():
                #restored = torch.sigmoid(model_restored(input_))
                restored = model_restored(input_)

            val_weights = make_weights_from_numpy(target, k=2, stroke_w=3.0, ring_w=(3.0, 2.0, 1.0))

            val_loss = charbonnier_loss(restored, target, weight=val_weights, eps=1e-3)

            val_epoch_loss += val_loss.item()
            '''
            for res, tar in zip(restored, target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))
                ssim_val_rgb.append(utils.torchSSIM(
                    res.unsqueeze(0), tar.unsqueeze(0)))
                # For confusion matrix: flatten and collect predictions and targets
                pred_bin = (res > 0.5).float().cpu().numpy().flatten()
                tar_bin = (tar > 0.5).float().cpu().numpy().flatten()
                epoch_val_preds.extend(pred_bin)
                epoch_val_targets.extend(tar_bin)
            '''
        # Log epoch loss
        loss_history.append(epoch_loss / len(train_loader))
        val_loss_history.append(val_epoch_loss / len(val_loader))
        '''
        # Plot and save confusion matrix for this epoch
        cm = confusion_matrix(epoch_val_targets, epoch_val_preds)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
            if cm.shape == (1, 1):
                if epoch_val_targets[0] == 1:
                    tp = cm[0, 0]
                else:
                    tn = cm[0, 0]
            elif cm.shape == (1, 2):
                tn, fp = cm[0]
            elif cm.shape == (2, 1):
                fn, tp = cm[:, 0]
        # Calculate True Positives (TP) and False Positives (FP)
        total = tn + fp + fn + tp
        # Normalize to [0, 1] if total > 0
        if total > 0:
            tp_normalized = tp / total
            fp_normalized = fp / total
        else:
            tp_normalized = 0.0
            fp_normalized = 0.0
        # ✅ محاسبه TPR و FPR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        tp_history.append(tpr)
        fp_history.append(fpr)

        #cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        #accuracy = accuracy_score(epoch_val_targets, epoch_val_preds)
        #precision = precision_score(
         #   epoch_val_targets, epoch_val_preds, zero_division=0)

        #accuracy_history.append(accuracy)
        #precision_history.append(precision)
        
        val_epoch_list.append(epoch)
        # Plot confusion matrix
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_normalized, display_labels=["Class 0", "Class 1"])
        disp.plot(cmap=plt.cm.Blues, values_format=".2%", ax=ax)
        plt.title(f'Normalized Confusion Matrix (Epoch {epoch})')
        plt.xlabel(
            f'Predicted Label\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}')
        plt.ylabel('True Label')
        plt.savefig(os.path.join(
            log_dir, f'val_confusion_matrix_epoch_{epoch}.png'))
        plt.close()

        # TensorBoard logs
       # writer.add_scalar('val/Accuracy', accuracy, epoch)
       # writer.add_scalar('val/Precision', precision, epoch)
'''
    scheduler.step()
    if epoch == 5:
        net = model_restored.module if hasattr(model_restored, "module") else model_restored
        lname, last = get_last_trainable_leaf(net)
        torch.save(last.state_dict(), os.path.join(model_dir, f"last_layer_{lname}_e{epoch:03d}.pth"))

    '''
    # Save "latest" every epoch (you already have this)
    torch.save({
    'epoch': epoch,
    'state_dict': (model_restored.module if hasattr(model_restored, "module") else model_restored).state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    }, os.path.join(model_dir, "model_latest.pth"))

# Save a copy at epoch 5
    if epoch == 5:
        torch.save({
        'epoch': epoch,
        'state_dict': (model_restored.module if hasattr(model_restored, "module") else model_restored).state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        }, os.path.join(model_dir, f"model_epoch_{epoch:03d}.pth"))
    '''
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_last_lr()[0]))
    print("------------------------------------------------------------------")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_restored.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
total_finish_time = (time.time() - total_start_time)  # seconds

writer.close()

# Plot training and validation loss per epoch with value labels
epochs_train = list(range(1, start_epoch + len(loss_history)))  # From 1 to 2
plt.figure(figsize=(10, 6))  # بزرگ‌تر برای وضوح بیشتر

# Offset برای لیبل‌ها
train_offset = 0.01
val_offset = 0.01

# رسم Training Loss
plt.plot(epochs_train, loss_history, marker='o',
         label='Training Loss', color='blue')
for x, y in zip(epochs_train, loss_history):
    plt.text(x, y + train_offset,
             f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='blue')

# رسم Validation Loss (در صورت وجود)
if val_loss_history:
    val_epochs = [start_epoch + Train['VAL_AFTER_EVERY'] - 1 + i * Train['VAL_AFTER_EVERY']
                  for i in range(len(val_loss_history))]
    plt.plot(val_epochs, val_loss_history, marker='o',
             color='red', label='Validation Loss')
    for x, y in zip(val_epochs, val_loss_history):
        plt.text(x, y + val_offset,
                 f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='red')

# تنظیمات نمودار
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(log_dir, 'train_val_loss_with_values.png'))
plt.show()


plt.figure(figsize=(10, 6))  # بزرگ‌تر برای وضوح
'''
# رسم خطوط
plt.plot(val_epoch_list, tp_history, marker='o',
         label='True Positive Rate (TPR) Percent', color='green')
plt.plot(val_epoch_list, fp_history, marker='x',
         label='False Positive Rate (FPR) Percent', color='orange')

# فاصله متن‌ها
tpr_offset = 0.03
fpr_offset = 0.07  # کمی بیشتر از TPR برای جلوگیری از هم‌پوشانی

# لیبل‌های TPR (بالای نقطه)
for x, y in zip(val_epoch_list, tp_history):
    plt.text(x, y + tpr_offset, f'{y*100:.1f}',
             ha='center', va='bottom', fontsize=9, color='green')

# لیبل‌های FPR (بالای نقطه ولی با فاصله بیشتر)
for x, y in zip(val_epoch_list, fp_history):
    plt.text(x, y + fpr_offset, f'{y*100:.1f}',
             ha='center', va='bottom', fontsize=9, color='orange')

# تنظیمات کلی نمودار
plt.xlabel('Epoch')
plt.ylabel('Rate')
plt.title('True Positive Rate (TPR) and False Positive Rate (FPR) per Epoch')
plt.ylim(0, 1.15)  # فضای عمودی بیشتر برای نمایش متن‌های بالا
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(log_dir, 'tpr_fpr_per_epoch_readable.png'))
plt.show()
'''

# Save loss values to a text file
loss_txt_path = os.path.join(log_dir, 'train_val_loss_values.txt')
with open(loss_txt_path, 'w') as f:
    f.write('Epoch\tTrain_Loss\tVal_Loss\n')
    max_epochs = max(len(epochs_train), len(
        val_loss_history) if val_loss_history else 0)
    for i in range(max_epochs):
        epoch_num = epochs_train[i] if i < len(epochs_train) else ''
        train_loss = f'{loss_history[i]:.6f}' if i < len(loss_history) else ''
        # Find val loss for this epoch if it exists
        val_loss = ''
        if val_loss_history and i < len(val_loss_history):
            # Only write val_loss at the correct epoch
            val_epoch = start_epoch + \
                Train['VAL_AFTER_EVERY'] - 1 + i * Train['VAL_AFTER_EVERY']
            if epoch_num == val_epoch:
                val_loss = f'{val_loss_history[i]:.6f}'
        f.write(f'{epoch_num}\t{train_loss}\t{val_loss}\n')
# Save accuracy and precision values to a text file
'''
acc_prec_txt_path = os.path.join(log_dir, 'val_accuracy_precision.txt')
with open(acc_prec_txt_path, 'w') as f:
    f.write('Epoch\tAccuracy\tPrecision\n')
    for i in range(len(val_epoch_list)):
        f.write(
            f'{val_epoch_list[i]}\t{accuracy_history[i]:.6f}\t{precision_history[i]:.6f}\n')
# Save TPR and FPR values to a text file
tp_fp_txt_path = os.path.join(log_dir, 'val_tpr_fpr.txt')
with open(tp_fp_txt_path, 'w') as f:
    f.write('Epoch\tTPR\tFPR\n')
    for i in range(len(val_epoch_list)):
        f.write(
            f'{val_epoch_list[i]}\t{tp_history[i]:.6f}\t{fp_history[i]:.6f}\n')
'''

total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} hours'.format(
    (total_finish_time / 60 / 60)))
