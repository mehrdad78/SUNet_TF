import os

import torch
import yaml

from utils import network_parameters
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, accuracy_score

import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import time
import utils
import numpy as np
import random
from data_RGB import get_training_data, get_validation_data

from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model.SUNet import SUNet_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

## Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

## Load yaml configuration file
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']

## Build Model
print('==> Build the model')
model_restored = SUNet_model(opt)
p_number = network_parameters(model_restored)
model_restored.cuda()

## Training model path direction
mode = opt['MODEL']['MODE']

model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
utils.mkdir(model_dir)
train_dir = Train['TRAIN_DIR']
val_dir = Train['VAL_DIR']

## GPU
gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
if len(device_ids) > 1:
    model_restored = nn.DataParallel(model_restored, device_ids=device_ids)

## Log
log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

## Optimizer
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

## Scheduler (Strategy)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

## Resume (Continue training by a pretrained model)
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

## Loss
#L1_loss = nn.L1Loss()
criterion = nn.BCELoss() 
loss_history = [] 

## DataLoaders
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
        # Convert target to grayscale if it is RGB
        if target.shape[1] == 3:
            target = 0.2989 * target[:,0:1] + 0.5870 * target[:,1:2] + 0.1140 * target[:,2:3]
        restored = torch.sigmoid(model_restored(input_))  # Add sigmoid activation
        #loss = criterion(restored, target)
        loss = 0
        for j in range(input_.size(0)):  # loop over batch
            target_j = target[j:j+1]         # shape: [1, 1, H, W]
            restored_j = restored[j:j+1]     # shape: [1, 1, H, W]

            mask_np = target_j.squeeze().cpu().numpy()
            pos_dist = distance_transform_edt(mask_np == 0)
            neg_dist = distance_transform_edt(mask_np == 1)
            weights = pos_dist + neg_dist
            weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)


            weight_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

            loss += F.binary_cross_entropy(restored_j, target_j, weight=weight_tensor)

        loss /= input_.size(0)  # Average loss over batch
        

        # Back propagation
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    ## Evaluation (Validation)
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model_restored.eval()
        psnr_val_rgb = []
        ssim_val_rgb = []
        val_epoch_loss = 0
        epoch_val_preds = []
        epoch_val_targets = []
        for ii, data_val in enumerate(val_loader, 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()
            # Convert target to grayscale if it is RGB
            if target.shape[1] == 3:
                target = 0.2989 * target[:,0:1] + 0.5870 * target[:,1:2] + 0.1140 * target[:,2:3]
            with torch.no_grad():
                restored = torch.sigmoid(model_restored(input_))
                val_loss = criterion(restored, target)
            val_epoch_loss += val_loss.item()
            for res, tar in zip(restored, target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))
                ssim_val_rgb.append(utils.torchSSIM(res.unsqueeze(0), tar.unsqueeze(0)))
                # For confusion matrix: flatten and collect predictions and targets
                pred_bin = (res > 0.5).float().cpu().numpy().flatten()
                tar_bin = (tar > 0.5).float().cpu().numpy().flatten()
                epoch_val_preds.extend(pred_bin)
                epoch_val_targets.extend(tar_bin)
        # Log epoch loss for plotting (after batch loop)
        #loss_history.append(epoch_loss / len(train_loader))
        #val_loss_history.append(val_epoch_loss / len(val_loader))
        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()
          # Log epoch loss
        loss_history.append(epoch_loss / len(train_loader))
        val_loss_history.append(val_epoch_loss / len(val_loader))
        #psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        #ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()
        #psnr_history.append(psnr_val_rgb)
        #ssim_history.append(ssim_val_rgb)

        # Save the best PSNR model of validation
        '''
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch_psnr = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestPSNR.pth"))
        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
            epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))
        '''
        # Save the best SSIM model of validation
        '''if ssim_val_rgb > best_ssim:
            best_ssim = ssim_val_rgb
            best_epoch_ssim = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestSSIM.pth"))
        print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
            epoch, ssim_val_rgb, best_epoch_ssim, best_ssim))
            '''
        # Plot and save confusion matrix for this epoch
        cm = confusion_matrix(epoch_val_targets, epoch_val_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        accuracy = accuracy_score(epoch_val_targets, epoch_val_preds)
        precision = precision_score(epoch_val_targets, epoch_val_preds, zero_division=0)

        accuracy_history.append(accuracy)
        precision_history.append(precision)
        val_epoch_list.append(epoch)

        # Plot confusion matrix
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=["Class 0", "Class 1"])
        disp.plot(cmap=plt.cm.Blues, values_format=".2%", ax=ax)
        plt.title(f'Normalized Confusion Matrix (Epoch {epoch})')
        plt.xlabel(f'Predicted Label\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}')
        plt.ylabel('True Label')
        plt.savefig(os.path.join(log_dir, f'val_confusion_matrix_epoch_{epoch}.png'))
        plt.close()

        # TensorBoard logs
        #writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)
        #writer.add_scalar('val/SSIM', ssim_val_rgb, epoch)
        writer.add_scalar('val/Accuracy', accuracy, epoch)
        writer.add_scalar('val/Precision', precision, epoch)

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_restored.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
total_finish_time = (time.time() - total_start_time)  # seconds

writer.close()

# Plot training and validation loss per epoch with value labels
import matplotlib.pyplot as plt
#epochs_train = list(range(start_epoch, start_epoch + len(loss_history)))
epochs_train = list(range(1, start_epoch + len(loss_history)))  # From 1 to 2

plt.figure()
plt.plot(epochs_train, loss_history, marker='o', label='Training Loss')
for x, y in zip(epochs_train, loss_history):
    plt.text(x, y, f'{y:.4f}', ha='center', va='bottom', fontsize=8, color='blue')

if val_loss_history:
    val_epochs = [start_epoch + Train['VAL_AFTER_EVERY'] - 1 + i * Train['VAL_AFTER_EVERY'] for i in range(len(val_loss_history))]
    plt.plot(val_epochs, val_loss_history, marker='o', color='red', label='Validation Loss')
    for x, y in zip(val_epochs, val_loss_history):
        plt.text(x, y, f'{y:.4f}', ha='center', va='bottom', fontsize=8, color='red')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(log_dir, 'train_val_loss_with_values.png'))
plt.show()

# Save loss values to a text file
loss_txt_path = os.path.join(log_dir, 'train_val_loss_values.txt')
with open(loss_txt_path, 'w') as f:
    f.write('Epoch\tTrain_Loss\tVal_Loss\n')
    max_epochs = max(len(epochs_train), len(val_loss_history) if val_loss_history else 0)
    for i in range(max_epochs):
        epoch_num = epochs_train[i] if i < len(epochs_train) else ''
        train_loss = f'{loss_history[i]:.6f}' if i < len(loss_history) else ''
        # Find val loss for this epoch if it exists
        val_loss = ''
        if val_loss_history and i < len(val_loss_history):
            # Only write val_loss at the correct epoch
            val_epoch = start_epoch + Train['VAL_AFTER_EVERY'] - 1 + i * Train['VAL_AFTER_EVERY']
            if epoch_num == val_epoch:
                val_loss = f'{val_loss_history[i]:.6f}'
        f.write(f'{epoch_num}\t{train_loss}\t{val_loss}\n')
# Save accuracy and precision values to a text file
acc_prec_txt_path = os.path.join(log_dir, 'val_accuracy_precision.txt')
with open(acc_prec_txt_path, 'w') as f:
    f.write('Epoch\tAccuracy\tPrecision\n')
    for i in range(len(val_epoch_list)):
        f.write(f'{val_epoch_list[i]}\t{accuracy_history[i]:.6f}\t{precision_history[i]:.6f}\n')



total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
