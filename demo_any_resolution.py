import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import os
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import cv2
import argparse
from model.SUNet import SUNet_model
import math
from tqdm import tqdm
import yaml
from sklearn.metrics import confusion_matrix

# Load YAML config
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)

# CLI args
parser = argparse.ArgumentParser(description='Demo Image Restoration')
parser.add_argument('--input_dir', default='C:/Users/Lab722 BX/Desktop/Kodak24_test/Kodak24_10/', type=str)
parser.add_argument('--mask_dir', default='C:/path/to/masks/', type=str)
parser.add_argument('--window_size', default=8, type=int)
parser.add_argument('--size', default=256, type=int)
parser.add_argument('--stride', default=128, type=int)
parser.add_argument('--result_dir', default='./demo_results/', type=str)
parser.add_argument('--weights', default='./pretrain-model/model_bestPSNR.pth', type=str)
args = parser.parse_args()

# Utility: crop to patches
def overlapped_square(timg, kernel=256, stride=128):
    patch_images = []
    b, c, h, w = timg.size()
    X = int(math.ceil(max(h, w) / float(kernel)) * kernel)
    img = torch.zeros(1, 3, X, X).type_as(timg)
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1.0)

    patch = img.unfold(3, kernel, stride).unfold(2, kernel, stride)
    patch = patch.contiguous().view(b, c, -1, kernel, kernel)
    patch = patch.permute(2, 0, 1, 4, 3)

    for each in range(len(patch)):
        patch_images.append(patch[each])

    return patch_images, mask, X

# Utility: save RGB image
def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# Utility: compute TPR & FPR
def calculate_tpr_fpr(pred, target):
    pred_bin = (pred > 127).astype(np.uint8).flatten()
    target_bin = (target > 200).astype(np.uint8).flatten()
    cm = confusion_matrix(target_bin, pred_bin, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr, fpr

# Load model
def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

# Directories
inp_dir = args.input_dir
mask_dir = args.mask_dir
out_dir = args.result_dir
os.makedirs(out_dir, exist_ok=True)

# Collect image files
files = natsorted(glob(os.path.join(inp_dir, '*.*')))
files = [f for f in files if f.lower().endswith(('.jpg', '.bmp', '.png'))]

if len(files) == 0:
    raise Exception(f"No image files found in {inp_dir}")

# Init model
model = SUNet_model(opt)
model.cuda()
load_checkpoint(model, args.weights)
model.eval()

print('Restoring images...')
stride = args.stride
model_img = args.size

# Prepare result log file
results_txt_path = os.path.join(out_dir, "tpr_fpr_results.txt")
with open(results_txt_path, 'w') as result_file:
    result_file.write("Filename\tTPR\tFPR\n")

    for file_ in tqdm(files):
        img = Image.open(file_).convert('RGB')
        input_ = TF.to_tensor(img).unsqueeze(0).cuda()

        with torch.no_grad():
            patches, mask, max_wh = overlapped_square(input_, kernel=model_img, stride=stride)
            output_patch = torch.zeros_like(patches[0])
            for i, patch in enumerate(patches):
                restored = model(patch)
                if i == 0:
                    output_patch += restored
                else:
                    output_patch = torch.cat([output_patch, restored], dim=0)

            B, C, H, W = output_patch.shape
            weight = torch.ones_like(output_patch)

            patch = output_patch.contiguous().view(B, C, -1, model_img*model_img)
            patch = patch.permute(2, 1, 3, 0).contiguous().view(1, C*model_img*model_img, -1)

            weight_mask = weight.contiguous().view(B, C, -1, model_img*model_img)
            weight_mask = weight_mask.permute(2, 1, 3, 0).contiguous().view(1, C*model_img*model_img, -1)

            restored = F.fold(patch, output_size=(max_wh, max_wh), kernel_size=model_img, stride=stride)
            we_mk = F.fold(weight_mask, output_size=(max_wh, max_wh), kernel_size=model_img, stride=stride)
            restored /= we_mk

            restored = torch.masked_select(restored, mask.bool()).reshape(input_.shape)
            restored = torch.clamp(restored, 0, 1)

        restored = restored.permute(0, 2, 3, 1).cpu().numpy()
        restored = img_as_ubyte(restored[0])

        # Save output image
        f = os.path.splitext(os.path.basename(file_))[0]
        save_img(os.path.join(out_dir, f + '.bmp'), restored)

        # Evaluate TPR/FPR
        mask_path = os.path.join(mask_dir, os.path.basename(file_))
        if os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert("L")
            mask_np = np.array(mask_img)
            pred_np = cv2.cvtColor(restored, cv2.COLOR_RGB2GRAY)

            tpr, fpr = calculate_tpr_fpr(pred_np, mask_np)
            print(f"{os.path.basename(file_)} — TPR: {tpr:.4f}, FPR: {fpr:.4f}")
            result_file.write(f"{os.path.basename(file_)}\t{tpr:.4f}\t{fpr:.4f}\n")
        else:
            print(f"Mask not found for {file_}, skipping TPR/FPR.")

print(f"\nAll results saved in: {out_dir}")
