import os
import cv2
import yaml
import torch
import argparse
import numpy as np
from PIL import Image
from glob import glob
from model.SUNet import SUNet_model
import torchvision.transforms.functional as TF
from skimage import img_as_ubyte
from natsort import natsorted

# ====================
# Argument parsing
# ====================
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)

parser = argparse.ArgumentParser(description='Demo Image Restoration')
parser.add_argument('--input_dir', type=str, required=True, help='Input images')
parser.add_argument('--result_dir', type=str, required=True, help='Directory for results')
parser.add_argument('--weights', type=str, required=True, help='Full model checkpoint or state_dict')
parser.add_argument('--last_layer_pth', type=str, default='', help='Path to last-layer-only state_dict (optional)')
parser.add_argument('--last_layer_path', type=str, default='', help='Submodule path (e.g., "swin_unet.output")')
args = parser.parse_args()

# ====================
# Helper functions
# ====================
def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def get_submodule_by_path(model, path_str: str):
    net = model.module if hasattr(model, "module") else model
    for p in path_str.split('.'):
        net = getattr(net, p)
    return net

def load_full_or_state(model, ckpt_path, strict=False):
    device = next(model.parameters()).device
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    target = model.module if hasattr(model, "module") else model
    missing, unexpected = target.load_state_dict(state, strict=strict)
    print("[full load] missing:", missing)
    print("[full load] unexpected:", unexpected)

def load_last_layer(model, last_layer_path, layer_path=None, strict=False):
    device = next(model.parameters()).device
    small = torch.load(last_layer_path, map_location=device)

    if layer_path is None:
        base = os.path.basename(last_layer_path)
        if "last_layer_" in base:
            layer_path = base.split("last_layer_")[1].rsplit("_e", 1)[0]
        else:
            raise ValueError("Please provide layer_path")

    sub = get_submodule_by_path(model, layer_path)
    for k, v in small.items():
        if hasattr(sub, k):
            mparam = getattr(sub, k)
            if hasattr(mparam, "shape") and hasattr(v, "shape") and tuple(mparam.shape) != tuple(v.shape):
                print(f"[warn] shape mismatch: {k}: ckpt {v.shape} vs model {mparam.shape}")
    missing, unexpected = sub.load_state_dict(small, strict=strict)
    print(f"[last-layer load -> {layer_path}] missing:", missing, "| unexpected:", unexpected)

# ====================
# Main
# ====================
inp_dir = args.input_dir
out_dir = args.result_dir
os.makedirs(out_dir, exist_ok=True)

files = natsorted(glob(os.path.join(inp_dir, '*.*')))
valid_exts = ('.jpg', '.JPG', '.bmp', '.BMP', '.png', '.PNG')
files = [f for f in files if f.endswith(valid_exts)]

if not files:
    raise Exception(f"No images found in {inp_dir}")

# Load model and weights
model = SUNet_model(opt).cuda()
load_full_or_state(model, args.weights, strict=False)
if args.last_layer_pth and args.last_layer_path:
    load_last_layer(model, args.last_layer_pth, args.last_layer_path, strict=False)
model.eval()

print("Restoring images...")

for file_ in files:
    img = Image.open(file_).convert('RGB')
    input_tensor = TF.to_tensor(img).unsqueeze(0).cuda()

    with torch.no_grad():
        restored = model(input_tensor)
        restored = torch.clamp(restored, 0, 1)
        restored_img = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
        restored_img = img_as_ubyte(restored_img)

    base = os.path.splitext(os.path.basename(file_))[0]
    out_path = os.path.join(out_dir, f"{base}.bmp")
    save_img(out_path, restored_img)

print(f"✅ Done! Results saved to: {out_dir}")
