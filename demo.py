import torch
import torchvision.transforms.functional as TF
from PIL import Image
import os
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import cv2
import argparse
from model.SUNet import SUNet_model
import yaml

with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)


parser = argparse.ArgumentParser(description='Demo Image Restoration')
parser.add_argument('--input_dir', default='C:/Users/Lab722 BX/Desktop/CBSD68_test/CBSD68_50_crop/', type=str, help='Input images')
parser.add_argument('--window_size', default=8, type=int, help='window size')
parser.add_argument('--result_dir', default='C:/Users/Lab722 BX/Desktop/CBSD68_test/SUNet_50_crop/', type=str, help='Directory for results')
parser.add_argument('--weights',
                    default='./pretrain-model/model_bestPSNR.pth', type=str,
                    help='Path to weights')
parser.add_argument('--weights', type=str, required=True,
                    help='Full model checkpoint or pure state_dict')
parser.add_argument('--last_layer_pth', type=str, default='',
                    help='Path to last-layer-only state_dict (optional)')
parser.add_argument('--last_layer_path', type=str, default='',
                    help='Dotted path to the submodule, e.g. "swin_unet.output"')


args = parser.parse_args()


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

import os, torch

def get_submodule_by_path(model, path_str: str):
    # works with DataParallel/DistributedDataParallel too
    net = model.module if hasattr(model, "module") else model
    # torch>=1.9 has get_submodule
    if hasattr(net, "get_submodule"):
        return net.get_submodule(path_str)
    # fallback: manual traversal
    sub = net
    for p in path_str.split('.'):
        sub = getattr(sub, p)
    return sub

def load_full_or_state(model, ckpt_path, strict=False):
    device = next(model.parameters()).device
    ckpt = torch.load(ckpt_path, map_location=device)
    # accept pure state_dict or dict with 'state_dict'
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    # strip DataParallel prefix if present
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    target = model.module if hasattr(model, "module") else model
    missing, unexpected = target.load_state_dict(state, strict=strict)
    print("[full load] missing:", missing)
    print("[full load] unexpected:", unexpected)

def load_last_layer(model, last_layer_path, layer_path=None, strict=False):
    """
    last_layer_path: file saved via last.state_dict() => keys like 'weight', 'bias'
    layer_path: dotted path to the submodule, e.g. 'swin_unet.output'
                If None, tries to parse from filename 'last_layer_<layer_path>_e*.pth'
    """
    device = next(model.parameters()).device
    small = torch.load(last_layer_path, map_location=device)

    # infer layer path from filename if not given
    if layer_path is None:
        base = os.path.basename(last_layer_path)
        # expects pattern: last_layer_<layer_path>_e*.pth
        if "last_layer_" in base:
            layer_path = base.split("last_layer_")[1].rsplit("_e", 1)[0]
        else:
            raise ValueError("Please provide layer_path, couldn't infer from filename.")

    sub = get_submodule_by_path(model, layer_path)

    # Optional: sanity check shapes
    for k, v in small.items():
        if not hasattr(sub, k):
            print(f"[warn] submodule has no attr '{k}'")
        else:
            mparam = getattr(sub, k)
            if hasattr(mparam, "shape") and hasattr(v, "shape") and tuple(mparam.shape) != tuple(v.shape):
                print(f"[warn] shape mismatch for {layer_path}.{k}: "
                      f"ckpt {tuple(v.shape)} vs model {tuple(mparam.shape)}")

    missing, unexpected = sub.load_state_dict(small, strict=strict)
    print(f"[last-layer load -> {layer_path}] missing:", missing, "| unexpected:", unexpected)



def _strip_module(sd: dict) -> dict:
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def _extract_state_dict(ckpt) -> dict:
    # Accept: pure state_dict OR {"state_dict": ...} OR {"model": ...}
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        # If dict of tensors, assume it's already a state_dict
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    # Fallback: assume ckpt itself is a state_dict-like object
    return ckpt

def load_checkpoint(model, weights_path, map_location=None, strict=False):
    device = map_location or (next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else "cpu")
    # PyTorch 2.6 default is weights_only=True; that’s fine for tensors.
    ckpt = torch.load(weights_path, map_location=device)
    state = _strip_module(_extract_state_dict(ckpt))

    target = model.module if hasattr(model, "module") else model
    missing, unexpected = target.load_state_dict(state, strict=strict)
    print("[load] missing:", missing)
    print("[load] unexpected:", unexpected)
'''    
def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
'''

inp_dir = args.input_dir
out_dir = args.result_dir

os.makedirs(out_dir, exist_ok=True)

files = natsorted(glob(os.path.join(inp_dir, '*.jpg'))
                  + glob(os.path.join(inp_dir, '*.JPG'))
                  + glob(os.path.join(inp_dir, '*.bmp'))
                 + glob(os.path.join(inp_dir, '*.BMP'))
                  + glob(os.path.join(inp_dir, '*.png'))
                  + glob(os.path.join(inp_dir, '*.PNG')))

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

# Load corresponding model architecture and weights
model = SUNet_model(opt).cuda()
load_full_or_state(model, args.weights, strict=False)
if args.last_layer_pth and args.last_layer_path:
    load_last_layer(model, args.last_layer_pth, args.last_layer_path, strict=False)
model.eval()

print('restoring images......')


for file_ in files:
    img = Image.open(file_).convert('RGB')
    input_ = TF.to_tensor(img).unsqueeze(0).cuda()

    with torch.no_grad():
        restored = model(input_)
        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

    restored = img_as_ubyte(restored[0])

    f = os.path.splitext(os.path.split(file_)[-1])[0]
    save_img((os.path.join(out_dir, f + '.bmp')), restored)

print(f"Files saved at {out_dir}")
print('finish !')