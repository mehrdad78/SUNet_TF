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

args = parser.parse_args()


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

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
model = SUNet_model(opt)
model.cuda()

load_checkpoint(model, args.weights)
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