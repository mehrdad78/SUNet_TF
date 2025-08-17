import torch
from collections import OrderedDict

def load_checkpoint(model, weight_path, device=None):
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = 'cpu'

    # 1) Explicitly allow full pickle load (you created the file, so it’s trusted)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)

    # 2) Determine the actual state_dict inside the checkpoint
    sd = None
    if isinstance(ckpt, (dict, OrderedDict)) and all(isinstance(k, str) for k in ckpt.keys()) and any(k.endswith(".weight") or k.endswith(".bias") for k in ckpt.keys()):
        sd = ckpt
    elif isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights", "params", "net"):
            if key in ckpt and isinstance(ckpt[key], (dict, OrderedDict)):
                sd = ckpt[key]
                break
    if sd is None:
        raise RuntimeError(f"Could not find a state_dict in checkpoint. Top-level keys: {list(ckpt.keys())[:20]}")

    # 3) Strip 'module.' if saved with DataParallel/DistributedDataParallel
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    # 4) Load (non-strict to tolerate harmless extras)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load_checkpoint] Missing keys: {missing}")
    if unexpected:
        print(f"[load_checkpoint] Unexpected keys: {unexpected}")

    model.eval()
