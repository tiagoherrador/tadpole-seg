from pathlib import Path
import os
import torch
import segmentation_models_pytorch as smp
from importlib.resources import files

IMG_SIZE = 1024  # enforced

def _default_ckpt_path() -> Path:
    try:
        p = files("tadpole_seg") / "models" / "best_unet.pth"
        return Path(p)
    except Exception:
        return Path("/content/drive/MyDrive/Embrio_RRNN_OK/FineTuning_Outputs/checkpoints/best_unet.pth")

_model_cache = {"model": None, "path": None, "encoder": None}

def load_model(checkpoint_path=None, device="cpu"):
    if checkpoint_path is None:
        checkpoint_path = os.getenv("TADPOLE_CKPT", None)
    if checkpoint_path is None:
        checkpoint_path = _default_ckpt_path()
    else:
        checkpoint_path = Path(checkpoint_path)

    if _model_cache["model"] is not None and _model_cache["path"] == checkpoint_path:
        return _model_cache["model"]

    assert Path(checkpoint_path).exists(), f"Checkpoint not found: {checkpoint_path}"
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = smp.Unet(
        encoder_name=ckpt["encoder"],
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    _model_cache["model"] = model
    _model_cache["path"] = checkpoint_path
    _model_cache["encoder"] = ckpt["encoder"]
    return model
