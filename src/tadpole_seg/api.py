from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import torch

from .model import load_model, IMG_SIZE
from .visuals import overlay as _overlay

def _fit_to_square_meta(h: int, w: int, size: int):
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    top  = (size - nh) // 2
    left = (size - nw) // 2
    return s, nh, nw, top, left

def _predict_mask_square(model, device, img_rgb: np.ndarray) -> np.ndarray:
    H, W = img_rgb.shape[:2]
    _, nh, nw, top, left = _fit_to_square_meta(H, W, IMG_SIZE)

    img_res = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    canvas[top:top+nh, left:left+nw] = img_res

    tfm = A.Compose([A.Normalize(), ToTensorV2()])
    tens = tfm(image=canvas)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tens)
        probs = torch.sigmoid(logits)
        pred_sq = (probs > 0.5).float().cpu().numpy()[0, 0].astype(np.uint8)

    pred_cropped = pred_sq[top:top+nh, left:left+nw]
    pred_orig = cv2.resize(pred_cropped, (W, H), interpolation=cv2.INTER_NEAREST)
    return (pred_orig * 255).astype(np.uint8)

def _list_images_in_folder(folder: Path, patterns: Tuple[str, ...]):
    pat_lower = tuple(p.lower() for p in patterns)
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in pat_lower])

def segment_tadpole(
    scale_px_per_cm: float,
    input_path: str | Path,
    output_dir: str | Path | None = None,
    return_visuals: bool = True,
    save_outputs: bool = False,
    alpha: float = 0.4,
    patterns: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".PNG"),
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> List[Dict[str, Any]]:
    assert scale_px_per_cm > 0, "scale_px_per_cm must be > 0 (px per cm)"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(checkpoint_path=checkpoint_path, device=device)

    input_path = Path(input_path)
    if input_path.is_dir():
        imgs = _list_images_in_folder(input_path, patterns)
    else:
        imgs = [input_path]
        assert input_path.exists(), f"Input not found: {input_path}"

    if output_dir is None:
        base = input_path if input_path.is_dir() else input_path.parent
        output_dir = base / "tadpole_preds"
    output_dir = Path(output_dir)
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for p in imgs:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Skipped unreadable file: {p}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        mask = _predict_mask_square(model, device, rgb)
        area_px = int(cv2.countNonZero(mask))
        area_cm2 = float(area_px / (scale_px_per_cm ** 2))

        ov = _overlay(rgb, mask, alpha=alpha)

        if save_outputs:
            stem = p.stem
            cv2.imwrite(str(output_dir / f"{stem}_pred.png"), mask)
            cv2.imwrite(str(output_dir / f"{stem}_overlay.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

        if return_visuals:
            fig = plt.figure(figsize=(12, 5))
            plt.suptitle(f"{p.name} — area: {area_px} px^2 ({area_cm2:.4f} cm^2)", fontsize=14)
            plt.subplot(1, 3, 1); plt.imshow(rgb);  plt.axis("off"); plt.title("Original")
            plt.subplot(1, 3, 2); plt.imshow(mask, cmap="gray", vmin=0, vmax=255); plt.axis("off"); plt.title("Mask")
            plt.subplot(1, 3, 3); plt.imshow(ov);   plt.axis("off"); plt.title("Overlay")
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            plt.show()

        item = {"path": p, "area_px": area_px, "area_cm2": area_cm2}
        if return_visuals:
            item["mask"] = mask
            item["overlay"] = ov
        results.append(item)

    print(f"Done. Processed {len(results)} image(s). Output dir: {output_dir if save_outputs else '(not saved)'}")
    return results
