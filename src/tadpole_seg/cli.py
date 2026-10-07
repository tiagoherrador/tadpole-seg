import argparse
from pathlib import Path
from .api import segment_tadpole

def main():
    p = argparse.ArgumentParser(description="Xenopus tadpole segmentation (IMG_SIZE=1024).")
    p.add_argument("--scale-px-per-cm", type=float, required=True, help="Linear pixels-per-centimeter calibration.")
    p.add_argument("--input", type=str, required=True, help="Image path or folder path.")
    p.add_argument("--output", type=str, default=None, help="Output dir for *_pred.png and *_overlay.jpg when --save is set.")
    p.add_argument("--alpha", type=float, default=0.4, help="Overlay transparency.")
    p.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint .pth path (overrides bundled/env).")
    p.add_argument("--show", action="store_true", help="Show matplotlib figures.")
    p.add_argument("--save", action="store_true", help="Save outputs to --output directory.")
    args = p.parse_args()

    segment_tadpole(
        scale_px_per_cm=args.scale_px_per_cm,
        input_path=Path(args.input),
        output_dir=Path(args.output) if args.output else None,
        return_visuals=bool(args.show),
        save_outputs=bool(args.save),
        alpha=args.alpha,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
    )

if __name__ == "__main__":
    main()
