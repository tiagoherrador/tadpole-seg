# tadpole-seg – U-Net segmentation and size measurement of tadpoles

Fine-tuned **U-Net (EfficientNet-B0 encoder, 1024 px)** that segments Xenopus tadpoles/embryos photographed next to a ruler,
and converts the mask to **physical area (cm²)** from a px/cm calibration. Ships as an installable package with a CLI and the trained checkpoint (~25 MB).

![Prediction](assets/example_prediction.png)

## Training (see `notebooks/train_unet_finetuning.ipynb`)
- `segmentation_models_pytorch` U-Net, `timm-efficientnet-b0` encoder, Albumentations augmentation, AMP, 50 epochs.
- **Tiny dataset: 30 manually annotated photographs.** Best validation **Dice ≈ 0.725, IoU ≈ 0.70**.
- Model then applied to 48 unseen photos (`preds_nuevas`) for qualitative inspection.

![Curves](assets/validation_metrics.png)

**Caveats:** with 30 images and a single validation split the metrics are noisy; the model is specific to this photographic setup
(uniform background, same camera/ruler). Area values depend on the calibration you provide.

## Install & use
```bash
pip install .
tadpole-seg --scale-px-per-cm 317.43 --input path/to/image_or_folder --output out/ --save --show
```
```python
import tadpole_seg as tps
res = tps.segment_tadpole(scale_px_per_cm=317.43, input_path="examples/tadpole.JPG",
                          return_visuals=True, save_outputs=False)
```
The checkpoint can be overridden with `checkpoint_path=` or the `TADPOLE_CKPT` environment variable.

> **Note:** originally developed and run in Google Colab. Google-Drive mounting cells were removed, Drive paths were rewritten as relative paths and outputs were stripped (figures of the original run are in `assets/`). The paths/cleanup were **not re-executed** after this change, so expect to adjust paths.

## Author
Tiago Alejo Herrador
