# CarDD — instance segmentation with ConvNeXt-V2

Mask R-CNN / Cascade Mask R-CNN over ConvNeXt-V2 backbones for car damage
instance segmentation on CarDD (6 classes: dent, scratch, crack,
glass_shatter, lamp_broken, tire_flat).

## Setup

Download the CarDD dataset from the authors' page and unpack it into `data/`
at the repository root, so the COCO split lands at `data/CarDD_COCO`:

- https://cardd-ustc.github.io

```
data/
└── CarDD_COCO/
    ├── annotations/
    ├── train2017/
    ├── val2017/
    └── test2017/
```

That path is `data.root` in `configs/base.yaml`. Nothing else needs editing to
train. Requires PyTorch, torchvision, timm and pycocotools.

## Running an experiment

Everything is driven from `configs/ablation/`. Three rows, one file each:

| config | what it runs |
|---|---|
| `a0_baseline.yaml` | baseline — horizontal flip and multi-scale only |
| `a1_nsa.yaml` | + NSA (composition augmentation), uniform class weights |
| `a2_nsacd.yaml` | + NSA with class-decayed weighting |

```bash
export PYTHONPATH=src
python scripts/train.py    --config configs/ablation/a1_nsa.yaml
python scripts/evaluate.py --config configs/ablation/a1_nsa.yaml \
                           --weights outputs/nsa/best.pth --split test
```

These three files are meant to be run as they are. They carry only the delta
that defines their row; the schedule, optimizer, image sizes and the
augmentation parameters themselves are constants shared by every row and live
in `configs/base.yaml`. **Change a row's config to pick an experiment; change
`base.yaml` to change the experiment itself.**

## Changing the backbone

Every row inherits the backbone from `a0_baseline.yaml`, which defaults to
tiny. Pico and base are already wired, with pretrained weights — switch by
pointing `_base_` at another file in `configs/backbones/`:

```yaml
# configs/ablation/a0_baseline.yaml
_base_: ../backbones/convnextv2_tiny.yaml    # or convnextv2_pico / convnextv2_base
```

One edit there moves all three rows, which is what keeps them comparable.

## Layout

```
configs/    base.yaml (project constants) · backbones/ · ablation/ (the rows)
src/cardd/  the package
scripts/    train.py · evaluate.py · compare_grid.py
tools/      count_flops.py · bootstrap_all.sh
data/       the dataset (not in the repo)
outputs/    checkpoints, logs and results, one directory per run
```

### src/cardd

- **`models/`** — the three interchangeable pieces of the detector, each
  selected by name from the config:
  - `backbones/` ConvNeXt-V2 (pico / tiny / base) producing the FPN feature
    pyramid. `model.backbone`.
  - `necks/` optional deformable convolution (DCNv2, modulated) applied to
    chosen pyramid levels. `model.neck`.
  - `detectors/mask_rcnn.py` `build_detector(cfg)`, the single entry point.
    `model.detector.name` selects `mask_rcnn`, `cascade_mask_rcnn` (three box
    stages with rising IoU thresholds) or `cascade_mask_rcnn_focal` (the same
    with focal loss on classification).
- **`data/`** — dataset, transforms and the augmentation system: the COCO
  dataset wrapper, the registry-built per-sample transforms, and `augs/` with
  the composition operators (radial, mosaic, mosaic_crop, copy_paste, mixup)
  plus the scheduler that moves their probabilities and intensities over
  training.
- **`engine/`** — the training loop, inference, and tile fusion for the tiled
  inference path.
- **`evaluation/`** — COCO AP, per-class and per-scale breakdowns, boundary
  metrics, confusion matrices, and bootstrap confidence intervals.
- **`utils/`** — config loading and `_base_` resolution, logging, and the
  registry that lets configs name components instead of importing them.
