# TG-Diff

Text-Guided Spatio-Frequency Collaborative Diffusion for Salient Object Detection in Optical Remote Sensing Images.

This is a preliminary workflow release. Model implementation and trained weights
will be released after acceptance. The current `model/TGDiff.py` is an interface
placeholder: training and inference are intentionally unavailable until that release.
This version has not undergone end-to-end training or inference validation.

## Setup

Use Python 3.10 or later and install dependencies:

```bash
pip install -r requirements.txt
```

Run commands from this directory. Configure data paths in `configs/default.json`.
Manifest examples are in `examples/`; image and mask paths are relative to the
manifest location. Example records are illustrative and contain no real data.

Training accepts `eight_way` or `none` augmentation. Eight-way augmentation requires
reviewed descriptions for all seven transformed views in addition to the original.
Rotations are counterclockwise; `flip_*` applies a horizontal reflection after rotation.
Use `none` for already augmented image/mask/text records. Test descriptions must
come from RGB images alone; test-input manifests contain no ground-truth paths.

## Workflow

The first two commands require the forthcoming model implementation:

```bash
python train.py --config configs/default.json
python infer.py --checkpoint outputs/run_01/final.pt --manifest dataset/test.json --output predictions/run_01
python evaluate.py --pred-dir predictions/run_01 --gt-dir dataset/test/masks --output outputs/metrics.json
```

Training saves periodic snapshots and the final epoch. It does not evaluate test
data or select checkpoints from test scores. Inference consumes one explicitly
specified checkpoint. Evaluation processes one prediction set and does not rank
epochs, seeds, or checkpoints. `F_max` and `E_max` refer only to threshold-based metrics.

## Acknowledgments

The workflow uses [PyTorch](https://pytorch.org/),
[Transformers](https://github.com/huggingface/transformers),
[timm](https://github.com/huggingface/pytorch-image-models), and
[PySODMetrics](https://github.com/lartpang/PySODMetrics).
Datasets and pretrained assets are obtained separately under their original terms.
