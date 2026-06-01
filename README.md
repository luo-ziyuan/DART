# DART

This repository contains the public release of DART and the image-space baseline used in the paper.

Included:
- `train_robust_inr.py`: DART training and evaluation under INR-parameter attacks.
- `train_robust_image.py`: image-space robust baseline used in the same evaluation pipeline.
- `configs/`: released configs.
- `noise_layers/`, `run_nerf_helpers.py`, `jpeg.py`, `utils.py`, `utils_img.py`: runtime dependencies for the released pipeline.
- `Decoder/ckpt/other_dec_48b_whit.torchscript.pt`: 48-bit decoder checkpoint required by the released configs.

Excluded on purpose:
- Draco cluster utilities
- rebuttal-only scripts
- experiment logs, cached files, datasets, and internal exploratory baselines
- optional perceptual-loss assets that were not needed for the released configuration

## Environment

Tested environment:
- Python `3.10`
- PyTorch `2.1`
- CUDA `12.2`

Install the Python dependencies in [requirements.txt](requirements.txt). A CUDA-capable GPU is required by the released scripts.

## Data Layout

The released configs expect an image folder at:

```text
data/test2014/
```

The folder should contain the images to be processed. The repository does not ship the dataset.

## Quick Start

Run DART with the released setting:

```bash
bash scripts/run_dart_image.sh /path/to/test2014
```

Run the released image-space baseline:

```bash
bash scripts/run_robust_image_baseline.sh /path/to/test2014
```

Both commands accept an optional second argument for the output log directory.

## Notes

- The released DART config uses `robust_inr` and `10` PGD steps during training.
- The released public package supports `--loss_i mse`, which is the setting used by the provided image configs.
- In `train_robust_image.py`, the original config naming still uses `method = robust_inr`; this is kept for compatibility with the original script logic.
- Outputs are written under `logs/...`, including `args.txt`, `average_results.json`, `log_stats.json`, TensorBoard logs, and optional saved INR reconstructions.

## Third-Party Notice

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the retained upstream attribution notes.
