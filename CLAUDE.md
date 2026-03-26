# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running Experiments

```bash
# Train + test on ACDC dataset (cardiac MRI)
bash run_ACDC.sh [GPU_ID]

# Train + test on CAMUS dataset (echocardiography)
bash run_CAMUS.sh [GPU_ID]

# Multi-modal (MM2) dataset
bash run_MM2.sh [GPU_ID]

# Ablation studies (ACDC 10%, sequential runs)
bash run_ablation.sh [GPU_ID]

# Single run (train then test)
python train.py \
    --data_dir /path/to/training \
    --train_data_csv ./data_dir/train_ACDC.csv \
    --valid_data_csv ./data_dir/valid_ACDC.csv \
    --test_data_dir /path/to/testing \
    --test_data_list ./data_dir/test_subj_ACDC.csv \
    --train_output_dir /path/to/output \
    --test_output_dir /path/to/output \
    --label_ratio 0.1 \
    --num_classes 4 \
    --device 0 \
    --mode train

# Test only
python train.py --mode test [other args]

# Post-hoc metric computation (run from project root after testing)
python calculate_metrics_ACDC_batch.py
python calculate_metrics_Echo_batch.py
python calculate_metrics_MM2_batch.py
```

## Key Arguments (`train.py`)

| Argument | Description |
|---|---|
| `--label_ratio` | Fraction of labeled data (0.05 / 0.1 / 0.2) |
| `--num_classes` | Number of segmentation classes (4 for ACDC/CAMUS) |
| `--lambda_dgpc` | Weight for dual-granularity prototype contrastive loss (default 0.2) |
| `--lambda_ucps` | Weight for uncertainty-guided cross-pseudo-supervision loss (default 1.0) |
| `--aug_mode` | Augmentation preset: `acdc`, `camus`, or `la` |
| `--ensemble` | Use model ensemble at test time |
| `--title` | Suffix appended to checkpoint/output directory names |
| `--contrastive_mode` | RSCL mode: `dual` / `hard_only` / `soft_only` / `all_hard` / `all_hard_filtered` |
| `--baseline_cps` | Use vanilla unweighted CPS (disables reliability-weighted UCPS); used in ablations |
| `--seed` | Random seed (default 42) |
| `--max_iterations` | Training iterations (default 30000) |
| `--image_size` | Input resolution (must be divisible by 32) |

## Architecture

### Training Loop (`train.py`)
Uses a **dual-network (CPS-style)** setup: two independent `ProjectUNet1` instances (`model1`, `model2`) trained jointly. Forward pass for both labeled and unlabeled batches at each iteration. Supervised loss = 0.5×CE + Dice on labeled data. RSCL losses are computed from the 128-ch decoder features (`dx`) of both networks. First 1000 iterations are warmup (only sup + UCPS, DGPC disabled).

### Model (`SegModel.py`)
- **`ProjectUNet1`** — primary model: ResNet50 encoder + UNet decoder + projection head. `forward()` returns `(seg_logits, proj_feat_128ch, encoder_feat)` (3-tuple); training uses positions `[0]`, `[1]`, `[2]` as `y, dx, _`.
- **`ProjectUNet_Vx`** — variant with multi-scale segmentation heads and projectors at each decoder level (used for ablation, returns 10-tuple).
- Encoder via `encoders/get_encoder()` (ResNet variants with ImageNet weights). Decoder via `decoders/UnetDecoder`.

### Base Classes (`base/base_model.py`)
- **`SegmentationModel`** — base for `ProjectUNet`; `forward()` returns `(masks, proj_feat)` or `(masks, proj_feat, encoder_feat)` with `out_features=True`.
- **`SegmentationModel1`** — base for `ProjectUNet1`; always returns the 3-tuple.
- **`SegmentationModel_v2`** — base for `ProjectUNet_Vx`; multi-scale outputs.
- All enforce input spatial dims divisible by 32 (`check_input_shape`).

### RSCL Module (`rscl.py`)
The core semi-supervised learning algorithm with two sub-modules:
- **Module A — Reliability Estimation**: Computes per-pixel reliability score from two networks' softmax outputs using prediction agreement, average max confidence, and Jensen-Shannon divergence. Pixels are stratified into high/medium/low reliability zones with annealed thresholds.
- **Module B — Dual-Granularity Prototype Contrastive (DGPC)**: Maintains a per-class EMA memory bank (`[C, feat_dim]`, L2-normalized). Hard zone pixels use InfoNCE loss; medium zone pixels use soft KL divergence against the average prediction.
- **UCPS**: Reliability-weighted cross-pseudo-supervision — uses only high-reliability pixels from one network's pseudo-labels to supervise the other. With `--baseline_cps`, all pixels are used (no weighting).

### Data Pipeline (`utilities/`)
- `MyDataSet.py`: `SemiSegDataset_1` / `SemiSegDataset_2` handle labeled/unlabeled splits via `label_ratio`. Data loaded from NIfTI (`.nii.gz`) files listed in CSV files under `data_dir/`.
- `Load_Data_v2.py`: `augment_data_batch` — per-batch augmentation during training (rotation, scale, shift; different strengths for labeled vs unlabeled).
- `val_2D.py`: Slice-by-slice 2D inference on 3D volumes with center-crop preprocessing.
- `losses.py`: Dice loss variants, entropy loss, KL divergence, and `DiceLoss` class.
- `ramps.py`: Sigmoid/cosine ramp-up schedules used for annealing thresholds.

## Data Format
- CSV files in `data_dir/` list relative paths to NIfTI subjects.
- Each CSV row: `subject_path, label_path` (relative to `--data_dir`).
- Separate CSVs for each dataset split: `train_ACDC.csv`, `valid_ACDC.csv`, `test_subj_ACDC.csv`.

## Supported Datasets
ACDC (cardiac MRI, 4 classes), CAMUS / Echo_4CH (echocardiography, 4 classes), WHS, MyoSAIQ, UTAH, LA, CETUS, Philips, SA.
