# Reliability-Stratified Contrastive Learning for Semi-Supervised Cardiac Image Segmentation

Official PyTorch implementation of RSCL.

## Requirements

```bash
pip install torch torchvision
pip install nibabel medpy scipy opencv-python
pip install tensorboardX pandas tqdm matplotlib scikit-image
```

## Data Preparation

Organize your data as NIfTI (`.nii.gz`) files and list them in CSV files under `data_dir/`. Each CSV row: `subject_path,label_path` (relative to `--data_dir`).

CSV files for each dataset split are already provided in `data_dir/`.

## Training & Testing

### Quick Start

```bash
# ACDC (cardiac MRI)
bash run_ACDC.sh [GPU_ID]

# CAMUS (echocardiography)
bash run_CAMUS.sh [GPU_ID]

# Multi-modal (MM2)
bash run_MM2.sh [GPU_ID]

# Ablation studies
bash run_ablation.sh [GPU_ID]
```

### Single Run

```bash
# Train
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
python train.py --mode test [other args as above]
```

### Compute Metrics

Run from the project root after testing:

```bash
python calculate_metrics_ACDC_batch.py
python calculate_metrics_Echo_batch.py
python calculate_metrics_MM2_batch.py
```

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--label_ratio` | `0.1` | Fraction of labeled subjects (0.05 / 0.1 / 0.2) |
| `--num_classes` | `4` | Number of segmentation classes |
| `--max_iterations` | `30000` | Training iterations |
| `--image_size` | `224 224` | Input resolution (must be divisible by 32) |
| `--lambda_dgpc` | `0.2` | Weight for DGPC loss |
| `--lambda_ucps` | `1.0` | Weight for UCPS loss |
| `--aug_mode` | `acdc` | Augmentation preset: `acdc`, `camus`, or `la` |
| `--contrastive_mode` | `dual` | RSCL mode: `dual` / `hard_only` / `soft_only` / `all_hard` |
| `--baseline_cps` | `False` | Use vanilla unweighted CPS (ablation baseline) |
| `--ensemble` | `False` | Use model ensemble at test time |
| `--device` | `0` | GPU index |
| `--seed` | `42` | Random seed |
