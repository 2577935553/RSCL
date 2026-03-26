#!/bin/bash
# RSCL - ACDC Experiments (5%, 10%, 20% labeling ratios)
# Usage: bash run_ACDC.sh [GPU_ID]

GPU=${1:-0}
DATA_DIR="/public/home/zhsy/data/zhsy_data/training"
TEST_DIR="/public/home/zhsy/data/zhsy_data/testing"
OUT_BASE="/public/home/zhsy/data/zhsy_data/RSCL/ACDC"

# ACDC: 4 classes (BG, RVC, Myo, LVC), 100 train / 50 test

# ========== 10% labeling ratio ==========
echo "===== ACDC 10% ====="
python train.py \
    --data_dir ${DATA_DIR} \
    --train_data_csv ./data_dir/train_ACDC.csv \
    --valid_data_csv ./data_dir/valid_ACDC.csv \
    --test_data_dir ${TEST_DIR} \
    --test_data_list ./data_dir/test_subj_ACDC.csv \
    --train_output_dir ${OUT_BASE} \
    --test_output_dir ${OUT_BASE} \
    --label_ratio 0.1 \
    --num_classes 4 \
    --batch_size 4 \
    --max_iterations 30000 \
    --learning_rate 0.02 \
    --lambda_dgpc 0.2 \
    --lambda_ucps 1.0 \
    --aug_mode acdc \
    --ensemble \
    --device ${GPU} \
    --title _10pct \
    --mode train

echo "===== All ACDC experiments done ====="
