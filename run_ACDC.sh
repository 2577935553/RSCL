#!/bin/bash
# RSCL - ACDC Experiments (5%, 10%, 20% labeling ratios)
# Usage: bash run_ACDC.sh [GPU_ID]

GPU=0
MODE="train"
DATA_DIR="/public/home/zhsy/data/zhsy_data/training"
TEST_DIR="/public/home/zhsy/data/zhsy_data/testing"
OUT_BASE="/public/home/zhsy/data/zhsy_data/RSCL/ACDC"

# ACDC: 4 classes (BG, RVC, Myo, LVC), 100 train / 50 test

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift ;;
        --mode) MODE="$2"; shift ;;
        --data_dir) DATA_DIR="$2"; shift ;;
        --test_dir) TEST_DIR="$2"; shift ;;
        --out_base) OUT_BASE="$2"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done


# ========== 5% labeling ratio ==========
echo "===== ACDC 5% ====="
nohup python train.py \
    --data_dir ${DATA_DIR} \
    --train_data_csv ./data_dir/train_ACDC.csv \
    --valid_data_csv ./data_dir/valid_ACDC.csv \
    --test_data_dir ${TEST_DIR} \
    --test_data_list ./data_dir/test_subj_ACDC.csv \
    --train_output_dir ${OUT_BASE} \
    --test_output_dir ${OUT_BASE} \
    --label_ratio 0.05 \
    --num_classes 4 \
    --batch_size 4 \
    --max_iterations 30000 \
    --learning_rate 0.02 \
    --lambda_dgpc 0.2 \
    --lambda_ucps 1.0 \
    --aug_mode acdc \
    --ensemble \
    --device ${GPU} \
    --title _5pct \
    --mode ${MODE} &

# ========== 10% labeling ratio ==========
echo "===== ACDC 10% ====="
nohup python train.py \
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
    --mode ${MODE} &

# ========== 20% labeling ratio ==========
echo "===== ACDC 20% ====="
nohup python train.py \
    --data_dir ${DATA_DIR} \
    --train_data_csv ./data_dir/train_ACDC.csv \
    --valid_data_csv ./data_dir/valid_ACDC.csv \
    --test_data_dir ${TEST_DIR} \
    --test_data_list ./data_dir/test_subj_ACDC.csv \
    --train_output_dir ${OUT_BASE} \
    --test_output_dir ${OUT_BASE} \
    --label_ratio 0.2 \
    --num_classes 4 \
    --batch_size 4 \
    --max_iterations 30000 \
    --learning_rate 0.02 \
    --lambda_dgpc 0.2 \
    --lambda_ucps 1.0 \
    --aug_mode acdc \
    --ensemble \
    --device ${GPU} \
    --title _20pct \
    --mode ${MODE} &

echo "===== All ACDC experiments done ====="
