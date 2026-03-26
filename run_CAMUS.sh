#!/bin/bash
# RSCL - CAMUS Experiments (5%, 10%, 20% labeling ratios)
# Usage: bash run_CAMUS.sh [GPU_ID]

GPU=0
MODE="train"
DATA_DIR="/public/home/zhsy/data/zhsy_data/database_nifti"
TEST_DIR="/public/home/zhsy/data/zhsy_data/database_nifti"
OUT_BASE="/public/home/zhsy/data/zhsy_data/RSCL/CAMUS"

# CAMUS: 4 classes (BG, LV, Myo, LA)

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
echo "===== CAMUS 5% ====="
nohup python train.py \
    --data_dir ${DATA_DIR} \
    --train_data_csv ./data_dir/train_Echo_4CH.csv \
    --valid_data_csv ./data_dir/valid_Echo_4CH.csv \
    --test_data_dir ${TEST_DIR} \
    --test_data_list ./data_dir/test_Echo_4CH.csv \
    --train_output_dir ${OUT_BASE} \
    --test_output_dir ${OUT_BASE} \
    --label_ratio 0.05 \
    --num_classes 4 \
    --batch_size 4 \
    --max_iterations 30000 \
    --learning_rate 0.02 \
    --lambda_dgpc 0.2 \
    --lambda_ucps 1.0 \
    --aug_mode camus \
    --ensemble \
    --device ${GPU} \
    --title _5pct \
    --mode ${MODE} &

# ========== 10% labeling ratio ==========
echo "===== CAMUS 10% ====="
nohup python train.py \
    --data_dir ${DATA_DIR} \
    --train_data_csv ./data_dir/train_Echo_4CH.csv \
    --valid_data_csv ./data_dir/valid_Echo_4CH.csv \
    --test_data_dir ${TEST_DIR} \
    --test_data_list ./data_dir/test_Echo_4CH.csv \
    --train_output_dir ${OUT_BASE} \
    --test_output_dir ${OUT_BASE} \
    --label_ratio 0.1 \
    --num_classes 4 \
    --batch_size 4 \
    --max_iterations 30000 \
    --learning_rate 0.02 \
    --lambda_dgpc 0.2 \
    --lambda_ucps 1.0 \
    --aug_mode camus \
    --ensemble \
    --device ${GPU} \
    --title _10pct \
    --mode ${MODE} &

# ========== 20% labeling ratio ==========
echo "===== CAMUS 20% ====="
nohup python train.py \
    --data_dir ${DATA_DIR} \
    --train_data_csv ./data_dir/train_Echo_4CH.csv \
    --valid_data_csv ./data_dir/valid_Echo_4CH.csv \
    --test_data_dir ${TEST_DIR} \
    --test_data_list ./data_dir/test_Echo_4CH.csv \
    --train_output_dir ${OUT_BASE} \
    --test_output_dir ${OUT_BASE} \
    --label_ratio 0.2 \
    --num_classes 4 \
    --batch_size 4 \
    --max_iterations 30000 \
    --learning_rate 0.02 \
    --lambda_dgpc 0.2 \
    --lambda_ucps 1.0 \
    --aug_mode camus \
    --ensemble \
    --device ${GPU} \
    --title _20pct \
    --mode ${MODE} &

echo "===== All CAMUS experiments done ====="
