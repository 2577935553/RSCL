#!/bin/bash
# RSCL - LA Experiments (5%, 10%, 20% labeling ratios)
# Usage: bash run_LA.sh [GPU_ID]

GPU=0
MODE="train"
DATA_DIR="/public/home/zhsy/data/MnM-2/all"
TEST_DIR="/public/home/zhsy/data/MnM-2/all"
OUT_BASE="/public/home/zhsy/data/zhsy_data/RSCL/LA"

# LA: 4 classes (BG, RVC, Myo, LVC), 100 train / 50 test
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


echo "===== LA 5% ====="
nohup python train.py \
    --data_dir ${DATA_DIR} \
    --train_data_csv ./data_dir/train_LA.csv \
    --valid_data_csv ./data_dir/valid_LA.csv \
    --test_data_dir ${TEST_DIR} \
    --test_data_list ./data_dir/test_LA.csv \
    --train_output_dir ${OUT_BASE} \
    --test_output_dir ${OUT_BASE} \
    --label_ratio 0.05 \
    --num_classes 4 \
    --batch_size 4 \
    --max_iterations 30000 \
    --learning_rate 0.02 \
    --lambda_dgpc 0.2 \
    --lambda_ucps 1.0 \
    --aug_mode la \
    --ensemble \
    --device ${GPU} \
    --title _5pct \
    --mode ${MODE} &

echo "MODE:${MODE} &"
echo "===== LA 10% ====="
nohup python train.py \
    --data_dir ${DATA_DIR} \
    --train_data_csv ./data_dir/train_LA.csv \
    --valid_data_csv ./data_dir/valid_LA.csv \
    --test_data_dir ${TEST_DIR} \
    --test_data_list ./data_dir/test_LA.csv \
    --train_output_dir ${OUT_BASE} \
    --test_output_dir ${OUT_BASE} \
    --label_ratio 0.1 \
    --num_classes 4 \
    --batch_size 4 \
    --max_iterations 30000 \
    --learning_rate 0.02 \
    --lambda_dgpc 0.2 \
    --lambda_ucps 1.0 \
    --aug_mode la \
    --ensemble \
    --device ${GPU} \
    --title _10pct \
    --mode ${MODE} &

echo "===== LA 20% ====="
nohup python train.py \
    --data_dir ${DATA_DIR} \
    --train_data_csv ./data_dir/train_LA.csv \
    --valid_data_csv ./data_dir/valid_LA.csv \
    --test_data_dir ${TEST_DIR} \
    --test_data_list ./data_dir/test_LA.csv \
    --train_output_dir ${OUT_BASE} \
    --test_output_dir ${OUT_BASE} \
    --label_ratio 0.2 \
    --num_classes 4 \
    --batch_size 4 \
    --max_iterations 30000 \
    --learning_rate 0.02 \
    --lambda_dgpc 0.2 \
    --lambda_ucps 1.0 \
    --aug_mode la \
    --ensemble \
    --device ${GPU} \
    --title _20pct \
    --mode ${MODE} &

echo "===== All LA experiments done ====="
