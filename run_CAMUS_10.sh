#!/bin/bash
# RSCL - CAMUS Experiments (5%, 10%, 20% labeling ratios)
# Usage: bash run_CAMUS.sh [GPU_ID]

GPU=${1:-0}
MODE=${2:-"train"}
DATA_DIR="/public/home/zhsy/data/zhsy_data/database_nifti"
TEST_DIR="/public/home/zhsy/data/zhsy_data/database_nifti"
OUT_BASE="/public/home/zhsy/data/zhsy_data/RSCL/CAMUS"

# CAMUS: 4 classes (BG, LV, Myo, LA)

# ========== 10% labeling ratio ==========
echo "===== CAMUS 10% ====="
python train.py \
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
    --mode ${MODE}


echo "===== All CAMUS experiments done ====="
