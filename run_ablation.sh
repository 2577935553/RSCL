#!/bin/bash
# RSCL Ablation Studies on ACDC 10%
# Usage: bash run_ablation.sh [GPU_ID]

GPU=${1:-0}
DATA_DIR="/public/home/zhsy/data/zhsy_data/training"
TEST_DIR="/public/home/zhsy/data/zhsy_data/testing"
OUT_BASE="/public/home/zhsy/data/zhsy_data/RSCL/ACDC_ablation"

COMMON="--data_dir ${DATA_DIR} \
    --train_data_csv ./data_dir/train_ACDC.csv \
    --valid_data_csv ./data_dir/valid_ACDC.csv \
    --test_data_dir ${TEST_DIR} \
    --test_data_list ./data_dir/test_subj_ACDC.csv \
    --train_output_dir ${OUT_BASE} \
    --test_output_dir ${OUT_BASE} \
    --label_ratio 0.1 --num_classes 4 --batch_size 4 \
    --max_iterations 30000 --learning_rate 0.02 \
    --aug_mode acdc --ensemble --device ${GPU} --mode train"

# ===== Table 3: Module-level ablation =====

# Vanilla CPS baseline (standard unweighted CPS, no contrastive)
echo "===== Vanilla CPS ====="
python train.py ${COMMON} --lambda_dgpc 0.0 --lambda_ucps 1.0 --baseline_cps --title _ablation_vanilla

# + RSPA only (reliability-weighted UCPS, no contrastive)
echo "===== +RSPA only ====="
python train.py ${COMMON} --lambda_dgpc 0.0 --lambda_ucps 1.0 --title _ablation_rspa_only

# + DGPC only (standard CPS + contrastive)
echo "===== +DGPC only ====="
python train.py ${COMMON} --lambda_dgpc 0.2 --lambda_ucps 1.0 --baseline_cps --title _ablation_dgpc_only

# Full model (UCPS + DGPC)
echo "===== Full model ====="
python train.py ${COMMON} --lambda_dgpc 0.2 --lambda_ucps 1.0 --title _ablation_full

# ===== Table 5: Hyperparameter sensitivity =====

# lambda_dgpc variants
for LD in 0.1 0.5; do
    echo "===== lambda_dgpc=${LD} ====="
    python train.py ${COMMON} --lambda_dgpc ${LD} --lambda_ucps 1.0 --title _sens_ld${LD}
done

# lambda_ucps variants
for LU in 0.5 2.0; do
    echo "===== lambda_ucps=${LU} ====="
    python train.py ${COMMON} --lambda_dgpc 0.2 --lambda_ucps ${LU} --title _sens_lu${LU}
done

# tau_high_final variants
for TH in 0.75 0.95; do
    echo "===== tau_high_final=${TH} ====="
    python train.py ${COMMON} --lambda_dgpc 0.2 --lambda_ucps 1.0 --tau_high_final ${TH} --title _sens_th${TH}
done

# tau_low_final variants
for TL in 0.2 0.4; do
    echo "===== tau_low_final=${TL} ====="
    python train.py ${COMMON} --lambda_dgpc 0.2 --lambda_ucps 1.0 --tau_low_final ${TL} --title _sens_tl${TL}
done

# tau_soft (soft temperature) variants
for TS in 0.3 1.0; do
    echo "===== tau_soft=${TS} ====="
    python train.py ${COMMON} --lambda_dgpc 0.2 --lambda_ucps 1.0 --tau_soft ${TS} --title _sens_ts${TS}
done

# ===== Table 4: Dual-granularity ablation =====

echo "===== all_hard (no filtering) ====="
python train.py ${COMMON} --lambda_dgpc 0.2 --lambda_ucps 1.0 --contrastive_mode all_hard --title _dg_all_hard

echo "===== all_hard_filtered (anchor+learn=hard, excl=none) ====="
python train.py ${COMMON} --lambda_dgpc 0.2 --lambda_ucps 1.0 --contrastive_mode all_hard_filtered --title _dg_all_hard_filt

echo "===== hard_only (anchor=hard, learn+excl=none) ====="
python train.py ${COMMON} --lambda_dgpc 0.2 --lambda_ucps 1.0 --contrastive_mode hard_only --title _dg_hard_only

echo "===== soft_only (anchor+learn=soft, excl=none) ====="
python train.py ${COMMON} --lambda_dgpc 0.2 --lambda_ucps 1.0 --contrastive_mode soft_only --title _dg_soft_only

echo "===== dual (ours, default) ====="
python train.py ${COMMON} --lambda_dgpc 0.2 --lambda_ucps 1.0 --contrastive_mode dual --title _dg_dual

echo "===== Ablation experiments done ====="
