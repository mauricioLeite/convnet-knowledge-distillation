#!/usr/bin/env bash
# Q4/Q5 loss ablation driver.
# Reduced grid: arch6_6conv_res x {resnet,convnext,vgg} x pre_gap x 2 datasets.
# Baseline mse=1,ce=1 is the existing main run (reused at analysis time).
set -euo pipefail
cd /workspace

IDS='arch6_6conv_res__*_pre_gap'
COMMON="--ids ${IDS} --teachers resnet convnext vgg --datasets oxford-pets flowers-102 --num-workers 6"

run () {
  local tag="$1"; shift
  echo "############ TRAIN tag=${tag}  args: $* ############"
  python src/phase2/train_students.py ${COMMON} --tag "${tag}" "$@"
  echo "############ TEST  tag=${tag} ############"
  python src/phase2/test_students.py --datasets oxford-pets flowers-102 --tag "${tag}"
}

# 1) pure feature imitation (no classifier signal during training)
run mse_only      --mse-weight 1 --ce-weight 0 --kd-weight 0
# 2) feature + CE + softened-logit KD
run mse_ce_kd     --mse-weight 1 --ce-weight 1 --kd-weight 1 --T 4
# 3) logit-only Hinton baseline (no feature matching)
run ce_kd         --mse-weight 0 --ce-weight 1 --kd-weight 1 --T 4
# 4) feature + CE + Relational KD (paper-standard weight)
run mse_ce_rkd    --mse-weight 1 --ce-weight 1 --rkd-weight 25
#5) feature + CE + KD splitted in 2 stages: 1st stage with MSE only, 2nd stage with MSE + 0.1*CE + 0.8*KD
#   - 1st stage: mse-only for 15 epochs with higher lr and eta-min to speed up convergence
#   - 2nd stage: mse+ce+kd for 30 epochs with lower lr and classifier unfrozen to allow better CE+KD convergence
uv run src/phase2/train_students.py \
    --no-freeze-classifier \
    --phase1-epochs 15 \
    --phase1-eta-min 1e-4 \
    --epochs 30 \
    --encoder-lr 1e-3 \
    --classifier-lr 1e-5 \
    --eta-min 1e-6 \
    --mse-weight 1 \
    --kd-weight 0.8 \
    --ce-weight 0.1

echo "ALL ABLATION RUNS DONE"
