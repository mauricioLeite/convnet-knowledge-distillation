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

echo "ALL ABLATION RUNS DONE"
