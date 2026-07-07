#!/usr/bin/env bash

set -euo pipefail
cd /workspace

IDS='arch6_6conv_res__*_pre_gap'
COMMON="--ids ${IDS} --teachers resnet convnext vgg --datasets oxford-pets flowers-102 tiny-imagenet-200 --num-workers 6"

run () {
  local tag="$1"; shift
  echo "############ TRAIN tag=${tag}  args: $* ############"
  python src/phase2/train_students.py ${COMMON} --tag "${tag}" "$@"
  echo "############ TEST  tag=${tag} ############"
  python src/phase3/test_students.py --datasets oxford-pets flowers-102 tiny-imagenet-200 --tag "${tag}"
}

# 1) pure feature imitation (no classifier signal during training)
run mse_only      --mse-weight 1 --ce-weight 0 --kd-weight 0
# 2) feature + CE + softened-logit KD
run mse_ce_kd     --mse-weight 1 --ce-weight 1 --kd-weight 1 --T 4
# 3) logit-only Hinton baseline (no feature matching)
run ce_kd         --mse-weight 0 --ce-weight 1 --kd-weight 1 --T 4
# 4) feature + CE + neighbourhood-restricted relational KD (NRKD)
run mse_ce_nrkd   --mse-weight 1 --ce-weight 1 --rkd-weight 25

# 5) two-stage feature + CE + KD:
#    phase 1: MSE-only for 15 epochs;
#    phase 2: MSE + 0.1*CE + 0.8*KD for 30 epochs.
run mse_kd_2stage \
  --no-freeze-classifier \
  --phase1-epochs 15 \
  --phase1-eta-min 1e-4 \
  --epochs 30 \
  --encoder-lr 1e-3 \
  --classifier-lr 1e-5 \
  --eta-min 1e-6 \
  --mse-weight 1 \
  --kd-weight 0.8 \
  --ce-weight 0.1 \
  --T 4

# 6) single-stage CE + KD + NRKD with the classifier fine-tuned.
run ce_kd_nrkd_1stage \
  --no-freeze-classifier \
  --phase1-epochs 0 \
  --epochs 40 \
  --encoder-lr 1e-3 \
  --classifier-lr 1e-5 \
  --eta-min 1e-6 \
  --mse-weight 0 \
  --ce-weight 1 \
  --kd-weight 2 \
  --rkd-weight 25 \
  --T 4

echo "ALL ABLATION RUNS DONE"
