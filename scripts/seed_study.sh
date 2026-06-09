#!/bin/bash
# Multi-seed study: train each backbone over K seeds (paired by seed = same data
# split), then run zero-shot China inference for each. Resumable — a run whose
# test_metrics.json AND china_predictions.csv already exist is skipped, so the
# script can be re-launched to fill in any runs that OOM'd or were interrupted.
#
#   bash scripts/seed_study.sh                 # seeds 42..49, all 3 backbones
#   SEEDS="42 43 44" BS=64 bash scripts/seed_study.sh
set -u
cd "$(dirname "$0")/.."
source /etc/miniconda3/etc/profile.d/conda.sh && conda activate acp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEEDS=${SEEDS:-"42 43 44 45 46 47 48 49"}
BACKBONES=${BACKBONES:-"resnet50 convnext_tiny"}   # core claim first; add vit_small to extend
BS=${BS:-64}
WK=${WK:-8}

for seed in $SEEDS; do
  for bb in $BACKBONES; do
    run="${bb}_s${seed}"
    dir="outputs/${run}"
    if [ -f "${dir}/test_metrics.json" ] && [ -f "${dir}/china_predictions.csv" ]; then
      echo "[skip] ${run} (already complete)"
      continue
    fi
    echo "================ TRAIN ${run} ================"
    if [ ! -f "${dir}/test_metrics.json" ]; then
      python scripts/train.py --config "configs/${bb}.yaml" \
        --seed "$seed" --run-name "$run" --batch-size "$BS" --workers "$WK" \
        || { echo "!!!! TRAIN FAILED: ${run}"; continue; }
    fi
    echo "---------------- CHINA ${run} ----------------"
    python scripts/predict_china.py --ckpt "${dir}/best.pt" --backbone "$bb" \
      || echo "!!!! CHINA FAILED: ${run}"
  done
done
echo "================ SEED STUDY DONE ================"
python scripts/analyze_seed_study.py || true
