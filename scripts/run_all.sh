#!/bin/bash
# Train all backbones sequentially, then build the comparison.
# Small batch by default to coexist with other jobs on the shared GB10.
#   bash scripts/run_all.sh          # batch 16
#   BS=64 bash scripts/run_all.sh    # when the GPU is free
set -u
cd "$(dirname "$0")/.."
source /etc/miniconda3/etc/profile.d/conda.sh && conda activate acp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BS=${BS:-16}
WK=${WK:-4}

for cfg in resnet50 convnext_tiny vit_small; do
  echo "================ TRAIN $cfg (bs=$BS) ================"
  python scripts/train.py --config "configs/$cfg.yaml" --batch-size "$BS" --workers "$WK" \
    || echo "!!!! FAILED: $cfg"
done

echo "================ COMPARE ================"
python scripts/compare_models.py || true
echo "================ ALL DONE ================"
