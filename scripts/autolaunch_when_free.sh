#!/bin/bash
# Watch the shared GB10 and auto-launch full training once the GPU can host a
# training-sized allocation (i.e. seonglae's vLLM has freed enough memory).
# Designed to run in the background; polls every 5 min, a good neighbour.
#   THRESH_GB=6 BS=64 bash scripts/autolaunch_when_free.sh
set -u
cd "$(dirname "$0")/.."
source /etc/miniconda3/etc/profile.d/conda.sh && conda activate acp
THRESH_GB=${THRESH_GB:-6}
BS=${BS:-48}

probe() {
  # Succeeds (exit 0) only if we can allocate THRESH_GB on the GPU.
  python - "$THRESH_GB" <<'PY' 2>/dev/null
import sys, torch
gb = float(sys.argv[1])
x = torch.empty(int(gb * 1e9 // 2), dtype=torch.float16, device="cuda")
del x; torch.cuda.empty_cache()
PY
}

echo "[$(date +%F\ %T)] waiting for >=${THRESH_GB}GB free on GB10 ..."
until probe; do
  echo "[$(date +%H:%M)] GPU still busy (vLLM active); retry in 5 min"
  sleep 300
done
echo "[$(date +%F\ %T)] GPU free enough — launching training (bs=$BS)"
BS="$BS" bash scripts/run_all.sh
