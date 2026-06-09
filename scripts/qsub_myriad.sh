#!/bin/bash -l
# UCL Myriad (SGE) batch job for training a backbone.
# Submit from ~/scratch/<project> after `git pull`:
#   qsub scripts/qsub_myriad.sh
# Edit CONFIG below or pass via `qsub -v CONFIG=configs/vit_small.yaml ...`.

#$ -l h_rt=4:00:00              # wall-clock limit
#$ -l mem=32G                   # memory per core
#$ -l gpu=1                     # one GPU
#$ -pe smp 8                    # 8 cores for dataloader workers
#$ -N acp_train
#$ -cwd                         # run from the submission directory
#$ -o logs/                     # stdout -> logs/
#$ -j y                         # merge stderr into stdout

set -euo pipefail
mkdir -p logs

# Myriad module setup (non-login shells don't load these automatically).
module purge
module load default-modules
module load python3/recommended
module load pytorch/2.1.0/gpu   # provides torch+CUDA on Myriad

CONFIG=${CONFIG:-configs/resnet50.yaml}
echo "host=$(hostname) date=$(date) config=$CONFIG"
nvidia-smi || true

python scripts/train.py --config "$CONFIG" --batch-size 64 --workers 8
echo "done."
