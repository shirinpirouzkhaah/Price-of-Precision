#!/bin/bash

jobname="CodeLLama_S1_P7_TimeLevel"

sbatch --job-name="$jobname" \
       --output="sbatchlogs/${jobname}.out" \
       --error="sbatchlogs/${jobname}.err" \
       <<EOF
#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH --mail-user=shirin@ifi.uzh.ch
#SBATCH --mail-type=end,fail
#SBATCH --constraint=A100
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-gpu=4

# ------------------ ENVIRONMENT CLEANUP ------------------
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE
unset http_proxy
unset https_proxy
unset all_proxy

export HF_HOME=/scratch/kkitsi/.cache/huggingface
export TRANSFORMERS_CACHE=\$HF_HOME/transformers
export HF_DATASETS_CACHE=\$HF_HOME/datasets
export HF_METRICS_CACHE=\$HF_HOME/metrics
export BITSANDBYTES_NOWELCOME=1
export TRITON_CACHE_DIR=/scratch/kkitsi/.cache/triton_cache
export HUGGINGFACE_HUB_TOKEN=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

# ------------------ MODULES AND ENV ------------------
module load mamba
module load gpu

# Load conda manually
source /apps/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.3.0/mamba-*/etc/profile.d/conda.sh
conda activate /scratch/kkitsi/envs/TinLora

# ------------------ SET MASTER PORT ------------------
MASTER_PORT=\$((12000 + RANDOM % 10000))
echo "Using MASTER_PORT=\$MASTER_PORT"

# ------------------ RUN TRAINING ------------------
python config.py
TRANSFORMERS_CACHE=\$TRANSFORMERS_CACHE torchrun --nproc_per_node=2 --master_port=\$MASTER_PORT Training.py --config config.yml
EOF

