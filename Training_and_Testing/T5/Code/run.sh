#!/bin/bash

jobname="T5_S2_P4_ProjectLevel"

sbatch --job-name="$jobname" \
       --output="sbatchlogs/${jobname}.out" \
       --error="sbatchlogs/${jobname}.err" \
       <<EOF
#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH --mail-user=shirin@ifi.uzh.ch
#SBATCH --mail-type=end,fail
#SBATCH --constraint=A100
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-gpu=4

 
module load gpu
module load mamba
source activate torchLLR2


python config.py
python Training.py --config config.yml
EOF
