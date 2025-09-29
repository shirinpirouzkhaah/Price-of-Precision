#!/bin/bash

jobname="T5"

sbatch --job-name="$jobname" \
       --output="sbatchlogs/${jobname}.out" \
       --error="sbatchlogs/${jobname}.err" \
       <<EOF
#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --mail-user=shirin@ifi.uzh.ch
#SBATCH --mail-type=end,fail
#SBATCH --constraint=A100
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
 
module load gpu
module load mamba
source activate torchLLR4


python config.py
python Training.py --config config.yml
EOF