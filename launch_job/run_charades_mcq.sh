#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1 -C a100_80
#SBATCH --time=00:10:00 
#SBATCH --job-name=run
#SBATCH --output=/home/hpc/g102ea/g102ea16/lazypatches/eval.log
#SBATCH --export=None

module add python
conda activate m3project
source .env/bin/activate

python3 inference/run.py > output.out