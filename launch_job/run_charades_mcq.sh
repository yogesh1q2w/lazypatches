#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1 -C a100_80
#SBATCH --time=05:00:00 
#SBATCH --job-name=run
#SBATCH --output=/home/hpc/g102ea/g102ea18/folder/lazypatches/eval.log
#SBATCH --export=None
cd /home/hpc/g102ea/g102ea18/folder/lazypatches/
module add python
conda activate m3project
source .env/bin/activate

python3 inference/run_sub_action.py > output@40.out