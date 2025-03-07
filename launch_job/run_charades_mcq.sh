#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1 -C a100_80
#SBATCH --time=05:00:00 
#SBATCH --job-name=run
#SBATCH --export=None

cd /home/hpc/g102ea/g102ea18/folder/lazypatches/
module add python
conda activate m3project
source .env/bin/activate

drop_percent=$((100 - $(echo "$2 * 100" | bc | awk '{print int($1)}')))

output_dir="${4}_${3}_${1}_${drop_percent}%_${5}"

mkdir -p "$output_dir"

python3 inference/run_sub_action.py "$1" "$2" "$3" "$4" "$5"> "$output_dir/output.out" 2> "$output_dir/eval.log"