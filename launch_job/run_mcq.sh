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


output_dir="${4}_${3}_${1}_${6}_${2}%_${5}"

mkdir -p "$output_dir"

#argument list by order: [LLM_FPS] [RETENTION_RATE] [SAMPLER_TYPE] [DATASET] [HYPERPARAM] [DROPPING_POSITION]
# [LLM_FPS] : [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2]; CHANGES ONLY FOR BASELINE W/O SAMPLING + ANALYSIS EXP FOR BEST SAMPLING RATE + SAMPLING POSITION
# [RETENTION_RATE] : [.10, .30, .50, .70, .90] CHANGES FOR ALL EXPERIMENTS BAR BEST SAMPLING RATE EXPERIMENT
# [SAMPLER_TYPE] : [uniform, st_gaussian, km_closest] 
# [DATASET] : [Charades, PerceptionTest]
# [HYPERPARAM] : 
# --For uniform sampler: [0]
# --For st_gaussian sampler: [0.5, 1, 10]
# --For km_closest sampler: [0.25, 0.50, 0.75, 1.0]
# [DROPPING_POSITION] : [0, 12, 24] 
# Example commands: 
# --uniform sampler: sbatch launch_job/run_mcq.sh 1.0 .50 uniform Charades 0 0 {Dataset is Charades, Hyperparam  is 0, retention_rate is 0.50, llm_fps is 1.0, dropping_position is 0}
# --st_gaussian sampler: sbatch launch_job/run_mcq.sh 1.0 .50 st_gaussian Charades 1 0 {Dataset is Charades, Hyperparam  is 1, retention_rate is 0.50, llm_fps is 1.0, dropping_position is 0}
# --km_closest sampler: sbatch launch_job/run_mcq.sh 1.0 .50 km_closest Charades 0.50 0 {Dataset is Charades, Hyperparam  is 0.5, retention_rate is 0.50, llm_fps is 1.0, dropping_position is 0}

python3 inference/run_sub_action.py "$1" "$2" "$3" "$4" "$5" "$6"> "$output_dir/output.out" 2> "$output_dir/eval.log"