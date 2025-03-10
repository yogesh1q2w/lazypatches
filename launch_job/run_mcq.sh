#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1 -C a100_80
#SBATCH --time=20:00:00 
#SBATCH --job-name=run
#SBATCH --export=None

cd /home/hpc/g102ea/appropriate_path/lazypatches/  # UPDATE ME! AND DON'T PUSH!
module add python
conda activate m3project
source .env/bin/activate

# Positional arguments:
# $1: RETENTION_RATE
# $2: SAMPLER_TYPE
# $3: DATASET
# $4: HYPERPARAM
# $5: DROPPING_POSITION

# [RETENTION_RATE] : [.10, .30, .50, .70, .90] CHANGES FOR ALL EXPERIMENTS BAR BEST SAMPLING RATE EXPERIMENT
# [SAMPLER_TYPE] : [None, uniform, st_gaussian, km_closest, tb] 
# [DATASET] : [Charades, PerceptionTest]
# [HYPERPARAM] : 
# --For uniform sampler: [0]
# --For st_gaussian sampler: [0.5, 1, 10]
# --For km_closest sampler: [0.25, 0.50, 0.75, 1.0]
# [DROPPING_POSITION] : [0, 12, 24] 

# Example commands: 
# --no sampler:
#       sbatch launch_job/run_mcq.sh 0 0 None Charades 0 0 
#       {Dataset is Charades, Hyperparam is 0, retention_rate is 0.50, dropping_position is 0}
# --uniform sampler: 
#       sbatch launch_job/run_mcq.sh .50 uniform Charades 0 0 
#       {Dataset is Charades, Hyperparam is 0, retention_rate is 0.50, dropping_position is 0}
# --st_gaussian sampler: 
#       sbatch launch_job/run_mcq.sh .50 st_gaussian Charades 1 0 
#       {Dataset is Charades, Hyperparam is 1, retention_rate is 0.50, dropping_position is 0}
# --km_closest sampler: 
#       sbatch launch_job/run_mcq.sh .50 km_closest Charades 0.50 0 
#       {Dataset is Charades, Hyperparam is 0.5, retention_rate is 0.50, dropping_position is 0}

TARGET_PATH="/home/atuin/g102ea/shared/group_10/results"

DATASET=$(echo "$3" | tr '[:upper:]' '[:lower:]')
RETENTION_RATE="$1"
SAMPLER_TYPE="$2"
HYPERPARAM="$4"
DROPPING_POSITION="$5"

if [ "$SAMPLER_TYPE" = "None" ]; then
    TARGET_PATH="${TARGET_PATH}/baseline"
elif [ "$SAMPLER_TYPE" = "km_closest" ] && [ "$(echo "$HYPERPARAM != 0.5" | bc -l)" -eq 1 ]; then
    TARGET_PATH="${TARGET_PATH}/ablations"
elif [ "$DATASET" = "charades" ]; then
    TARGET_PATH="${TARGET_PATH}/charades"
elif [ "$DATASET" = "perceptiontest" ]; then
    TARGET_PATH="${TARGET_PATH}/perceptiontest"
fi

retention_percent=$(printf "%.0f" "$(echo "$RETENTION_RATE * 100" | bc -l)")

SUB_FOLDER="${DATASET}_${SAMPLER_TYPE}_${DROPPING_POSITION}_${retention_percent}%_${HYPERPARAM}"
TARGET_PATH="${TARGET_PATH}/${SUB_FOLDER}"

mkdir -p "$TARGET_PATH"

python3 inference/run_subset_mcq.py "$1" "$2" "$3" "$4" "$5" $TARGET_PATH > "$TARGET_PATH/output.out" 2> "$TARGET_PATH/error.log"