# lazypatches
A spatio-temporal video subsampling strategy for efficient video-language modelling. 


# Installation with conda (for development)

## Load python module
Always have to load python as:
```
module load python
```

If this is first time installation, run:
```
if [ ! -f ~/.bash_profile ]; then
  echo "if [ -f ~/.bashrc ]; then . ~/.bashrc; fi" > ~/.bash_profile
fi
module add python
conda config --add pkgs_dirs $WORK/software/private/conda/pkgs
conda config --add envs_dirs $WORK/software/private/conda/envs
```

## Create conda environment (one time only)

```
conda create -n m3project python=3.10.12
```

## Activate conda environment (everytime)

```
conda activate m3project
```

## Create a python virtual environment (one time only)

```
pip install --upgrade pip
python3 -m venv .env
```

## Install all requirements in python venv

```
source .env/bin/activate
pip install -e .
```
Now you are ready to run inference scripts within the venv!\
To test if the installation was successful, run:

```
source .env/bin/activate
python3 inference/run.py
```

In cluster, you can run (notice that cluster activates the venv too):
```
sbatch launch_job/run.sh
```

## For complete experiments on subset of Charades and PerceptionTest:

## Command-Line Arguments

The script accepts the following arguments (in order):

### LLM_FPS:
Frame rate for the model (e.g. 1.0)
### RETENTION_RATE:
Retention rate for the video processor (e.g. 0.50)
### SAMPLER_TYPE:
Sampler type (e.g. uniform, st_gaussian, km_closest, or None for baseline)
### DATASET:
Dataset name (e.g. Charades or PerceptionTest)
### HYPERPARAM:
Hyperparameter value (e.g. 0.50 for km_closest, or 0 for uniform sampler)
### DROPPING_POSITION:
Dropping position (e.g. 0, 12, or 24)

## Execution Command Format

```
sbatch launch_job/run_mcq.sh [LLM_FPS] [RETENTION_RATE] [SAMPLER_TYPE] [DATASET] [HYPERPARAMETER] [DROPPING_POSITION]
```

## Example Command
For example, to run the script using the 'km_closest' sampler on the Charades dataset with the following settings:

LLM_FPS: 1.0
RETENTION_RATE: 0.50
SAMPLER_TYPE: km_closest
DATASET: Charades
HYPERPARAM: 0.50
DROPPING_POSITION: 0

Use the command:
```
sbatch launch_job/run_mcq.sh 1.0 0.50 km_closest Charades 0.50 0
```

## Results Directory Structure

The results are stored under a base directory:
```
/home/atuin/g102ea/shared/group_10/results/
```

Based on the command-line arguments, the output directory is built as follows:

```
<BASE_PATH>/
├── baseline/
│   └── <DATASET>_None_<LLM_FPS>_<DROPPING_POSITION>_<RETENTION_RATE*100>%_<HYPERPARAM>/
│       ├── output.out
│       ├── evaluation.log
│       ├── error.log
│       ├── results.json
│       └── failure.json
├── ablation/
│   └── <DATASET>_km_closest_<LLM_FPS>_<DROPPING_POSITION>_<RETENTION_RATE*100>%_<HYPERPARAM>/
│       ├── output.out
│       ├── evaluation.log
│       ├── error.log
│       ├── results.json
│       └── failure.json
├── charades/
│   └── charades_<SAMPLER_TYPE>_<LLM_FPS>_<DROPPING_POSITION>_<RETENTION_RATE*100>%_<HYPERPARAM>/
│       ├── output.out
│       ├── evaluation.log
│       ├── error.log
│       ├── results.json
│       └── failure.json
└── perceptiontest/
    └── perceptiontest_<SAMPLER_TYPE>_<LLM_FPS>_<DROPPING_POSITION>_<RETENTION_RATE*100>%_<HYPERPARAM>/
        ├── output.out
        ├── evaluation.log
        ├── error.log
        ├── results.json
        └── failure.json
```

Note: The folder chosen depends on the following logic in the bash script:

If SAMPLER_TYPE is None, the results go under the baseline folder.
If SAMPLER_TYPE is km_closest and HYPERPARAM is not 0.5, the results go under the ablation folder.
If DATASET is charades, the results go under the charades folder.
If DATASET is perceptiontest, the results go under the perceptiontest folder.
Within that folder, the subdirectory is named as:
<DATASET>_<SAMPLER_TYPE>_<LLM_FPS>_<DROPPING_POSITION>_<RETENTION_RATE*100>%_<HYPERPARAM>


