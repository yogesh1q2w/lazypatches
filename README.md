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
cd transformers
pip install -e .
cd ..
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

# With Apptainer

## Build images

For CPU usage: 

```docker build -t qwen-pytorch-fast-cpu -f docker/consistency.dockerfile .```


For GPU usage:

```docker build -t qwen2-pytorch-gpu -f docker/transformers-pytorch-gpu/Dockerfile .```

Check [./docker/README.md](./docker/README.md) for more details


## Run in the container

For CPU usage: 

```docker run -it -v ${HOME}/lazypatches/:/home  qwen-pytorch-fast-cpu bash```


For GPU usage:

```docker_run_nvidia -v ${HOME}/lazypatches/:/home  qwen2-pytorch-gpu bash```

## Get into a running container

```docker exec -it <container-id> bash```


## Converting to Apptainer

```apptainer build qwen2-pytorch-gpu.sif  docker://levoz/lazypatches:qwen2-pytorch-gpu```

## Run in the Apptainer

```apptainer exec qwen2-pytorch-gpu.sif bash```
