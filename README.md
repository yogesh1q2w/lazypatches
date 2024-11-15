# lazypatches
A spatio-temporal video subsampling strategy for efficient video-language modelling. 



# Build Images

For CPU usage: 

```docker build -t qwen-pytorch-fast-cpu -f docker/consistency.dockerfile .```


For GPU usage:

```docker build -t qwen2-pytorch-gpu -f docker/transformers-pytorch-gpu/Dockerfile .```

Check [./docker/README.md](./docker/README.md) for more details


# Run in the container

For CPU usage: 

```docker run -it -v ${HOME}/lazypatches/:/home  qwen-pytorch-fast-cpu bash```


For GPU usage:

```docker_run_nvidia -v ${HOME}/lazypatches/:/home  qwen2-pytorch-gpu bash```

# Get into a running container

```docker exec -it <container-id> bash```




