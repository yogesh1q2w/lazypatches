# lazypatches
A spatio-temporal video subsampling strategy for efficient video-language modelling. 



# Build Images

For CPU usage: 

```docker build -t qwen-pytorch-fast-cpu -f docker/consistency.dockerfile .```


For GPU usage:

```docker build -t qwen2-pytorch-gpu -f docker/transformers-pytorch-gpu/Dockerfile .```

Check [./docker/README.md](./docker/README.md) for more details
