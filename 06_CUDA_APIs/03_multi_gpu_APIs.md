**cuBLASmp** vs **NCCL** vs **MIG** (multi instance GPU)

## cuBLASmp
- **NVIDIA cuBLASmp** is a high performance, multi-process, GPU accelerated library for distributed basic dense linear algebra. For multi-gpu, single node level tensor ops. Useful if a model can't fit on a single GPU instance.

## NCCL
- NVIDIA Collective Communications Library ⇒ for distributed cluster computing 
- NCCL used for distributing information, collecting it, and acting as a general cluster level communicator. cublasMP is doing the grunt work of doing matmuls across 8xH100s and NCCL is going to run this in batches. Remember “collective communications” ⇒ all-reduce, broadcast, gather, and scatter across multiple GPUs or nodes
- In PyTorch, *Distributed Data Parallel* ⇒ https://pytorch.org/tutorials/intermediate/ddp_tutorial.html, but if the task is as example GPT-5’s training run, want to squeeze every bit of performance out at the datacenter level.
- [CUDA MODE: NCCL Lecture](https://www.youtube.com/watch?v=T22e3fgit-A&ab_channel=CUDAMODE)
- [Extended GPU Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-gpu-memory)
- Model Parallelism (weights) VS Data parallelism (batches)
- Setup [here](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)
- [Operations breakdown](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html)

## MIG (Multi-Instance GPU)
- MIG ⇒ taking a big GPU and literally slicing it into smaller, independent GPUs
- Datacenter usecases where can get more value splitting one node into a bunch of others (customers might not be maxing out the compute utilization)
