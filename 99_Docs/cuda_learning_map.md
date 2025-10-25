# Learning Map for CUDA & GPU Parallelism

## 1) Intuition (Week 1-2)
> Mental model of the hardware and the why first.

- UC Berkeley CS 61C, Lecture 17 -- [YT Link](https://www.youtube.com/watch?v=xdcW52tEPfE)

- Coursera Parallel Computing Course (Recommended First 3 modules) -- [Course Link](https://www.coursera.org/learn/scala-parallel-programming)

- Stanford CS231n Lecture 15 - Hardware/Software interface -- [YT Link](https://www.youtube.com/watch?v=eZdOkDtYMoo)

## 2) CUDA Basics (Week 3-4)
> Fundamental paradigm behind CUDA, code basics, etc.

- NVIDIA's official CUDA C++ Programming Guide (Recommended Chapters 1-5) -- [Docs Link](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
  - Learn threads, blocks, grids and kernel structure

- cuda-samples repo - [GitHub Link](https://github.com/NVIDIA/cuda-samples)

## 3) Memory Management (Week 5-8)
> The chokehold or the strength of CUDA, everything depends on it

- Mark Harris's GTC Talk on Coalesced Memory Access -- [GTC Sesh Link](https://nvidia.com/en-us/on-demand/session/gtc24-s62550)
  - Single most important CUDA performance concept.
  - How threads must access global memory in aligned groups.

- ! GPU Gems 3, Chapter 39 - "Parallel Prefix Sum with CUDA" -- [Book Chapter Link](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
  - Shared memory to avoid bank conflicts, a fundamental optimization.

- CUDA C++ Best Practices Guide - "Memory Optimizations" Chapter -- [Docs Link](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)

## 4) Kernels (Week 9-12)
> As hands-on as possible 

First examples worth considering:

- Activation functions

- Write a basic GEMM against cuBLAS

- Port one PyTorch operation to CUDA

Cool inspirations:

- tiny-cuda-nn by NVIDIA -- [GitLab Link](https://github.com/NVlabs/tiny-cuda-nn)

- FlashAttention -- [GitLab Link](https://github.com/Dao-AILab/flash-attention)
  - Insane memory-aware kernel design

- Triton Language Examples -- [GitLab Link](https://github.com/triton-lang/triton)
