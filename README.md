# CUDA-Research
Research &amp; Code for CUDA components developed/experimented by me.

## Resources Outline
- [cuda-course](https://github.com/Infatoshi/cuda-course)
- [cuda-mnist](https://github.com/Infatoshi/cuda-course)
- [cuda-opencv-examples](https://github.com/evlasblom/cuda-opencv-examples/tree/master)
- [Getting Started with Accelerated Computing in CUDA C/C++ -- NVIDIA DLI Course](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-AC-04+V1)
- [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis/tree/main) - this seems like a very-very good resource for understanding PyTorch GPU traces, for end-to-end ML model evaluation
  - [Open Torch Perf Traces](https://reimbar.org/dev/torch-profile-trace/)
- [GPU Gems Book](https://developer.nvidia.com/gpugems/gpugems3/contributors) - probably one of the best example-led books out there about the overall topic of GPU parallelism & programming model

## Environment
- [CUDA Installation Guide Ubuntu](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#ubuntu)
- [CUDNN Installation - Latest](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/index.html)

## Related
- [[repo] kis-balazs/CUDA-Containers-Infrastructure-Repository](https://github.com/kis-balazs/cuda-containers-infra)
- [relevant(+adjacent) CUDADocs.md](99_Docs/CUDADocs.md)
- [[PDF] Multi-GPU Programming @ Supercomputing 2011](https://www.nvidia.com/docs/IO/116711/sc11-multi-gpu.pdf)

## Particularly Handy Gists
- [kis-balazs/macro__check_cuda_return.cu](https://gist.github.com/kis-balazs/03f8023320639632db46523aa6e2bc69)
  - Macro to check CUDA Error from Function Returning exit code
- [kis-balazs/verif_last_cuda_error.cu](https://gist.github.com/kis-balazs/3a2590d4bf90f33b0f8776d94da25a92)
  - Verify last CUDA Error code, mainly from kernels, to use for straight-forward debugging

---
## Important Notice - GPU-specific nvcc params

When running `nvcc` commands, the `-arch` command is not always synced to the correct physical GPU. This can be fixed by specifying this.

Steps:
- find compute version: find the GPU compute version [here](https://developer.nvidia.com/cuda-gpus)
- specify when compiling: `nvcc -o exec code.cu -arch=compute_XX`
- optionally, `code` can be specified as well: `-code=sm_XX,compute_XX`

