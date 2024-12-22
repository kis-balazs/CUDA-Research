# CUDA-Research
Research &amp; Code for CUDA components developed/experimented by me.

## Resources Outline
- [cuda-course](https://github.com/Infatoshi/cuda-course)
- [cuda-mnist](https://github.com/Infatoshi/cuda-course)
- [cuda-opencv-examples](https://github.com/evlasblom/cuda-opencv-examples/tree/master)

## Related
- [kis-balazs/CUDA-Containers-Infrastructure-Repository](https://github.com/kis-balazs/cuda-containers-infra)

 Important Notice

When running `nvcc` commands, the `-arch` command is not always synced to the correct physical GPU. This can be fixed by specifying this.

Steps:
- find compute version: find the GPU compute version [here](https://developer.nvidia.com/cuda-gpus)
- specify when compiling: `nvcc -o exec code.cu -arch=compute_XX`
- optionally, `code` can be specified as well: `-code=sm_XX,compute_XX`

