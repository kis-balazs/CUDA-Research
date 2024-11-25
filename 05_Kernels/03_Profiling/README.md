# CUDA Kernel Profiling

## Profiling
1. 
```bash
nvcc -o 00 00_nvtx_matmul.cu -lnvToolsExt
nsys profile --stats=true ./00
```

> for these two: `ncu` on linux and drag and drop the .nsys-rep file into the left sidebar.
> the .sqlite file can be plugged directly into sqlite DBs for more customized analysis
2. 
```bash
nvcc -o 01 01_naive_matmul.cu`
nsys profile --stats=true ./01
```

3. 
```bash
nvcc -o 02 02_tiled_matmul.cu
nsys profile --stats=true ./02
```

## CLI tools
- some cli tools to visualize GPU resource usage & utilization
- `nvitop`
- `nvidia-smi` or `watch -n 0.1 nvidia-smi`


# Nsight systems & compute
- Nsight systems & compute --> cli tool calls above
- Unless there is a specific profiling goal, the suggested profiling strategy is starting with Nsight Systems to determine system bottlenecks and identifying kernels that affect performance the most. On a second step, using Nsight Compute to profile the identified kernels and find ways to optimize them [SoF discussion](https://stackoverflow.com/questions/76291956/nsys-cli-profiling-guidance)
- If available `.nsys-rep` file, run `nsys stats file.nsys-rep` for a more quantitative profile. For `.sqlite` run `nsys analyze file.sqlite` to give a more qualitative profile.
  - to see a detailed GUI of this, run `nsight-sys` --> file --> open --> rep file

- `nsys` nsight systems is higher level; `ncu` nsight compute is lower level

- Python scirpt profiling files: `nsys profile --stats=true -o mlp python mlp.py`
- To profile w/ nsight systems GUI, find optimize-bound kernels (ex: `ampere_sgemm`), open in event view, zoom to selected on timeline, analyze kernel w/ ncu by right clicking on timeline
- ncu may deny permissions --> [SoF discussion](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
- *Memory Leaks*: `compute-sanitizer ./exec`
- kernel performance UI --> ncu-ui (might have to `sudo apt install libxcb-cursor0`)

## Kernel Profiling
- [Nsight Compute Kernel Profiling](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- `ncu --kernel-name matrixMulKernelOptimized --launch-skip 0 --launch-count 1 --section Occupancy "./nvtx_matmul"`
- turns out nvidia profiling tools won't give everything needed to optimize deep learning kernels: [Here](https://stackoverflow.com/questions/2204527/how-do-you-profile-optimize-cuda-kernels)

## NVTX `nvtx` profiling
```bash
# Compile the code
nvcc -o matmul matmul.cu -lnvToolsExt

# Run the program with Nsight Systems
nsys profile --stats=true ./matmul
```
- `nsys stats report.qdrep` to see the stats


## CUPTI
- allows to build own profiler tools
- The *CUDA Profiling Tools Interface* (CUPTI) enables the creation of profiling and tracing tools that target CUDA applications. CUPTI provides the following APIs: the *Activity API*, the *Callback API*, the *Event API*, the *Metric API*, the *Profiling API*, the *PC Sampling API*, the *SASS Metric API* and the *Checkpoint API*. Using these APIs, you can develop profiling tools that give insight into the CPU and GPU behavior of CUDA applications. CUPTI is delivered as a dynamic library on all platforms supported by CUDA.
- https://docs.nvidia.com/cupti/overview/overview.html
