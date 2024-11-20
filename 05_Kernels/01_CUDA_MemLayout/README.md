# CUDA Basics & Memory Layout

## Cookbook
Host == CPU: Uses RAM sticks on the motherboard

Device == GPU: Uses on Chip VRAM (video memory for desktop PCs)

CUDA program surface level runtime:

1. copy input from host to device
2. load GPU program and execute using the transferred on-device data
3. copy results from device back to host to be used

## Device VS Host naming scheme
`h_A` refers to host (CPU) for variable name "A"

`d_A` refers to device (GPU) for variable name "A"

`__global__` is visible globally, meaning the CPU or *host* can call these global functions. These don't typically return anything but just do fast operations to a variable passed. For example, to multiply matrix A and B together: pass in a matrix of the needed size as C and change the values in C to the outputs of A * B matmul. CUDA kernels in a nutshell. 

`__device__` is a function representing a small job that only the GPU can call. As example the following problem: having a raw attention score matrix living on the `__global__` gpu cuda kernel and it needs to apply a scalar mask. Instead of also doing this in the cuda kernel, just have a `__device__` function defined in another .cu file that does this SIMD (SingleInstruction/MultipleData) scalar masking on any matrix. This is the CUDA equivalent of calling a function in a library instead of writing the function in a `main.py` file.

`__host__` is only going to run on CPU. same as running a regular c/c++ script on CPU without cuda.

## Memory Management

- `cudaMalloc` memory allocation on VRAM only (also called global memory)

```
    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, N*N*sizeof(float));
    cudaMalloc(&d_b, N*N*sizeof(float));
    cudaMalloc(&d_c, N*N*sizeof(float));
```

- `cudaMemcpy` can copy from device to host, host to device, or device to device (edge cases)
    - host to device: CPU to GPU
    - device to host: GPU to CPU
    - device to device: GPU location to different GPU location
    - **`cudaMemcpyHostToDevice`**, **`cudaMemcpyDeviceToHost`**, or **`cudaMemcpyDeviceToDevice`**
- `cudaFree` will free memory on the device

# `nvcc` compiler
- Host code
    - modifed to run kernels
    - compiled to x86 binary

- Device code
    - compiled to PTX (parallel thread execution)

    - stable across multiple GPU generations

- JIT (just-in-time)

    - PTX into native GPU instructions

    - allows for forward compatibility

## CUDA Hierarchy?
> grid >> block >> thread

1. Kernel executes in a thread
2. Threads grouped into Thread Blocks (aka Blocks)
3. Blocks grouped into a Grid
4. Kernel executed as a Grid of Blocks of Threads

### 4 technical terms:
All structured in 3D format, having .x, .y, .z attributes

- `gridDim`: number of blocks in the grid
- `blockIdx`: index of the block in the grid
- `blockDim`: number of threads in a block
- `threadIdx`: index of the thread in the block

## Threads
- each thread has local memory (registers) and is private to the thread
- if want to add `a = [1, 2, 3, ... N]` and `b = [2, 4, 6, ... N]` each thread would do a single add ⇒ `a[0] + b[0]` (thread 1); `a[1] + b[1]` (thread 2); etc...

## Warps
- [Warp and Weft Wiki](https://en.wikipedia.org/wiki/Warp_and_weft)
- The warp is the set of yarns stretched in place on a loom before the weft is introduced during the weaving process. It is regarded as the *longitudinal* set in a finished fabric with two or more sets of elements.
- Each warp is inside of a block and parallelizes 32 threads (fixed)
- Instructions are issued to warps that then tell the threads what to do (not directly sent to threads)
- There is no way of getting around using warps
- Warp scheduler makes the warps run
- 4 warp schedulers per SM (Streaming Multiprocessor)

$ceil(\frac{T}{W_{size}})$, where $T$ is #threads/block, $W_{size}$ is warp size =32

## Blocks
- each block has shared memory (visible to all threads in thread block)
- execute the same code on different data, shared memory space, more efficient memory reads and writes since coordination is better

## Grids
- during kernel execution, the threads within the blocks within the grid can access global memory (VRAM)
- analogy: grids handle batch processing, where each block in the grid is a batch element

> Q: why not just use only threads instead of blocks and threads? add to this given knowledge of how warps group and execute a batch of 32 threads in lockstep

> A: Logically, this shared memory is partitioned among the blocks. This means that a thread can communicate with the other threads in its block via the shared memory chunk. 

- CUDA parallelism is scalable because there aren't sequential block run-time dependencies. This means each of the splitted jobs are solving a subset of the problem independent of the others. Combine at end, divide and conquer.

> [How do threads map onto CUDA cores?](https://stackoverflow.com/questions/10460742/how-do-cuda-blocks-warps-threads-map-onto-cuda-cores)
