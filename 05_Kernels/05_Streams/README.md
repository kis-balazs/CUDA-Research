# CUDA Streams

## Intuition
Streams are like "river streams" where the direction of operations flows only forward in time (like a timeline). For example, copy some data over (time step 1), then do some computation (time step 2), then copy some data back (time step 3). This is the basic idea behind streams. 

One can have multiple streams at once in CUDA, and each stream can have its own timeline. This allows to overlap operations and make better use of the GPU.

When training e.g., massive language model, it would be suboptimal to spend a ton of time loading all the tokens in and out of the GPU. Streams allow to move data around while also doing computation at all times. Streams introduce a software abstraction called "prefetching", which is a way to move data around before it is needed. This is a way to hide the latency of moving data around. 

## Kernel Launch Parameters
- default **stream** = **stream** 0 = null **stream**
```cpp
// This kernel launch uses the null stream (0)
myKernel<<<gridSize, blockSize>>>(args);

// This is equivalent to
myKernel<<<gridSize, blockSize, 0, 0>>>(args);
```

- The execution configuration (of a global function call) is specified by inserting an expression of the form `<<<gridDim, blockDim, Ns, S>>>`, where:

  - Dg (dim3) specifies the dimension and size of the grid.
  - Db (dim3) specifies the dimension and size of each block.
  - Ns (size_t) specifies the number of bytes in shared memory that is dynamically allocated per block for this call in addition to the statically allocated memory. (typically omitted)
  - S (cudaStream_t) specifies the associated stream, is an optional parameter which defaults to 0.

- stream 1 and stream 2 are created with different priorities. this means they are executed in a certain order at runtime. this essentially gives more control over the concurrent execution of our kernels.

```cpp
    // Create streams with different priorities
    int leastPriority, greatestPriority;
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestPriority));
```

## Docs
- https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf

## Pinned Memory
- "Gonna need this for later, so don't play with it" is a good way to think about it.
- Pinned memory is memory that is locked in place and cannot be moved around by the OS. This is useful for when need to move data to the GPU and do some computation on it. If the OS moves the data around, the GPU will be looking for the data in the wrong place and will result in a segfault.

```cpp
// Allocate pinned memory
float* h_data;
cudaMallocHost((void**)&h_data, size);
```

## Events
- Measuring kernel execution time: Events are placed before and after kernel launches to measure execution time accurately.

- Synchronizing between streams: Events can be used to create dependencies between different streams, ensuring one operation starts only after another has completed.

- Overlapping computation and data transfer: Events can mark the completion of a data transfer, signaling that computation can begin on that data.


```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

## Callbacks
-  By using callbacks, one can set up a pipeline where the completion of one operation on the GPU triggers the start of another operation on the CPU, which might then queue more work for the GPU. (as seen in the nvidia concurrency docs above)

```cpp
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("GPU operation completed\n");
    // Trigger next batch of work
}

kernel<<<grid, block, 0, stream>>>(args);
cudaStreamAddCallback(stream, MyCallback, nullptr, 0);
```
