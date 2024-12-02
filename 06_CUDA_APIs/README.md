# CUDA API 
> Includes cuBLAS, cuDNN, cuBLASmp

- The term "API" can be confusing at first. All this mean is there is a library which can see internals of a system. There is documentation on the function calls within the API, but its a precompiled binary that doesn't expose source code. The code is highly optimized but no access.

## Opaque Struct Types (CUDA API):
- No access to the internals of the type, just external like names, function args, etc. `.so` (shared object) file referenced as an opaque binary to just run the compiled functions at high throughput. Searching *cuFFT, cuDNN*, or any other CUDA extension, it will be clear that it comes as an API, the inability to see through to the assembly/C/C++ source code refers to usage of the word "opaque". The struct types are just a general type in C that allows NVIDIA to build the ecosystem properly. cublasLtHandle_t is an example of an opaque type containing the context for a cublas Lt operation.

To just figure out how to get the fastest possible inference to work on available cluster, there is a big need to understand the details under the hood. To navigate the CUDA API, worth throwing a look at the following tricks:
1. [perplexity.ai](http://perplexity.ai) (most up to date information and will fetch data in real time - particularly good for niche tasks)
2. google search (< perplexity but exploration process often helps)
3. chatGPT for general knowledge that's less likely to be heplful past its training cutoff
4. keyword search in nvidia docs


## Error Checking (API Specific)

- cuBLAS for example

```cpp
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

- cuDNN example

```cpp
#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudnnGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

- The need for error checking goes as follows: configured CUDA API has a context, then operation is called, then check the status of the operation by passing the API call into the "call" field in the macro. If it returns successful, the code will continue running as expected. If it fails, will return a descriptive error message instead of just a segmentation fault or silently incorrect result. This is massively helpful compared to low-level operations, where verbose debugging is a massive problem.
- There are obviously more error checking macros for other CUDA APIs, but these are the most common examples.
- Must read [Proper CUDA Error Checking](https://leimao.github.io/blog/Proper-CUDA-Error-Checking/)


## Matrix Multiplication
- cuDNN implicitly supports matmul through specific convolution and deep learning operations but isn't presented as one of the main features of cuDNN (\***mindblown**\*).
- (imo crazily, but who am I to say) It is best overall to use the deep learning linear algebra operations in cuBLAS for matrix multiplication since it has wider coverage and is tuned for high throughput matmul.
> Side notes (present to show that its not that hard to transfer knowledge of, say, cuDNN to cuFFT with the way call operations are configured)

## Resources:
- [CUDA Library Samples](https://github.com/NVIDIA/CUDALibrarySamples)

