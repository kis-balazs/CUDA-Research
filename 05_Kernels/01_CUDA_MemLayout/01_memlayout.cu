#include <stdio.h>

__global__ void whoami(void) {
	int block_id =  // 3dim position of block inside of the grid
		blockIdx.x +  // horizontal
		blockIdx.y * gridDim.x + // vertical
		blockIdx.z * gridDim.x * gridDim.y; // depth

	int block_offset =  // offset in grid memory
		block_id *
		blockDim.x * blockDim.y * blockDim.z;  // total threads per block

	int thread_id =  // 3dim position of thread inside of the block, SINCE NO MORE SPATIAL OFFSET
		threadIdx.x + 
		threadIdx.y * blockDim.x +
		threadIdx.z * blockDim.x * blockDim.y;

	int id = block_offset + thread_id;

	printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
			id,
			blockIdx.x, blockIdx.y, blockIdx.z, block_id,
			threadIdx.x, threadIdx.y, threadIdx.z, thread_id);
}

int main() {
	const int b_x = 2, b_y = 3, b_z = 4;
	const int t_x = 3, t_y = 3, t_z = 3; // this will be done in 1 warp of 32 threads per block

	int blocks_per_grid = b_x * b_y * b_z;
	int threads_per_block = t_x * t_y * t_z;

	printf("blocks/grid = %d\n", blocks_per_grid);
	printf("threads/block = %d\n", threads_per_block);
	printf("total threads = %d\n", blocks_per_grid * threads_per_block);

	dim3 blocksPerGrid(b_x, b_y, b_z); // 3d cube of shape 24
	dim3 threadsPerBlock(t_x, t_y, t_z); // 3d cube of shape 27

	whoami<<<blocksPerGrid, threadsPerBlock>>>();
	cudaDeviceSynchronize();

	// see how cool because of the warp, the threads are linear, but the blocks are completely randomly ordered (because of parallelity)
}
