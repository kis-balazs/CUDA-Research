// size_t = size type for memory allocation
// size_t is an unsigned integer data type used to represent the size of objects in bytes. 
// It is guaranteed to be big enough to contain the size of the biggest object the host system can handle.

// https://cplusplus.com/reference/cstdio/printf/

#include <stdio.h>
#include <stdlib.h>

int main() {
	int arr[] = {0, 1, 2, 3, 4};

	size_t size = sizeof(arr) / sizeof(arr[0]);
	printf("Arr size: %zu\n", size);
	
	printf("size_t size= %zu\n", sizeof(size_t));
	printf("int size= %zu\n", sizeof(int));

	// z -> size_t; u -> unsigned int
	return 0;
}
