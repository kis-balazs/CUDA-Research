#include <stdio.h>

int main() {
	int arr[] = {0, 1, 2, 3, 4};
	int* ptr = arr;  // arrays are pointers on their own

	printf("%d\n", *ptr);
	

	for (int i = 0; i < 5; i++) {
		printf("%d ", *ptr);
		printf("%p\n", ptr);
		ptr++; // equivalent with arr[i]
	}

	// pointer is incremented by 4 bytes (size of int = 4 bytes * 8 bits/bytes = 32 bits = int32) each time. 
    	// ptrs are 64 bits in size (8 bytes). 2**32 = 4,294,967,296 which is too small given how much memory we typically have.
    	// arrays are layed out in memory in a contiguous manner (one after the other rather than at random locations in the memory grid)
	return 0;
}
