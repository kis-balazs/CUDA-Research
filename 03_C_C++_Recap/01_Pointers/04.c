#include <stdio.h>
#include <stdlib.h>

int main() {
	int* ptr = NULL;
	printf("%p\n", (void*)ptr);

	// check before use
	if (ptr == NULL) printf("cannot dereference!\n");

	ptr = malloc(sizeof(int));
	if (ptr == NULL) {
		printf("malloc failed!\n");
		return 1;
	}
	// checks are quite useful for graceful handling of uninitialized/failed allocations!

	*ptr = 10;
	printf("%d\n", *ptr);

	free(ptr);

	ptr = NULL;
	printf("after free, ptr value: %p\n", (void*)ptr);

    	// Demonstrate safety of NULL check after free
    	if (ptr == NULL) {
        	printf("ptr is NULL, safely avoided use after free\n");
    	}
	return 0;
}
