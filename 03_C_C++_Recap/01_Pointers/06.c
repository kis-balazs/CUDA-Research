#include <stdio.h>

int main() {
	int arr1[] = {1, 2, 3, 4};
	int arr2[] = {2, 3, 4, 5};

	int* ptr1 = arr1; // 8 bytes, each
	int* ptr2 = arr2;

	int* matrix[] = {ptr1, ptr2};
	printf("%ld\n", sizeof(matrix));

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 4; j++)
			printf("%d ", *matrix[i]++);
		printf("\n");
	}
	return 0;
}
