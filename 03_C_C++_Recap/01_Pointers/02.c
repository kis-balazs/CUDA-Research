#include <stdio.h>

int main() {
	int val = 10;
	int* ptr = &val;
	int** ptr1 = &ptr;
	int*** ptr2 = &ptr1;

	printf("%d\n", ***ptr2);
	return 0;
}
