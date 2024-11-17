#include <stdio.h>

int main() {
	int x = 10;
	int* ptr = &x; // & - address of operator; get mem address of
	printf("%p\n", ptr);
	printf("%d\n", *ptr);  // * - dereference operator; get value of
	return 0;
}
