#include <stdio.h>

int main() {
	int num = 10;
	float fnum = 1.23;
	void* vptr;  // when don't know the data type, this is quite functional; malloc returns this, that's why cast is needed

	vptr = &num;
	printf("%d\n", *(int*)vptr);
	
	vptr = &fnum;
	printf("%.2f\n", *(float*)vptr);
	return 0;
}
