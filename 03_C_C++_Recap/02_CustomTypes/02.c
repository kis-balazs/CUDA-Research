#include <stdio.h>

typedef struct {
	float x;
	int y;
	float z;
} Point;


int main() {
	Point p = {1.2, 3};
	printf("size of Point: %zu\n", sizeof(Point));
	
	return 0;
}
