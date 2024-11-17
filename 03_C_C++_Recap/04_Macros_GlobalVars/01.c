#include <stdio.h>

// >> conditional macros
// #if
// #ifdef
// #ifndef
// #elif
// #else
// #endif

#define PI 3.14159
#define area(r) (PI * r * r)

#ifndef radius
#define radius 11
#endif

#if radius > 15
#define radius 15
#elif radius < 5
#define radius 5
#else
#define radius 11
#endif

int main() {
	printf("Area of circle with radius %3d is %.3f\n", radius, area(radius));

	return 0;
}
