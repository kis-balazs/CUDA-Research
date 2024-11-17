#include <stdio.h>

int main() {
	float f = 97.34;
	int i = (int)f;
	printf("%d\n", i);

	char c = (char)f;
	printf("%c\n", c);
	
	c = (char)i;
	printf("%c\n", c);
	return 0;
}
