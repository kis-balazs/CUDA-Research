#include <iostream>

typedef struct {
        float x;
        int y;
        float z;
} Point;


int main() {
        Point p = {1.2, 3};
	std::cout << "size of Point:" << sizeof(Point) << std::endl;

        return 0;
}
