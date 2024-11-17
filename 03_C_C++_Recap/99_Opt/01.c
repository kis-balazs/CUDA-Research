#include <stdio.h>
#include <stdlib.h>

struct Person {
	int age;
	char* name;
};


int main() {
	struct Person person1;
	person1.age = 5;
	person1.name = "Wayne";

	printf("age: %d\tname: %s\n", person1.age, person1.name);

	struct Person* person2 = malloc(sizeof(struct Person));
	// alternatively can alsoe use malloc(sizeof(person1)); -- existing variable declared

	person2->age = 2;
	person2->name = "Bruce";

	printf("age: %d\tname: %s\n", person2->age, person2->name);
  	return 0;
}

