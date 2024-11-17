# 01
Direct or reference addressing for structs in C/C++. Relevant, mainly in parameter passing (by reference/value), and ofc memory handling.


# 02
## Purpose of `#pragma once`?
- include `#pragma once` so that file is only included once. otherwise you get `error: redefinition of 'foo'` in the code example below
- wiki [Pragmaonce](https://en.wikipedia.org/wiki/Pragma_once#:~:text=In%20the%20C%20and%20C,once%20in%20a%20single%20compilation)
	- Example with "inheritance" approach, where f1 is included in f2, but both f1 and f2 are directly included in f3; solution: pragma once f1
