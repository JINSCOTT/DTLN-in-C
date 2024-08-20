
#include "utility.h"
void swap(void* a, void* b)
{
	 void* c = a;
	 a = b;
	 b = c;
}

void safe_free(void* p) {
	if (p != 0) {
		free(p);
		p = 0;
	}
}