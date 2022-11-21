#include "hashcode.cuh"

__device__ void hashcode(const char* str, int32_t* hash, const int* start) {
	*hash = 0;

	int i;
	for (i = *start; i < PEPPER1_START; i++) {
		*hash = ((*hash << 5) - *hash) + str[i];
	}

	if (str[i] == '0') i++;

	for (; i < MAX_STRING_LEN; i++) {
		*hash = ((*hash << 5) - *hash) + str[i];
	}
}

__device__ bool compare_hashcode_36(const int32_t* target, const int32_t* calc) {
	int32_t res = *target - *calc;
	// characters have a higher charcode than digits
	return (res < 10 && res >= 0) || (res <= 'z' - '0' && res >= 'a' - '0');
}