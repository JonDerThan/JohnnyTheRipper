#include "gen_strings.cuh"

/*
__device__ void gen_string(char* out, const double* uid, const char* pepper1, const int* pepper2) {
	double prod = *uid * *pepper2;
	uint_fast16_t pepper2_start = PEPPER1_START;

	// copy pepper2 to end of out
	for (int i = 0; i < PEPPER1_MAX_LEN; i++) {
		out[pepper2_start + i] = pepper1[i];
	}

	// convert prod to string
}
*/

void gen_strings(char* out) {
	get_uid_pepper2(out);
	for (int i = 0; i < PEPPER2_POSSIBILITIES; i++) {
		gen_pepper1(&out[PEPPER1_START], 0);
		out += MAX_STRING_LEN;
	}
}