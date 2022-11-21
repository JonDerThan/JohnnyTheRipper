#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BASE 36 // to avoid mistyping

#define PEPPER1_MIN_NUM 1679616 // -> 10000 in base36
#define PEPPER1_MAX_NUM 2176782336UL // 36^6
#define PEPPER1_POSSIBILITIES PEPPER1_MAX_NUM - PEPPER1_MIN_NUM
#define PEPPER1_MAX_LEN 6

#define PEPPER2_POSSIBILITIES 900

#define MAX_STRING_LEN 29
#define PEPPER1_START MAX_STRING_LEN - PEPPER1_MAX_LEN

void gen_pepper1(char* out, unsigned int i);
// TODO: check if needed, delete otherwise
// __device__ void incr_pepper1(char* out);
__device__ void incr_pepper1_36(char* out);