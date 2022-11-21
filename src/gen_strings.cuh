#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdint>

#include "pepper.cuh"
#include "get_uid_pepper2.cuh"

// TODO: check if needed, delete otherwise
// __device__ void gen_string(char* out, const double* uid, const char* pepper1, const int* pepper2);

void gen_strings(char* out);