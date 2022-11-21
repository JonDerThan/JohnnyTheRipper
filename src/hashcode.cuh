#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdint>

#include "pepper.cuh"

__device__ void hashcode(const char* str, int32_t* hash, const int* start);
__device__ bool compare_hashcode_36(const int32_t* target, const int32_t* calc);