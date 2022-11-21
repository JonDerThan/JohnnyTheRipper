#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gen_strings.cuh"
#include "hashcode.cuh"

void start(int32_t target);
__global__ void Start(char* d_str, int32_t* d_target);
__device__ void calculate(char* str, int32_t* target);