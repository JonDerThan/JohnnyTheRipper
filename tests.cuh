#ifdef _DEBUG

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/pepper.cuh"
#include "src/hashcode.cuh"
#include "src/gen_strings.cuh"

/*
__device__ bool compare_pepper1(char* out, char* r);
__device__ bool test_pepper1();
__device__ bool test_hashcode();

__global__ void Test(bool res[]);
*/

void test_pepper1();
__global__ void TestPepper1(char* d_str);
char* test_gen_strings();
void test_hashcode(char* str);
__global__ void TestHashcode(char* d_str, int32_t* d_res);
void test();

#endif
