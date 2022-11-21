#include "start.cuh"

void start(int32_t target) {
	// generate strings to work on
	size_t size = sizeof(char) * MAX_STRING_LEN * PEPPER2_POSSIBILITIES;
	char* str = (char*)malloc(size);
	gen_strings(str);

	// copy strings + target to device
	char* d_str;
	int32_t* d_target;
	cudaMalloc(&d_str, size);
	cudaMalloc(&d_target, sizeof(int32_t));
	cudaMemcpy(d_str, str, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_target, &target, sizeof(int32_t), cudaMemcpyHostToDevice);

	// start kernels
#ifdef _DEBUG
	printf("Starting kernels...\n");
#endif // _DEBUG
	Start<<<9, 100>>>(d_str, d_target);
	cudaDeviceSynchronize();
	cudaFree(d_str);
	cudaFree(d_target);
}

__global__ void Start(char* d_str, int32_t* d_target) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;	// blockDim.x is number of threads in block -> e.g. 2 * 100 + 12 = 212 -> 3rd block, 13th thread
	d_str += i * MAX_STRING_LEN;

	calculate(d_str, d_target);
}

__device__ void calculate(char* str, int32_t* target) {
	int32_t hash;
	int start = 0;
	while (str[start] == '_') start++;

	for (int i = 0; i < PEPPER1_POSSIBILITIES; i++) {
		hashcode(str, &hash, &start);
		if (compare_hashcode_36(target, &hash))
			printf("%.29s\n", str);
		incr_pepper1_36(&str[PEPPER1_START]);
	}
}