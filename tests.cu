// all of this is heavily dependend on specific values
// so most of the tests will probably fail when the files
// were generated with different ones (e.g. ./uid_pepper)

#ifdef _DEBUG

#include "tests.cuh"

void test() {
    test_pepper1();
    char* str = test_gen_strings();
    test_hashcode(str);
}

void test_pepper1() {
    char* str = (char*)malloc(3 * MAX_STRING_LEN);
    size_t size = sizeof(char) * MAX_STRING_LEN;
    char* str0 = str;
    char* str1 = str + MAX_STRING_LEN;
    char* str2 = str1 + MAX_STRING_LEN;
    memcpy(str0, "___28911433160864563000------", size);
    memcpy(str1, "___29200547492473210000------", size);
    memcpy(str2, "___29489661824081854000------", size);
    gen_pepper1(&str0[PEPPER1_START], 0);
    gen_pepper1(&str1[PEPPER1_START], 35);
    gen_pepper1(&str2[PEPPER1_START], 58786559); // "zzzzz"

    char* d_str;
    size *= 3;
    cudaMalloc(&d_str, size);
    cudaMemcpy(d_str, str, size, cudaMemcpyHostToDevice);

    char* res[3];
    res[0] = "___28911433160864563000_10010";
    res[1] = "___29200547492473210000_1001z";
    res[2] = "___2948966182408185400010000?";

    TestPepper1 <<<1, 3 >>> (d_str);
    cudaDeviceSynchronize();
    cudaMemcpy(str, d_str, size, cudaMemcpyDeviceToHost);
    cudaFree(d_str);

    printf("String should be %s and is actually: %.29s\n", res[0], str0);
    printf("String should be %s and is actually: %.29s\n", res[1], str1);
    printf("String should be %s and is actually: %.29s\n", res[2], str2);
}

__global__ void TestPepper1(char* d_str) {
    char* str = &d_str[threadIdx.x * MAX_STRING_LEN];
    incr_pepper1_36(&str[PEPPER1_START]);
}

char* test_gen_strings() {
    char* str = (char*)malloc(MAX_STRING_LEN * PEPPER2_POSSIBILITIES);
    gen_strings(str);

    printf("First entry is : %.29s\n", str);
    printf("20. entry is   : %.29s\n", &str[MAX_STRING_LEN * 20]);
    printf("Last entry is  : %.29s\n", &str[MAX_STRING_LEN * (PEPPER2_POSSIBILITIES - 1)]);

    return str;
}

void test_hashcode(char* str) {
    size_t size = sizeof(char) * PEPPER2_POSSIBILITIES * MAX_STRING_LEN;
    char* d_str;
    int32_t* d_res;
    cudaMalloc(&d_str, size);
    cudaMalloc(&d_res, sizeof(int32_t) * 3);
    cudaMemcpy(d_str, str, size, cudaMemcpyHostToDevice);

    int32_t* res = (int32_t*)malloc(3 * MAX_STRING_LEN);

    TestHashcode<<<1, 3>>>(d_str, d_res);
    cudaDeviceSynchronize();
    cudaMemcpy(res, d_res, 3 * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(str, d_str, sizeof(char) * MAX_STRING_LEN, cudaMemcpyDeviceToHost);
    cudaMemcpy(&str[MAX_STRING_LEN], &d_str[MAX_STRING_LEN * 50], sizeof(char) * MAX_STRING_LEN, cudaMemcpyDeviceToHost);
    cudaMemcpy(&str[MAX_STRING_LEN * 2], &d_str[MAX_STRING_LEN * (PEPPER2_POSSIBILITIES - 1)], sizeof(char) * MAX_STRING_LEN, cudaMemcpyDeviceToHost);
    cudaFree(d_str);
    cudaFree(d_res);

    printf("Hash should be -1815501465 and is: %d\n", res[0]);
    printf("Hash should be -1704195523 and is: %d\n", res[1]);
    printf("Hash should be -276972291 and is: %d\n", res[2]);
    printf("The following lines should start with 1, 1, and 0:\n");
    printf("%.29s\n", str);
    printf("%.29s\n", &str[MAX_STRING_LEN]);
    printf("%.29s\n", &str[MAX_STRING_LEN * 2]);
}

__global__ void TestHashcode(char* d_str, int32_t* d_res) {
    int i;
    int32_t hash;

    if (threadIdx.x == 0) {
        d_str += MAX_STRING_LEN * 0;
        // 010000
    }

    else if (threadIdx.x == 1) {
        d_str += MAX_STRING_LEN * 50;
        incr_pepper1_36(&d_str[PEPPER1_START]); // 010010
    }

    else {
        d_str += MAX_STRING_LEN * (PEPPER2_POSSIBILITIES - 1);
        for (int i = 0; i < 50; i++)
            incr_pepper1_36(&d_str[PEPPER1_START]);
        d_str[PEPPER1_START] = '1'; // 1101e0
    }

    int j = 0;
    while (d_str[j] == '_') j++;

    hashcode(d_str, &hash, &j);

    size_t size = sizeof(int32_t);
    int32_t target;
    if (threadIdx.x == 0) {
        target = -1815501391; // not actual hash but 1000z as pepper1; should work too
        memcpy(d_res, &hash, size);
    }

    else if (threadIdx.x == 1) {
        target = -1704195523; // correct hash
        memcpy(&d_res[1], &hash, size);
    }

    else {
        target = -27697229; // actual hash is -276972291 -> should not work
        memcpy(&d_res[2], &hash, size);
    }

    d_str[0] = '0' + compare_hashcode_36(&target, &hash);
}

/*
__global__ void Test(bool res[]) {
    test_pepper1();

    switch (threadIdx.x) {
    case 0:
        res[threadIdx.x] = test_pepper1();
        break;
    }
}

__device__ bool compare_pepper1(char* out, char* r) {
    for (int i = 0; i < PEPPER1_MAX_LEN; i++) {
        if (out[i] != r[i])
            return false;
    }

    return true;
}

__device__ bool test_pepper1() {
    char out[PEPPER1_MAX_LEN + 1];
    out[PEPPER1_MAX_LEN] = '\0';

    char r1[PEPPER1_MAX_LEN + 1] = "_10001";
    gen_pepper1(out, 0);
    incr_pepper1(out);
    if (!compare_pepper1(out, r1))
        return false;

    char r2[PEPPER1_MAX_LEN + 1] = "_10011";
    gen_pepper1(out, 35); // 01000z
    incr_pepper1(out);
    incr_pepper1(out);
    if (!compare_pepper1(out, r2))
        return false;

    char r3[PEPPER1_MAX_LEN + 1] = "_10100";
    gen_pepper1(out, (36 * 36) - 1); // 0100zz
    incr_pepper1(out);
    if (!compare_pepper1(out, r3))
        return false;

    char r4[PEPPER1_MAX_LEN + 1] = "zzzzzz";
    gen_pepper1(out, 2176782335); // zzzzzz
    incr_pepper1(out);
    if (!compare_pepper1(out, r4))
        return false;

    return true;
}

__device__ bool test_hashcode() {
    // TODO
}
*/

#endif