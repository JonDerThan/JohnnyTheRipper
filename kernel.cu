
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "src/start.cuh"

#ifdef _DEBUG
#include "tests.cuh"
#endif

int main(int argc, char* argv[])
{
#ifdef _DEBUG
    test();
#endif

    int32_t target;
    sscanf(argv[1], "%d", &target);

#ifdef _DEBUG
    printf("Working on hash: %d\n", target);
#endif

    start(target);

    return 0;
}
