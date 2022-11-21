#include "pepper.cuh"

// inefficient, only use for generating starting points
// result has leading zeros e.g. "010000"
void gen_pepper1(char* out, unsigned int i)
{
    i += PEPPER1_MIN_NUM;       // offset by 010000 because pepper1 is at least 5 characters long
    int j = PEPPER1_MAX_LEN - 1;

    while (i > 0) {
        out[j] = '0' + i % BASE;
        if (out[j] > '9')
            out[j] += 39;

        i /= BASE;
        j--;
    }

    // set leading characters to 0
    while (j >= 0)
        out[j--] = '0';
}

/*
__device__ void incr_pepper1(char* out) {
    int i = PEPPER1_MAX_LEN;

    // while out[i] is 'z' set to '0' and continue, else increase by one
    while (i-- > 0) {
        if (out[i] == 'z') {
            out[i] = '0';
            continue;
        }

        if (out[i] == '9') out[i] += 40;
        else out[i]++;
        break;
    }
}
*/


/**
* Increases pepper1 by 36, because the hashcode simply adds the rightmost character in the last step.
* Therefore, you only have to generate the hashcode for 010000, 010010, 010020, etc.
* With the target hash x and the calculated hash y: 
* if 0 <= x-y < 36 -> success
*/
__device__ void incr_pepper1_36(char* out) {
    int i = PEPPER1_MAX_LEN - 1;

    // while out[i] is 'z' set to '0' and continue, else increase by one
    while (i-- > 0) {
        if (out[i] == 'z') {
            out[i] = '0';
            continue;
        }

        if (out[i] == '9') out[i] += 40;
        else out[i]++;
        break;
    }
}
