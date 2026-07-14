/* This file is used ONLY to test performance of various cfm_tensor ops
 * (add, mul, matmul, etc.). Avoid compiling this with -DDEBUG flag. */

#define _GNU_SOURCE

#include <stdio.h>
#include <time.h>
#include "cfm.h"

struct timespec start, end;

int main(void) {
    cfm_tensor *m1 = cfm_tensor_rand("m1", CFM_FLOAT32, 3, (uint16_t[]){2, 4096, 4096});
    cfm_tensor *m2 = cfm_tensor_rand("m2", CFM_FLOAT32, 3, (uint16_t[]){1, 4096, 4096});
   
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    (void*)cfm_tensor_mul("mul", m1, m2);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    double elaps_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)*1e-9;
    printf("Elapsed seconds: %.*f\n", 4, elaps_sec);

    cfm_tensor_free(m1);
    cfm_tensor_free(m2);
    return 0;
}
