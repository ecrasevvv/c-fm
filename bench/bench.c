#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../cfm.h"

int main(void) {
    cfm_set_num_threads(69);
    printf("NTHREADS: %d\n", cfm_get_num_threads());
    srand(time(NULL));
    
    cfm_tensor *A = cfm_tensor_full("A", CFM_FLOAT32, 2, ((uint16_t[]){3, 3}), 3.f);
    cfm_tensor *B = cfm_tensor_full("B", CFM_FLOAT32, 2, ((uint16_t[]){3, 3}), 3.f);

    cfm_tensor_print(A, 2);
    cfm_tensor_print(B, 2);

    cfm_tensor *C = cfm_tensor_matmul("C", A, B);
    cfm_tensor_print(C, 2);

    cfm_tensor_free(A);
    cfm_tensor_free(B);
    cfm_tensor_free(C);
    exit(EXIT_SUCCESS);
}
