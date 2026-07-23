#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../cfm.h"

int main(void) {
    srand(time(NULL));
    
    cfm_tensor *A = cfm_tensor_full("A", CFM_FLOAT32, 2, ((uint16_t[]){3, 3}), 3.f);
    cfm_tensor *B = cfm_tensor_full("B", CFM_FLOAT32, 1, ((uint16_t[]){3}), 3.f);

    cfm_tensor_print(A, 2);
    cfm_tensor_print(B, 2);

    cfm_tensor *C = cfm_tensor_matmul("C", A, B);
    cfm_tensor_print(C, 2);

    cfm_tensor_free(A);
    cfm_tensor_free(B);
    cfm_tensor_free(C);
    exit(EXIT_SUCCESS);
}
