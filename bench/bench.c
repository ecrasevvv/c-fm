#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../cfm.h"

int main(void) {
    srand(time(NULL));
    // Tensor_x0 rand
    cfm_tensor *x0 = cfm_tensor_rand("rand",
            CFM_FLOAT32,
            4,
            (uint16_t[]){4, 2, 3, 3});
    //cfm_tensor_print(x0, 4);

    // Tensor_x1 randn
    cfm_tensor *x1 = cfm_tensor_randn("randn",
            CFM_FLOAT64,
            3,
            (uint16_t[]){2, 2, 10});
    //cfm_tensor_print(x1, 6);

    // linspace Tensor_x2
    cfm_tensor *x2 = cfm_tensor_linspace("linspace",
            CFM_FLOAT64,
            1.0,
            0.0,
            0.1);
    //cfm_tensor_print(x2, 2);

    // full Tensor_x3
    cfm_tensor *x3 = cfm_tensor_full("full",
            CFM_FLOAT64,
            3,
            ((uint16_t[]){2, 3, 1}),
            42.0);
    //cfm_tensor_print(x3, 2);

    // zeros Tensor_x4
    cfm_tensor *x4 = cfm_tensor_ones("x4-ones",
            CFM_FLOAT64,
            3,
            (uint16_t[]){2, 3, 3});
    //cfm_tensor_print(x4, 2);

    // ones Tensor_x5
    cfm_tensor *x5 = cfm_tensor_ones("ones",
            CFM_FLOAT64,
            3,
            (uint16_t[]){2, 3, 3});
    //cfm_tensor_print(x5, 2);

    // cat Tensor_x6
    const cfm_tensor *tensors[2] = {x4, x5};
    cfm_tensor *x6 = cfm_tensor_cat("cat",
            tensors,
            2,
            2);
    //cfm_tensor_print(x6, 4);

    // last Tensor_x7
    cfm_tensor *x7 = cfm_tensor_get_last(x0);
    //cfm_tensor_print(x7, 4);
    
    // add Tensor_x8
    // x4 + x_add
    cfm_tensor *x_add = cfm_tensor_ones("x_add", CFM_FLOAT64, 2, (uint16_t[]){3, 1});
    //cfm_tensor_print(x_add, 2);
    cfm_tensor *x8 = cfm_tensor_add("add", x4, x_add);
    //cfm_tensor_print(x8, 2);
    
    // mul Tensor_mul
    cfm_tensor *m1 = cfm_tensor_full("full",
            CFM_FLOAT64,
            3,
            ((uint16_t[]){1, 3, 1}),
            2.0);
    cfm_tensor *m2 = cfm_tensor_full("full",
            CFM_FLOAT64,
            3,
            ((uint16_t[]){2, 3, 1}),
            3.0);
    cfm_tensor *mul = cfm_tensor_mul("mul", m1, m2);
    cfm_tensor_print(m1, 2);
    cfm_tensor_print(m2, 2);
    cfm_tensor_print(mul, 2);

    // exp Tensor_x9
    cfm_tensor *x9 = cfm_tensor_exp("exp", x5);
    //cfm_tensor_print(x9, 6);
    
    // expanded Tensor_x10
    cfm_tensor *_x10 = cfm_tensor_ones("_x10", CFM_FLOAT32, 2, (uint16_t[]){3, 1});
    //cfm_tensor_print(_x10, 2);
    cfm_tensor *x10 = cfm_tensor_expand(_x10, 3, (uint16_t[]){4, 3, 10});
    //cfm_tensor_print(x10, 2);

    cfm_tensor_free(x0);
    cfm_tensor_free(x1);
    cfm_tensor_free(x2);
    cfm_tensor_free(x3);
    cfm_tensor_free(x4);
    cfm_tensor_free(x5);
    cfm_tensor_free(x6);
    cfm_tensor_free(x7);
    cfm_tensor_free(x_add);
    cfm_tensor_free(x8);
    cfm_tensor_free(x9);
    cfm_tensor_free(_x10);
    cfm_tensor_free(x10);
    cfm_tensor_free(m1);
    cfm_tensor_free(m2);
    cfm_tensor_free(mul);
    return 0;
}
