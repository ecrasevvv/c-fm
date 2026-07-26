/* This file is used to benchmark (measure time and GFLOPS) various
 * functions from cfm.c (primarly OPS on cfm_tensor functions). 
 * #define what you need to benchmark. To understand look at the code 
 * down below. This solution just works, I know that is not the cleanest
 * one, but I dont want to mess too much the Makefile. */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../cfm.h"

struct timespec start, end;

#define W_NITER 20
#define NITER 10

#define MM_2D_2D
#ifdef MM_2D_2D
int main(void) {
#ifdef __AVX2__
    printf("AVX2 micro-kernel active using mm_f\n");
#else
    printf("Using mm_base_f (no AVX2)\n");
#endif
    printf("Benchmarking 2Dx2D matmul.\n");
    cfm_set_num_threads(12);
    printf("NTHREADS: %d\n", cfm_get_num_threads());
    srand(time(NULL));
    
    cfm_tensor *A = cfm_tensor_rand("A", CFM_FLOAT32, 2, ((uint16_t[]){4098, 512}));
    cfm_tensor *B = cfm_tensor_rand("B", CFM_FLOAT32, 2, ((uint16_t[]){512, 4096}));

    uint16_t m = A->shape[0];
    uint16_t n = B->shape[1];
    uint16_t k = B->shape[0];

    float *A_data = A->data;
    float *B_data = B->data;
    cfm_tensor *C = cfm_tensor_zeros("C", CFM_FLOAT32, 2,
            ((uint16_t[]){m, n}));
    float *__restrict__ C_data = C->data;

    /* warmup */
    for (size_t i = 0; i < W_NITER; ++i) {
        memset(C->data, 0.0, C->numel*sizeof(float));
#ifdef __AVX2__
        mm_f_wrapper(C_data, m, n, A_data, k, B_data);
#else
        mm_base_f_wrapper(C_data, m, n, A_data, k, B_data);
#endif
    }
    
    /* Evaluation metrics */
    const int precision = 2;
    double flops = 2.0*m*n*k;
    double gflops = 0.0;
    double best_time = 1000.0;

    /* Timed bench */
    printf("GFLOPS:\t\tTIME:\n");
    for (size_t i = 0; i < NITER; ++i) {
        memset(C->data, 0.0, C->numel*sizeof(float));
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
#ifdef __AVX2__
        mm_f_wrapper(C_data, m, n, A_data, k, B_data);
#else
        mm_base_f_wrapper(C_data, m, n, A_data, k, B_data);
#endif
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        
        double time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)*1e-9;
        if (time < best_time) best_time = time;
        gflops = (flops/time)*1e-9;
        printf("%.*f\t\t%.*fms\n", precision, gflops, precision, time*1e3);
    }
    printf("Best time:          %.*fms\n", precision, best_time*1e3);
    printf("Best time GFLOPS:   %.*f\n", precision, (flops/best_time)*1e-9);

    cfm_tensor_free(A);
    cfm_tensor_free(B);
    cfm_tensor_free(C);
    exit(EXIT_SUCCESS);
}
#endif /* MM_2D_2D */

//#define MM_2D_1D
#ifdef MM_2D_1D
int main(void) {
#ifdef __AVX2__
    printf("AVX2 micro-kernel active using mm_f\n");
#else
    printf("Using mm_base_f (no AVX2)\n");
#endif
    printf("Benchmarking 2Dx1D matmul.\n");
    cfm_set_num_threads(12);
    printf("NTHREADS: %d\n", cfm_get_num_threads());
    srand(time(NULL));
    
    cfm_tensor *A = cfm_tensor_rand("A", CFM_FLOAT32, 2, ((uint16_t[]){4098, 512}));
    cfm_tensor *B = cfm_tensor_rand("B", CFM_FLOAT32, 1, ((uint16_t[]){512}));

    uint16_t m = A->shape[0];
    uint16_t k = A->shape[1];

    float *A_data = A->data;
    float *B_data = B->data;
    cfm_tensor *C = cfm_tensor_zeros("C", CFM_FLOAT32, 1,
            ((uint16_t[]){m}));
    float *__restrict__ C_data = C->data;

    /* warmup */
    for (size_t i = 0; i < W_NITER; ++i) {
        memset(C->data, 0.0, C->numel*sizeof(float));
        cfm_tensor_matrix_vector_prod_wrapper(A_data, m, k, B_data, C_data);
    }

    /* Evaluation metrics */
    const int precision = 2;
    double flops = 2.0*m*k;
    double gflops = 0.0;
    double best_time = 1000.0;

    /* Timed bench */
    printf("GFLOPS:\t\tTIME:\n");
    for (size_t i = 0; i < NITER; ++i) {
        memset(C->data, 0.0, C->numel*sizeof(float));
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        cfm_tensor_matrix_vector_prod_wrapper(A_data, m, k, B_data, C_data);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        double time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)*1e-9;
        if (time < best_time) best_time = time;
        gflops = (flops/time)*1e-9;
        printf("%.*f\t\t%.*fms\n", precision, gflops, precision, time*1e3);
    }
    printf("Best time:          %.*fms\n", precision, best_time*1e3);
    printf("Best time GFLOPS:   %.*f\n", precision, (flops/best_time)*1e-9);

    cfm_tensor_free(A);
    cfm_tensor_free(B);
    cfm_tensor_free(C);
    exit(EXIT_SUCCESS);
}
#endif /* MM_2D_1D */
