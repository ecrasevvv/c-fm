/*
 * M=512,N=504,K=256
 *
 * Theoretical maximum on single core:  ~147 GFLOPS
 * numpy archives (single-core):        ~125 GFLOPS (10 NITER)
 * Current best:                        ~100 GFLOPS (10 NITER) 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* i5-1335U */
#define L1d_CACHE_SIZE 576
#define L1i_CACHE_SIZE 352
#define L2_CACHE_SIZE 6.5
#define L3_CACHE_SIZE 12

#define ARR_TYPE float

#define M 512
#define N 504
#define K 256 
#define MAX_VAL 10.f
#define NITER 10

/* IDX formula = i*rows + j where:
 *      - i = col index of the related matrix
 *      - j = row index of the related matrix */
#define idx(i, rows, j) (((i)*rows)+(j))

#define HLINE "--------------------------------------------------------------------------------"
#define INDENT ((int)(log10(((double)MAX_VAL))))

struct timespec start, end;

void fill(size_t rows, size_t cols, ARR_TYPE *m) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            m[idx(j,rows,i)] =  ((float)rand()/(float)(RAND_MAX))*MAX_VAL;
        }
    }
}

void print(const char *name, size_t rows, size_t cols, ARR_TYPE *m) {
    printf("\n%s\n", name);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            printf("%*f ", INDENT, m[idx(j,rows,i)]);
        }
        putchar('\n');
    }
}

void zero_fill(size_t rows, size_t cols, ARR_TYPE *m) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            m[idx(j,rows,i)] = 0.0;
        }
    }
}

/* For further correctness checks */
void baseline(const ARR_TYPE *A, const ARR_TYPE *B, ARR_TYPE *__restrict__ C) {
    // A[M][K], B[K][N], C[M][N]
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t p = 0; p < K; ++p) {
                C[idx(j,M,i)] += A[idx(p,M,i)] * B[idx(j,K,p)];
            }
        }
    }
}

/* AVX2 CPU contains 16 YMM registers.
 * Each YMM register can store up to 8 floats (256 bits in total).
 *
 * Compute an m_{r} x n_{r} sub-matrix \tilde{C} of C. Where r denotes "register".
 *
 * Considering:
 *  - \tilde{C} of size m_{r}   x n_{r} is equals to:
 *  - \tilde{A} of size m_{r}   x K
 *  - \tilde{B} of size K       x r_{r}
 *
 * \tilde{C} = \tilde{A}\tilde{B}
 *
 * Zero-init the \tilde{C} accumulator, load 1 column of \tilde{A} and one row
 * of \tilde{B}. After K iterations the computation of \tilde{C} is completed
 * computing the outer product of the two loaded vectors.
 *
 * In this case "load" means load from RAM to YMM registers. Each column of the 
 * accumulator \tilde{C} can spans over more than 1 YMM register, the same goes
 * for the \tilde{A} column vector.
 *
 * m_{r} must be divisible by 8.
 *
 * (m_{r}/8)*n_{r}  = total YMM registers needed to store the accumulator \tilde{C}
 * m_{r}/8          = total YMM registers needed to store the \tilde{A} col-vector
 * 1                = total YMM registers needed to store the \tilde{B} row-vector single broadcasted value
 *
 * (m_{r}/8 * n_{r} + m_{r}/8 + 1) <= 16
 *
 * for example: m_{r}=8 and n_{r}=12
 */
#define MR 16
#define NR 6
__attribute__((noinline))
void kernel_16x6(float *A_start, float *B_start, float *__restrict__ C_start) {
    __m256 acc[6][2] = {};
    __m256 b_broadcast;
    __m256 a0;
    __m256 a1;

    for (size_t p = 0; p < K; ++p) {
        a0 = _mm256_loadu_ps(&A_start[p * M    ]);
        a1 = _mm256_loadu_ps(&A_start[idx(p,M,8)]);
    
        b_broadcast = _mm256_broadcast_ss(&B_start[p]);
        acc[0][0] = _mm256_fmadd_ps(a0, b_broadcast, acc[0][0]);
        acc[0][1] = _mm256_fmadd_ps(a1, b_broadcast, acc[0][1]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(1,K,p)]);
        acc[1][0] = _mm256_fmadd_ps(a0, b_broadcast, acc[1][0]);
        acc[1][1] = _mm256_fmadd_ps(a1, b_broadcast, acc[1][1]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(2,K,p)]);
        acc[2][0] = _mm256_fmadd_ps(a0, b_broadcast, acc[2][0]);
        acc[2][1] = _mm256_fmadd_ps(a1, b_broadcast, acc[2][1]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(3,K,p)]);
        acc[3][0] = _mm256_fmadd_ps(a0, b_broadcast, acc[3][0]);
        acc[3][1] = _mm256_fmadd_ps(a1, b_broadcast, acc[3][1]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(4,K,p)]);
        acc[4][0] = _mm256_fmadd_ps(a0, b_broadcast, acc[4][0]);
        acc[4][1] = _mm256_fmadd_ps(a1, b_broadcast, acc[4][1]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(5,K,p)]);
        acc[5][0] = _mm256_fmadd_ps(a0, b_broadcast, acc[5][0]);
        acc[5][1] = _mm256_fmadd_ps(a1, b_broadcast, acc[5][1]);
    }

    for (size_t j = 0; j < 6; ++j) {
        _mm256_storeu_ps(&C_start[j * M], acc[j][0]);
        _mm256_storeu_ps(&C_start[idx(j,M,8)], acc[j][1]);
    }
}

void mm(ARR_TYPE *A, ARR_TYPE *B, ARR_TYPE *__restrict__ C) {
    // A[M][K], B[K][N], C[M][N]
    for (size_t i = 0; i < M; i+=MR) {
        for (size_t j = 0; j < N; j+=NR) {
            kernel_16x6(&A[i], &B[j*K], &C[idx(j,M,i)]);
        }
    }
}

void memory_per_array(const char *name, size_t rows, size_t cols) {
    int bytes_per_word = sizeof(ARR_TYPE);
    printf("Memory per array %s = %.1f MiB (%.3f GiB).\n", 
            name,
            bytes_per_word * ((double)rows*cols / 1024.0/1024.0),
            bytes_per_word * ((double)rows*cols / 1024.0/1024.0/1024.0));
}

#define MAX_DIFFERENCE 1e-3
void check(ARR_TYPE *__restrict__ C, ARR_TYPE *__restrict__ _C) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float cij = C[idx(j,M,i)];
            float _cij = _C[idx(j,M,i)];
            float diff = cij - _cij;
            if (fabsf(diff) > MAX_DIFFERENCE) {
                printf("NO MATCH: %f - %f = %f\n", cij, _cij, diff);
                printf("%s\n", HLINE);
                break;
            }
        }
    }
    printf("MATCH\n");
    printf("%s\n", HLINE);
}

void summary(void) {
#ifdef __AVX2__
    printf("Running AVX2 instructions.\n");
#endif
#if defined(__linux__)
    const long int l1_cache = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
#endif
    printf("%s\n", HLINE);
    printf("L1i_CACHE_SIZE:             %d Kib\n", L1i_CACHE_SIZE);
    printf("L1d_CACHE_SIZE:             %d Kib\n", L1d_CACHE_SIZE);
    printf("L2_CACHE_SIZE:              %.1f Mib\n", L2_CACHE_SIZE);
    printf("L3_CACHE_SIZE:              %d Mib\n", L3_CACHE_SIZE);
#if defined(__linux__)
    printf("L1_DCACHE_LINESIZE:         %ld\n", l1_cache);
#endif
    printf("%s\n", HLINE);
    printf("ARR_TYPE_SIZE:              %zu\n", sizeof(ARR_TYPE));
#if defined(__linux__)
    printf("Elements per L1 cache line: %d\n", (int)l1_cache/(int)sizeof(ARR_TYPE));
#endif
    printf("Kernel size:                %dx%d\n", MR, NR);
    printf("%s\n", HLINE);
    printf("Elements of array A: %d\n", M*K); 
    printf("Elements of array B: %d\n", K*N); 
    printf("Elements of array C: %d\n", M*N); 
    printf("%s\n", HLINE);
    memory_per_array("A", M, K);  
    memory_per_array("B", K, N);  
    memory_per_array("C", M, N);  
    printf("%s\n", HLINE);
}

int main(void) {
#ifndef __AVX2__
    fprintf(stderr, "AVX2 not supported.");
    exit(EXIT_FAILURE);
#endif
    /* the pointer returned from malloc is suitably alligned
     * https://pubs.opengroup.org/onlinepubs/7908799/xsh/malloc.html */
    ARR_TYPE *A = (ARR_TYPE*)malloc(sizeof(ARR_TYPE) * M*K);
    ARR_TYPE *B = (ARR_TYPE*)malloc(sizeof(ARR_TYPE) * K*N);
    ARR_TYPE *C = (ARR_TYPE*)malloc(sizeof(ARR_TYPE) * M*N);
    if (!A || !B || !C) exit(EXIT_FAILURE);

    /* Summary: array sizes, number of elements, etc. */
    summary();

    /* Array fill. */
    fill(M, K, A);
    fill(K, N, B);

    /* cache warmup matmul. */
#ifdef WARMUP
    memset(C, 0, M*N*sizeof(ARR_TYPE));
    mm(A, B, C);
#endif

#ifdef DEBUG
    memset(C, 0, M*N*sizeof(ARR_TYPE));
    ARR_TYPE *_C = (ARR_TYPE*)malloc(sizeof(ARR_TYPE) * M*N);
    baseline(A, B, _C);
    mm(A, B, C);
    check(C, _C);
    free(_C);
#endif

    /* Evaluation metrics */
    const int precision = 2;
    double flops = 2.0*M*N*K;
    double gflops = 0.0;

    /* matmul NITER times. */
    double best_time = 1000.0;
    printf("GFLOPS:\t\tTIME:\n");
    for (size_t i = 0; i < NITER; ++i) {
        memset(C, 0, M*N*sizeof(ARR_TYPE));

        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        mm(A, B, C);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        double time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)*1e-9;
        if (time < best_time) best_time = time;
        gflops = (flops/time)*1e-9;
        printf("%.*f\t\t%.*fms\n", precision, gflops, precision, time*1e3);
    }

    printf("%s\n", HLINE);
    printf("Best time:          %.*fms\n", precision, best_time*1e3);
    printf("Best time GFLOPS:   %.*f\n", precision, (flops/best_time)*1e-9);
    printf("%s\n", HLINE);

    /* Otherwise clang with -O3 will assume that C is "dead" and delete all the FMA istructions */
    volatile ARR_TYPE sink = C[0];
    (void)sink;
    free(A); free(B); free(C);
    exit(EXIT_SUCCESS);
}
