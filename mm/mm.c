/*
 * i5-1335U specs: https://www.intel.com/content/www/us/en/products/sku/232153/intel-core-i51335u-processor-12m-cache-up-to-4-60-ghz/specifications.html
 *
 * M=512,N=504,K=256
 *
 * Theoretical maximum on single core:  ~147 GFLOPS
 * numpy archives (single-core):        ~125 GFLOPS (10 NITER)
 * Current best:                        ~123 GFLOPS (10 NITER) 
 *
 * 10 cores in total: 2 P-core and 8 E-core. Both supports AVX2 instructions.
 * https://en.wikipedia.org/wiki/Raptor_Lake
 *
 * Maximum P-core Turbo frequency: 4.6Ghz
 * Maximum E-core Turbo frequency: 3.4Ghz
 *
 * Considering running always at the top frequency:
 *  - theoretical maximum of P-cores (combined): ~294 GFLOPS
 *  - theoretical maximum of E-cores (combined): ~870 GFLOPS
 *
 * So the theoretical maximum on mult-core will be: ~1164 GFLOPS
 * 
 * numpy archives (12 threads):         ~420 GFLOPS (10 NITER)
 * Current best (12 threads):           ~435 GFLOPS (10 NITER)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* i5-1335U */
#if defined(__linux__)
#define L1_DCACHE_LINESIZE (sysconf(_SC_LEVEL1_DCACHE_LINESIZE))
/* Hardcoded, read L2_CACHE_SIZE below */
#define L1_DCACHE_SIZE 352
#define L1_ICACHE_SIZE 576
/* getconf LEVEL2_CACHE_SIZE reports the L2 size visible to the specific CPU core it happened to be scheduled on. 
 * E-cores: 2Mib, P-cores: 1.2Mib, not deterministic behavior. That's why it is hardcoded. 
 * Run $lscpu | grep cache to get the total ammount and the number of instances */
//#define L2_CACHE_SIZE ((sysconf(_SC_LEVEL2_CACHE_SIZE))/1024.0/1024.0)
#define L2_CACHE_SIZE 6.5
#define L3_CACHE_SIZE ((sysconf(_SC_LEVEL3_CACHE_SIZE))/1024/1024)
#endif

#define ARR_TYPE float

#define M 512
#define N 504
#define K 256 
#define MAX_VAL 10.f
#define WARMUPS 20
#define NITER 10

#ifdef MULTI_THREAD
#include <omp.h>
#define NTHREADS 12
#endif

/* Col-major indexing */
#define idx(i, rows, j) (((i)*rows)+(j))

/* https://stackoverflow.com/questions/1898153/how-to-determine-if-memory-is-aligned */
#define is_aligned(POINTER, BYTE_COUNT) \
    (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0)

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
 * Compute an m_{r} x n_{r} sub-matrix _C of C. Where r denotes "register".
 *
 * Considering:
 *  - _C of size m_{r}   x n_{r} is equals to:
 *  - _A of size m_{r}   x K
 *  - _B of size K       x r_{r}
 *
 * _C = _A _B
 *
 * Zero-init the _C accumulator, load 1 column of _A and one row
 * of _B. After K iterations the computation of _C is completed
 * computing the outer product of the two loaded vectors.
 *
 * In this case "load" means load from RAM to YMM registers. Each column of the 
 * accumulator _C can spans over more than 1 YMM register, the same goes
 * for the _A column vector.
 *
 * m_{r} must be divisible by 8.
 *
 * (m_{r}/8)*n_{r}  = total YMM registers needed to store the accumulator _C
 * m_{r}/8          = total YMM registers needed to store the _A col-vector
 * 1                = total YMM registers needed to store the _B row-vector single broadcasted value
 *
 * (m_{r}/8 * n_{r} + m_{r}/8 + 1) <= 16
 *
 * for example: m_{r}=8 and n_{r}=12
 */
//#define MR 16
//#define NR 6 
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

//#define MR 8
//#define NR 14
__attribute__((noinline))
void kernel_8x14(float *A_start, float *B_start, float *__restrict__ C_start) {
    __m256 acc[14] = {};
    __m256 a0;
    __m256 b_broadcast;

    for (size_t p = 0; p < K; ++p) {
        a0 = _mm256_loadu_ps(&A_start[p*M]);

        b_broadcast = _mm256_broadcast_ss(&B_start[p]);
        acc[0] = _mm256_fmadd_ps(a0, b_broadcast, acc[0]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(1,K,p)]);
        acc[1] = _mm256_fmadd_ps(a0, b_broadcast, acc[1]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(2,K,p)]);
        acc[2] = _mm256_fmadd_ps(a0, b_broadcast, acc[2]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(3,K,p)]);
        acc[3] = _mm256_fmadd_ps(a0, b_broadcast, acc[3]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(4,K,p)]);
        acc[4] = _mm256_fmadd_ps(a0, b_broadcast, acc[4]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(5,K,p)]);
        acc[5] = _mm256_fmadd_ps(a0, b_broadcast, acc[5]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(6,K,p)]);
        acc[6] = _mm256_fmadd_ps(a0, b_broadcast, acc[6]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(7,K,p)]);
        acc[7] = _mm256_fmadd_ps(a0, b_broadcast, acc[7]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(8,K,p)]);
        acc[8] = _mm256_fmadd_ps(a0, b_broadcast, acc[8]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(9,K,p)]);
        acc[9] = _mm256_fmadd_ps(a0, b_broadcast, acc[9]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(10,K,p)]);
        acc[10] = _mm256_fmadd_ps(a0, b_broadcast, acc[10]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(11,K,p)]);
        acc[11] = _mm256_fmadd_ps(a0, b_broadcast, acc[11]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(12,K,p)]);
        acc[12] = _mm256_fmadd_ps(a0, b_broadcast, acc[12]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(13,K,p)]);
        acc[13] = _mm256_fmadd_ps(a0, b_broadcast, acc[13]);
    }

    for (size_t j = 0; j < 14; ++j) {
        _mm256_storeu_ps(&C_start[j * M], acc[j]);
    }
}

//#define MR 8
//#define NR 12
__attribute__((noinline))
void kernel_8x12(float *A_start, float *B_start, float *__restrict__ C_start) {
    __m256 acc[12] = {};
    __m256 a0;
    __m256 b_broadcast;

    for (size_t p = 0; p < K; ++p) {
        a0 = _mm256_loadu_ps(&A_start[p*M]);

        b_broadcast = _mm256_broadcast_ss(&B_start[p]);
        acc[0] = _mm256_fmadd_ps(a0, b_broadcast, acc[0]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(1,K,p)]);
        acc[1] = _mm256_fmadd_ps(a0, b_broadcast, acc[1]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(2,K,p)]);
        acc[2] = _mm256_fmadd_ps(a0, b_broadcast, acc[2]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(3,K,p)]);
        acc[3] = _mm256_fmadd_ps(a0, b_broadcast, acc[3]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(4,K,p)]);
        acc[4] = _mm256_fmadd_ps(a0, b_broadcast, acc[4]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(5,K,p)]);
        acc[5] = _mm256_fmadd_ps(a0, b_broadcast, acc[5]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(6,K,p)]);
        acc[6] = _mm256_fmadd_ps(a0, b_broadcast, acc[6]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(7,K,p)]);
        acc[7] = _mm256_fmadd_ps(a0, b_broadcast, acc[7]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(8,K,p)]);
        acc[8] = _mm256_fmadd_ps(a0, b_broadcast, acc[8]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(9,K,p)]);
        acc[9] = _mm256_fmadd_ps(a0, b_broadcast, acc[9]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(10,K,p)]);
        acc[10] = _mm256_fmadd_ps(a0, b_broadcast, acc[10]);

        b_broadcast = _mm256_broadcast_ss(&B_start[idx(11,K,p)]);
        acc[11] = _mm256_fmadd_ps(a0, b_broadcast, acc[11]);
    }

    for (size_t j = 0; j < 12; ++j) {
        _mm256_storeu_ps(&C_start[j * M], acc[j]);
    }
}

#define MR 32
#define NR 2 
__attribute__((noinline))
void kernel_32x2(float *A_start, float *B_start, float *__restrict__ C_start) {
    __m256 acc[2][4] = {};
    __m256 a0;
    __m256 a1;
    __m256 a2;
    __m256 a3;
    __m256 b_broadcast;

    for (size_t p = 0; p < K; ++p) {
        a0 = _mm256_loadu_ps(&A_start[p*M        ]);
        a1 = _mm256_loadu_ps(&A_start[idx(p,M,8) ]);
        a2 = _mm256_loadu_ps(&A_start[idx(p,M,16)]);
        a3 = _mm256_loadu_ps(&A_start[idx(p,M,24)]);

        b_broadcast = _mm256_loadu_ps(&B_start[p]);
        acc[0][0] = _mm256_fmadd_ps(a0, b_broadcast, acc[0][0]);
        acc[0][1] = _mm256_fmadd_ps(a1, b_broadcast, acc[0][1]);
        acc[0][2] = _mm256_fmadd_ps(a2, b_broadcast, acc[0][2]);
        acc[0][3] = _mm256_fmadd_ps(a3, b_broadcast, acc[0][3]);

        b_broadcast = _mm256_loadu_ps(&B_start[idx(1,K,p)]);
        acc[1][0] = _mm256_fmadd_ps(a0, b_broadcast, acc[1][0]);
        acc[1][1] = _mm256_fmadd_ps(a1, b_broadcast, acc[1][1]);
        acc[1][2] = _mm256_fmadd_ps(a2, b_broadcast, acc[1][2]);
        acc[1][3] = _mm256_fmadd_ps(a3, b_broadcast, acc[1][3]);
    }

    for (size_t j = 0; j < 2; ++j) {
        _mm256_storeu_ps(&C_start[j * M], acc[j][0]);
        _mm256_storeu_ps(&C_start[idx(j,M,8)], acc[j][1]);
        _mm256_storeu_ps(&C_start[idx(j,M,16)], acc[j][2]);
        _mm256_storeu_ps(&C_start[idx(j,M,24)], acc[j][3]);
    }
}

void mm(ARR_TYPE *A, ARR_TYPE *B, ARR_TYPE *__restrict__ C) {
    // A[M][K], B[K][N], C[M][N]
    #pragma omp parallel for collapse(2) num_threads(NTHREADS)
    for (size_t i = 0; i < M; i+=MR) {
        for (size_t j = 0; j < N; j+=NR) {
            //kernel_16x6(&A[i], &B[j*K], &C[idx(j,M,i)]);
            //kernel_8x14(&A[i], &B[j*K], &C[idx(j,M,i)]);
            //kernel_8x12(&A[i], &B[j*K], &C[idx(j,M,i)]);
            kernel_32x2(&A[i], &B[j*K], &C[idx(j,M,i)]);
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
#ifdef MULTI_THREAD
    printf("Running on: %d threads.\n", NTHREADS);
#else
    printf("Running on: 1 thread.\n");
#endif
#if defined(__linux__)
    printf("%s\n", HLINE);
    printf("L1i_CACHE_SIZE:             %d Kib\n", L1_ICACHE_SIZE);
    printf("L1d_CACHE_SIZE:             %d Kib\n", L1_DCACHE_SIZE);
    printf("L2_CACHE_SIZE:              %.1f Mib\n", L2_CACHE_SIZE);
    printf("L3_CACHE_SIZE:              %ld Mib\n", L3_CACHE_SIZE);
    printf("L1_DCACHE_LINESIZE:         %ld\n", L1_DCACHE_LINESIZE);
#endif
    printf("%s\n", HLINE);
    printf("ARR_TYPE_SIZE:              %zu\n", sizeof(ARR_TYPE));
#if defined(__linux__)
    printf("Elements per L1 cache line: %d\n", (int)L1_DCACHE_LINESIZE/(int)sizeof(ARR_TYPE));
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
    assert(M%MR==0 && N%NR==0);
#ifndef __AVX2__
    fprintf(stderr, "AVX2 not supported.");
    exit(EXIT_FAILURE);
#endif
    /* the pointer returned from malloc is suitably alligned
     * https://pubs.opengroup.org/onlinepubs/7908799/xsh/malloc.html */
    ARR_TYPE *A = (ARR_TYPE*)aligned_alloc(L1_DCACHE_LINESIZE, sizeof(ARR_TYPE) * M*K);
    ARR_TYPE *B = (ARR_TYPE*)aligned_alloc(L1_DCACHE_LINESIZE, sizeof(ARR_TYPE) * K*N);
    ARR_TYPE *C = (ARR_TYPE*)aligned_alloc(L1_DCACHE_LINESIZE, sizeof(ARR_TYPE) * M*N);
    if (!A || !B || !C) exit(EXIT_FAILURE);
    if (is_aligned(A, L1_DCACHE_LINESIZE)) printf("A aligned.\n");
    if (is_aligned(B, L1_DCACHE_LINESIZE)) printf("B aligned.\n");
    if (is_aligned(C, L1_DCACHE_LINESIZE)) printf("C aligned.\n");

    /* Summary: array sizes, number of elements, etc. */
    summary();

    /* Array fill. */
    fill(M, K, A);
    fill(K, N, B);

#ifdef MULTI_THREAD
    #pragma omp parallel num_threads(NTHREADS)
#endif
    /* Cache warmup */
    for (size_t w = 0; w < WARMUPS; ++w) {
        memset(C, 0, M*N*sizeof(ARR_TYPE));
        mm(A, B, C);
    }

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
