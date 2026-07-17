/*
 * M=N=512
 * K=256
 *
 * Theoretical maximum on single core:  ~147 GFLOPS
 * numpy archives (single-core):        ~100 GFLOPS
 * Current best:                        nd
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
#define N 512
#define K 256 
#define BS 8
#define MAX_VAL 10
#define NITER 10

#ifdef __AVX2__
_Static_assert(BS%8==0, "BS must be a multiple of 8 when using AVX2 intrinsics and ARR_TYPE=float");
#endif

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
            m[idx(j,rows,i)] = rand()%MAX_VAL;
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

void mm(const ARR_TYPE *A, const ARR_TYPE *B, ARR_TYPE *__restrict__ C) {
    // A[M][K], B[K][N], C[M][N]
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t p = 0; p < K; ++p) {
                C[idx(j,M,i)] += A[idx(p,M,i)] * B[idx(j,K,p)];
            }
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

void summary(void) {
#if defined(__AVX2__)
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
    printf("Block dimension:            %dx%d\n", BS, BS);
    printf("Elements per block:         %d\n", BS*BS);
    printf("Block size (bytes):         %ld\n", sizeof(ARR_TYPE)*(BS*BS));
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
    assert(N%BS==0 && M%BS==0 && K%BS==0);
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

    free(A); free(B); free(C);
    exit(EXIT_SUCCESS);
}
