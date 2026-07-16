/*
 * Theoretical maximum on single core: ~147 GFLOPS
 *
 * Current best:
 *      - M=N=2048
 *      - P=1024
 *      - B=2
 *      - NITER=10
 *      
 *      ./mm.sh 1 1
 *      Best time:          0.1451  
 *      Best time GFLOPS:   59.2046
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* i5-1335U */
#define L1d_CACHE_SIZE 576
#define L1i_CACHE_SIZE 352
#define L2_CACHE_SIZE 6.5
#define L3_CACHE_SIZE 12

#define ARR_TYPE float

#define M 2048
#define N 2048
#define P 2048 
#define BS 2
#define MAX_VAL 10
#define NITER 10

#if (M%BS != 0 || N%BS != 0 || P%BS != 0) 
#error "BS and matrix dimensions are not compatible."
#endif

/* IDX formula = i*cols + j where:
 *      - i = rows index of the related matrix
 *      - j = cols index of the related matrix */
#define idx(i, cols, j) (((i)*cols)+(j))

#define HLINE "--------------------------------------------------------------------------------"
#define INDENT ((int)(log10(((double)MAX_VAL))))

struct timespec start, end;

void fill(size_t rows, size_t cols, ARR_TYPE *m) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            m[i * cols + j] = rand()%MAX_VAL;
        }
    }
}

void print(const char *name, size_t rows, size_t cols, ARR_TYPE *m) {
    printf("\n%s\n", name);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            printf("%*f ", INDENT, m[i * cols + j]);
        }
        putchar('\n');
    }
}

void zero_fill(size_t rows, size_t cols, ARR_TYPE *m) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            m[i * cols + j] = 0.0;
        }
    }
}

void mm(size_t n,   size_t m,   const ARR_TYPE *A,
                    size_t p,   const ARR_TYPE *B,
                                ARR_TYPE *__restrict__ C) {
    /* A[N][M], B[M][P], C[N][P] */
    for (size_t bi = 0; bi < n; bi+=BS) {                           /* block: A/C rows */
        for (size_t bk = 0; bk < m; bk+=BS) {                       /* block: A/B shared dim, A cols B rows */
            for (size_t bj = 0; bj < p; bj+=BS) {                   /* block: B/C cols */

                for (size_t i = bi; i < bi+BS; ++i) {               /* in-block: A/C block rows */
                    for (size_t k = bk; k < bk+BS; ++k) {           /* in-block: A/B block shared dim */
                        ARR_TYPE aik = A[idx(i,m,k)];               /* scalar, constant in inner loop, broadcasted */
                        for (size_t j = bj; j < bj+BS; ++j) {       /* in-block: B/C block cols */
                            C[idx(i,p,j)] += aik * B[idx(k,p,j)];
                        }
                    }
                }
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
    const long int l1_cache = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    printf("%s\n", HLINE);
    printf("L1i_CACHE_SIZE:             %d Kib\n", L1i_CACHE_SIZE);
    printf("L1d_CACHE_SIZE:             %d Kib\n", L1d_CACHE_SIZE);
    printf("L2_CACHE_SIZE:              %.1f Mib\n", L2_CACHE_SIZE);
    printf("L3_CACHE_SIZE:              %d Mib\n", L3_CACHE_SIZE);
    printf("L1_DCACHE_LINESIZE:         %ld\n", l1_cache);
    printf("%s\n", HLINE);
    printf("ARR_TYPE_SIZE:              %zu\n", sizeof(ARR_TYPE));
    printf("Elements per L1 cache line: %d\n", (int)l1_cache/(int)sizeof(ARR_TYPE));
    printf("Block dimension:            %dx%d\n", BS, BS);
    printf("Elements per block:         %d\n", BS*BS);
    printf("Block size (bytes):         %ld\n", sizeof(ARR_TYPE)*(BS*BS));
    printf("%s\n", HLINE);
    printf("Elements of array A: %d\n", M*N); 
    printf("Elements of array B: %d\n", N*P); 
    printf("Elements of array C: %d\n", M*P); 
    printf("%s\n", HLINE);
    memory_per_array("A", M, N);  
    memory_per_array("B", N, P);  
    memory_per_array("C", M, P);  
    printf("%s\n", HLINE);
}

int main(void) {
    ARR_TYPE *A = (ARR_TYPE*)malloc(sizeof(ARR_TYPE) * N*M);
    ARR_TYPE *B = (ARR_TYPE*)malloc(sizeof(ARR_TYPE) * M*P);
    ARR_TYPE *C = (ARR_TYPE*)malloc(sizeof(ARR_TYPE) * N*P);

    /* Summary: array sizes, number of elements, etc. */
    summary();

    /* Array fill. */
    fill(N, M, A);
    fill(M, P, B);
    zero_fill(N, P, C);

    /* cache warmup matmul. */
#ifdef DWARMUP
    mm(N, M, A, P, B, C);
#endif

    /* Evaluation metrics */
    const int precision = 4;
    double flops = 2.0*M*P*N;
    double gflops = 0.0;

    /* matmul NITER times. */
    double best_time = 1000.0;
    printf("GFLOPS:\t\tTIME:\n");
    for (size_t i = 0; i < NITER; ++i) {
        zero_fill(N, P, C);
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        mm(N, M, A, P, B, C);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        double time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)*1e-9;
        if (time < best_time) best_time = time;
        gflops = (flops/time)*1e-9;
        printf("%.*f\t\t%.*fs\n", precision, gflops, precision, time);
    }
    printf("%s\n", HLINE);

    printf("Best time:          %.*f\n", precision, best_time);
    printf("Best time GFLOPS:   %.*f\n", precision, (flops/best_time)*1e-9);
    printf("%s\n", HLINE);

    free(A); free(B); free(C);
    return 0;
}
