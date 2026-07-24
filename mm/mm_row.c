/* This is just the same as mm.c but for row-major matrices. */

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

#define ARR_TYPE float

#if defined(__linux__)
#define L1_DCACHE_LINESIZE (sysconf(_SC_LEVEL1_DCACHE_LINESIZE))
#endif

#define M_RAW 512
#define N_RAW 512
#define K 256
#define MR 6
#define NR 16

#define M_PAD(n) ((n) % MR ? (n) + MR - ((n) % MR) : (n))
#define N_PAD(n) ((n) % NR ? (n) + NR - ((n) % NR) : (n))

#define M M_PAD(M_RAW)
#define N N_PAD(N_RAW)

#define MAX_VAL 1.f
#define WARMUPS 20
#define NITER 10

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef NTHREADS
#define NTHREADS 12
#endif

/* Row-major indexing */
#define idx(row, cols, col) (((row)*cols)+(col))

/* https://stackoverflow.com/questions/1898153/how-to-determine-if-memory-is-aligned */
#define is_aligned(POINTER, BYTE_COUNT) \
    (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0)

#define INDENT ((int)(log10(((double)MAX_VAL))))

struct timespec start, end;

void fill(ARR_TYPE *m, const size_t rows, const size_t cols) {
    for (size_t i = 0; i < rows*cols; ++i) m[i] = ((ARR_TYPE)rand()/(ARR_TYPE)(RAND_MAX))*MAX_VAL;
}

void print(const char *name, const ARR_TYPE *m, const size_t rows, const size_t cols) {
    printf("%s:\n", name);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            printf("%*f ", INDENT, m[idx(i,cols,j)]);
        }
        putchar('\n');
    }
}

__attribute__((noinline)) static void kernel_6x16(const ARR_TYPE *A_start, const size_t k,
        const ARR_TYPE *B_start, const size_t n,
        ARR_TYPE *C_start) {
    __m256 acc[6][2] = {};
    __m256 a_broadcast;
    __m256 b0;
    __m256 b1;

    for (size_t p = 0; p < k; ++p) {
        b0 = _mm256_loadu_ps(&B_start[p*n]);
        b1 = _mm256_loadu_ps(&B_start[idx(p,n,8)]);

        a_broadcast = _mm256_broadcast_ss(&A_start[p]);
        acc[0][0] = _mm256_fmadd_ps(a_broadcast, b0, acc[0][0]);
        acc[0][1] = _mm256_fmadd_ps(a_broadcast, b1, acc[0][1]);

        a_broadcast = _mm256_broadcast_ss(&A_start[idx(1,k,p)]);
        acc[1][0] = _mm256_fmadd_ps(a_broadcast, b0, acc[1][0]);
        acc[1][1] = _mm256_fmadd_ps(a_broadcast, b1, acc[1][1]);

        a_broadcast = _mm256_broadcast_ss(&A_start[idx(2,k,p)]);
        acc[2][0] = _mm256_fmadd_ps(a_broadcast, b0, acc[2][0]);
        acc[2][1] = _mm256_fmadd_ps(a_broadcast, b1, acc[2][1]);

        a_broadcast = _mm256_broadcast_ss(&A_start[idx(3,k,p)]);
        acc[3][0] = _mm256_fmadd_ps(a_broadcast, b0, acc[3][0]);
        acc[3][1] = _mm256_fmadd_ps(a_broadcast, b1, acc[3][1]);

        a_broadcast = _mm256_broadcast_ss(&A_start[idx(4,k,p)]);
        acc[4][0] = _mm256_fmadd_ps(a_broadcast, b0, acc[4][0]);
        acc[4][1] = _mm256_fmadd_ps(a_broadcast, b1, acc[4][1]);

        a_broadcast = _mm256_broadcast_ss(&A_start[idx(5,k,p)]);
        acc[5][0] = _mm256_fmadd_ps(a_broadcast, b0, acc[5][0]);
        acc[5][1] = _mm256_fmadd_ps(a_broadcast, b1, acc[5][1]);
    }

    for (size_t r = 0; r < 6; ++r) {
        _mm256_storeu_ps(&C_start[r * n    ], acc[r][0]);
        _mm256_storeu_ps(&C_start[r * n + 8], acc[r][1]);
    }
}

static ARR_TYPE *pad_rows_rm(const ARR_TYPE *src, size_t rows_raw, size_t cols,
        size_t align, size_t *rows_padded) {
    size_t add = (rows_raw % align) ? align - (rows_raw % align) : 0;
    *rows_padded = rows_raw + add;
    if (!add) return NULL;
    ARR_TYPE *dst = (ARR_TYPE*)calloc((*rows_padded) * cols, sizeof(ARR_TYPE));
    if (!dst) exit(EXIT_FAILURE);
    memcpy(dst, src, rows_raw * cols * sizeof(ARR_TYPE));
    return dst;
}

static ARR_TYPE *pad_cols_rm(const ARR_TYPE *src, size_t rows, size_t cols_raw,
        size_t align, size_t *cols_padded) {
    size_t add = (cols_raw % align) ? align - (cols_raw % align) : 0;
    *cols_padded = cols_raw + add;
    if (!add) return NULL;
    ARR_TYPE *dst = (ARR_TYPE*)calloc(rows * (*cols_padded), sizeof(ARR_TYPE));
    if (!dst) exit(EXIT_FAILURE);
    for (size_t r = 0; r < rows; ++r)
        memcpy(&dst[r * (*cols_padded)], &src[r * cols_raw], cols_raw * sizeof(ARR_TYPE));
    return dst;
}

void mm(const ARR_TYPE *A, const size_t m, const size_t k,
        const ARR_TYPE *B, const size_t n,
        ARR_TYPE *__restrict__ C) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) num_threads(NTHREADS)
#endif
    for (size_t i = 0; i < m; i+=MR) {
        for (size_t j = 0; j < n; j+=NR) {
            kernel_6x16(&A[i * k], k, &B[j], n, &C[i * n + j]);
        }
    }
}

void baseline(const ARR_TYPE *A, const size_t m, const size_t k,
        const ARR_TYPE *B, const size_t n,
        ARR_TYPE *__restrict__ C) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            ARR_TYPE acc = 0.f;
            for (size_t p = 0; p < k; ++p) {
                acc += A[idx(i,k,p)] * B[idx(p,n,j)];
            }
            C[idx(i,n,j)] = acc;
        }
    }
}

#define MAX_DIFFERENCE 1e-3
void check(ARR_TYPE *__restrict__ C, ARR_TYPE *__restrict__ _C,
        const size_t rows, const size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            float cij = C[idx(i,cols,j)];
            float _cij = _C[idx(i,cols,j)];
            float diff = cij - _cij;
            if (fabsf(diff) > MAX_DIFFERENCE) {
                printf("NO MATCH: %f - %f = %f\n", cij, _cij, diff);
                break;
            }
        }
    }
    printf("MATCH\n");
}

int main(void) {
#ifndef __AVX2__
    fprintf(stderr, "AVX2 not supported.");
    exit(EXIT_FAILURE);
#endif
    ARR_TYPE *A_raw = (ARR_TYPE*)aligned_alloc(L1_DCACHE_LINESIZE, sizeof(ARR_TYPE) * M_RAW*K);
    ARR_TYPE *B_raw = (ARR_TYPE*)aligned_alloc(L1_DCACHE_LINESIZE, sizeof(ARR_TYPE) * K*N_RAW);
    ARR_TYPE *C_raw = (ARR_TYPE*)aligned_alloc(L1_DCACHE_LINESIZE, sizeof(ARR_TYPE) * M_RAW*N_RAW);
    if (!A_raw || !B_raw || !C_raw) exit(EXIT_FAILURE);
    if (is_aligned(A_raw, L1_DCACHE_LINESIZE)) printf("A aligned.\n");
    if (is_aligned(B_raw, L1_DCACHE_LINESIZE)) printf("B aligned.\n");
    if (is_aligned(C_raw, L1_DCACHE_LINESIZE)) printf("C aligned.\n");

    fill(A_raw, M_RAW, K);
    fill(B_raw, K, N_RAW);

    size_t M_pad, N_pad;
    ARR_TYPE *A_pad = pad_rows_rm(A_raw, M_RAW, K, MR, &M_pad);
    ARR_TYPE *B_pad = pad_cols_rm(B_raw, K, N_RAW, NR, &N_pad);
    ARR_TYPE *C_pad = NULL;

    ARR_TYPE *A = A_pad ? A_pad : A_raw;
    ARR_TYPE *B = B_pad ? B_pad : B_raw;
    if (A_pad || B_pad) {
        size_t M_c = A_pad ? M_pad : M_RAW;
        size_t N_c = B_pad ? N_pad : N_RAW;
        C_pad = (ARR_TYPE*)calloc(M_c * N_c, sizeof(ARR_TYPE));
        if (!C_pad) exit(EXIT_FAILURE);
    }
    ARR_TYPE *C = C_pad ? C_pad : C_raw;

    size_t m = A_pad ? M_pad : M_RAW;
    size_t n = B_pad ? N_pad : N_RAW;

    /* Cache warmup */
    for (size_t w = 0; w < WARMUPS; ++w) {
        memset(C, 0, m * n * sizeof(ARR_TYPE));
        mm(A, m, K, B, n, C);
    }

#ifdef CHECK
    ARR_TYPE *_C = (ARR_TYPE*)aligned_alloc(L1_DCACHE_LINESIZE, sizeof(ARR_TYPE) * m * n);
    if (!_C) exit(EXIT_FAILURE);
    memset(_C, 0, m * n * sizeof(ARR_TYPE));
    baseline(A, m, K, B, n, _C);
    memset(C, 0, m * n * sizeof(ARR_TYPE));
    mm(A, m, K, B, n, C);
    check(C, _C, m, n);
#endif

    /* Evaluation metrics */
    const int precision = 2;
    double flops = 2.0 * M_RAW * N_RAW * K;
    double gflops = 0.0;

    /* matmul NITER times. */
    double best_time = 1000.0;
    printf("GFLOPS:\t\tTIME:\n");
    for (size_t i = 0; i < NITER; ++i) {
        memset(C, 0, m * n * sizeof(ARR_TYPE));

        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        mm(A, m, K, B, n, C);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        double time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)*1e-9;
        if (time < best_time) best_time = time;
        gflops = (flops/time)*1e-9;
        printf("%.*f\t\t%.*fms\n", precision, gflops, precision, time*1e3);
    }
    printf("Best time:          %.*fms\n", precision, best_time*1e3);
    printf("Best time GFLOPS:   %.*f\n", precision, (flops/best_time)*1e-9);

    /* Otherwise clang with -O3 will assume that C is "dead" and delete all the FMA istructions */
    volatile ARR_TYPE sink = C[0];
    (void)sink;

    free(A_raw); free(B_raw); free(C_raw);
    free(A_pad); free(B_pad); free(C_pad);
    exit(EXIT_SUCCESS);
}
