/* This file is deliberately vertical: it owns cfm_tensor logic and ops, 
 * GGUF loading and FM inference. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cfm.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifndef CFM_FREE
#define CFM_FREE free
#endif  /* CFM_FREE */

#ifndef CFM_ASSERT
#include <assert.h>
#define CFM_ASSERT assert
#endif  /* CFM_ASSERT */

#ifndef CFMDEF
#define CFMDEF static inline
#endif /* CFMDEF */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif 

#ifndef _2_M_PI
#define _2_M_PI (M_PI*2.0)
#endif

#define CFM_EPS_FLOAT32 (1e-10f)
#define CFM_EPS_FLOAT64 ( 1e-10)

# ifndef CFM_MIN
# define CFM_MIN(x,y) ((x)<(y)?(x):(y))
# endif /* CFM_MIN */
# ifndef CFM_MAX
# define CFM_MAX(x,y) ((x)>(y)?(x):(y))
# endif /* CFM_MAX */

/* Debug macro used to print a vector. */
#define CFM_D_VEC_PRINT(V, s)                                   \
    do {                                                        \
        CFM_ASSERT(V != NULL);                                  \
        putchar('[');                                           \
        for(int i = 0; i < s; ++i) {                            \
            if (i == s-1) printf("%d", (int)V[i]);              \
            else printf("%d, ", (int)V[i]);                     \
        }                                                       \
        putchar(']');                                           \
        putchar('\n');                                          \
    } while(0)

static void cfm_die(const int line, const char *msg) {
    fprintf(stderr, "\033[33m[cf-m]\033[0m %s:%d: %s\n", __FILE__, line, msg);
    exit(EXIT_FAILURE);
}

cfm_string *cfm_string_new(const char *content) {
    size_t l = strlen(content);
    CFM_ASSERT(l > 0);
    cfm_string *s = (cfm_string*)malloc(sizeof(cfm_string));
    if (!s) cfm_die(__LINE__, "Out of memory");
    s->content = content;
    s->len = l;
    return s;
}

void cfm_string_print(const cfm_string *str) {
    printf("%s\n", str->content);
}

void cfm_string_free(cfm_string *str) {
    CFM_FREE(str);
}

void cfm_tensor_free(cfm_tensor *t) {
    cfm_string_free(t->name);
    CFM_FREE(t->data);
    CFM_FREE(t);
}

/* This function returns the size of the single element present in t->data given its cfm_dtype. */
static size_t cfm_element_size(cfm_dtype dtype) {
    return (dtype == CFM_FLOAT32) ? sizeof(float) : sizeof(double);
}

/* This function returns the stride (https://numpy.org/doc/2.1/reference/arrays.ndarray.html#arrays-ndarray
 * in the "Internal memory layout of an ndarray" section) which corresponds to the bytes to step in each dimension
 * when traversing an array. */
static void cfm_set_tensor_strides(cfm_tensor *t) {
    /* strides[k] = shape[k+1] * shape[k+2] * ... * shape[ndims] */
    for (size_t k = 0; k < t->ndims; ++k) {
        t->strides[k] = 1;
        for (size_t i = k+1; i < t->ndims; ++i) {
            t->strides[k] *= t->shape[i];
        }
    }
}

/* This function creates a new cfm_tensor with zero-inited data. */
CFMDEF cfm_tensor *cfm_tensor_new(const char *name, cfm_dtype dtype,
        uint8_t ndims, const uint16_t *shape) {
    cfm_tensor *t = (cfm_tensor*)malloc(sizeof(cfm_tensor));
    if (!t) cfm_die(__LINE__, "Out of memory"); 
    t->name = cfm_string_new(name);
    t->dtype = dtype;
    t->ndims = ndims;
    memcpy(t->shape, shape, sizeof(uint16_t) * ndims);
    t->numel = 1;
    for (size_t i = 0; i < t->ndims; ++i) t->numel *= t->shape[i]; 
    cfm_set_tensor_strides(t);
    if (!(t->data = calloc(t->numel, cfm_element_size(t->dtype))))
        cfm_die(__LINE__, "Out of memory");
    return t;
}

cfm_tensor *cfm_tensor_from(const char *name, cfm_dtype dtype,
        uint8_t ndims, const uint16_t *shape, void *data) {
    cfm_tensor *t = (cfm_tensor*)malloc(sizeof(cfm_tensor));
    if (!t) cfm_die(__LINE__, "Out of memory");
    t->name = cfm_string_new(name);
    t->dtype = dtype;
    t->ndims = ndims;
    memcpy(t->shape, shape, sizeof(uint16_t) * ndims);
    t->data = data;
    t->numel = 1;
    for (size_t i = 0; i < t->ndims; ++i) t->numel *= t->shape[i]; 
    cfm_set_tensor_strides(t);
    return t;
}

cfm_tensor *cfm_tensor_rand(const char *name, cfm_dtype dtype,
        uint8_t ndims, const uint16_t *shape) {
    cfm_tensor *t = cfm_tensor_new(name, dtype, ndims, shape);
    CFM_ASSERT(t != NULL);

    switch (dtype) {
        case CFM_FLOAT32:
            float *f_data = t->data;
            for (size_t i = 0; i < t->numel; ++i) {
                f_data[i] = (float)rand()/(float)RAND_MAX;
            }
            break;
        case CFM_FLOAT64:
            double *d_data = t->data;
            for (size_t i = 0; i < t->numel; ++i) {
                d_data[i] = (double)rand()/(double)RAND_MAX;
            }
            break;
    }
    return t;
}

cfm_tensor *cfm_tensor_randn(const char *name, cfm_dtype dtype, 
        uint8_t ndims, const uint16_t *shape) {
    cfm_tensor *t = cfm_tensor_new(name, dtype, ndims, shape);
    switch (dtype) {
        case CFM_FLOAT32:
            float *f_data = t->data;
            for (size_t i = 0; i < t->numel; ++i) {
                float u1 = (float)rand()/((float)RAND_MAX+CFM_EPS_FLOAT32);
                float u2 = (float)rand()/(float)RAND_MAX;
                f_data[i] = sqrtf(-2.f*log((float)u1)) * cosf(_2_M_PI*u2);
            }
            break;
        case CFM_FLOAT64:
            double *d_data = t->data;
            for (size_t i = 0; i < t->numel; ++i) {
                double u1 = (double)rand()/((double)RAND_MAX+CFM_EPS_FLOAT64);
                double u2 = (double)rand()/(double)RAND_MAX;
                d_data[i] = sqrt(-2.0*log(u1)) * cos(_2_M_PI*u2);
            }
            break;
    }
    return t;
}

cfm_tensor *cfm_tensor_linspace_float32(const char *name, cfm_dtype dtype,
        float start, float end, float step_size) {
    const int nsteps = (start > end) ? (start-end)/step_size+1 : (end-start)/step_size+1;
    const uint8_t ndims = 1;
    uint16_t shape[1] = {nsteps};
    cfm_tensor *t = cfm_tensor_new(name, dtype, ndims, shape);
    float *data = t->data;
    if (start > end) {
        for (int i = 0; i < nsteps; ++i) {
            data[i] = start-i*step_size;
        }
    } else {
        for (int i = 0; i < nsteps; ++i) {
            data[i] = start+i*step_size;
        }
    }
    return t;
}

cfm_tensor *cfm_tensor_linspace_float64(const char *name, cfm_dtype dtype,
        double start, double end, double step_size) {
    const int nsteps = (start > end) ? (start-end)/step_size+1 : (end-start)/step_size+1;
    const uint8_t ndims = 1;
    uint16_t shape[1] = {nsteps};
    cfm_tensor *t = cfm_tensor_new(name, dtype, ndims, shape);
    double *data = t->data;
    if (start > end) {
        for (int i = 0; i < nsteps; ++i) {
            data[i] = start-i*step_size;
        }
    } else {
        for (int i = 0; i < nsteps; ++i) {
            data[i] = start+i*step_size;
        }
    }
    return t;
}

cfm_tensor *cfm_tensor_full_float32(const char *name, cfm_dtype dtype,
        uint8_t ndims, const uint16_t *shape, float fill_value) {
    cfm_tensor *t = cfm_tensor_new(name, dtype, ndims, shape);
    float *data = t->data;
    for (size_t i = 0; i < t->numel; ++i) {
        data[i] = fill_value;
    }
    return t;
}

cfm_tensor *cfm_tensor_full_float64(const char *name, cfm_dtype dtype,
        uint8_t ndims, const uint16_t *shape, double fill_value) {
    cfm_tensor *t = cfm_tensor_new(name, dtype, ndims, shape);
    double *data = t->data;
    for (size_t i = 0; i < t->numel; ++i) {
        data[i] = fill_value;
    }
    return t;
}

cfm_tensor *cfm_tensor_zeros(const char *name, cfm_dtype dtype,
        uint8_t ndims, const uint16_t *shape) {
    return (dtype == CFM_FLOAT32) 
        ? cfm_tensor_full_float32(name, dtype, ndims, shape, 0.f)
        : cfm_tensor_full_float64(name, dtype, ndims, shape, 0.0);
}

cfm_tensor *cfm_tensor_ones(const char *name, cfm_dtype dtype,
        uint8_t ndims, const uint16_t *shape) {
    return (dtype == CFM_FLOAT32) 
        ? cfm_tensor_full_float32(name, dtype, ndims, shape, 1.f)
        : cfm_tensor_full_float64(name, dtype, ndims, shape, 1.0);
}

/* This function returns the element in the position specified by idx from a given cfm_tensor t. */
CFMDEF double cfm_tensor_get_element(const cfm_tensor *t, uint64_t idx) {
    return (t->dtype == CFM_FLOAT32) ? ((float*)t->data)[idx] : ((double*)t->data)[idx];
}

/* This function converts a flat index i into an n-dimensional coordinates. */
CFMDEF void cfm_tensor_unravel_index(uint16_t *coords, uint64_t rem,
        const uint64_t *strides, const uint8_t ndims) {
    for (uint8_t d = 0; d < ndims; ++d) {
        coords[d] = rem / strides[d];
        rem %= strides[d];
    }
}

/* This function does the opposite of cfm_tensor_unravel_index, given an n-dimensional
 * set of coordinates it returns the corresponding flat index i. */
CFMDEF uint64_t cfm_tensor_ravel_index(const uint16_t *coords, const uint64_t *strides, 
        const uint8_t ndims) {
    uint64_t i = 0;
    for (uint8_t d = 0; d < ndims; ++d) {
        i += coords[d] * strides[d];
    }
    return i;
}

cfm_tensor *cfm_tensor_cat(const char *name, const cfm_tensor **tensors, 
        int ntensors, uint8_t cat_dim) {
    if (ntensors <= 1) cfm_die(__LINE__, "cfm_tensor_cat not enough tensors to cat.");

    /* Promote a CFM_FLOAT32 to CFM_FLOAT64 will only add a bunch of useless zeros. */
    int i = ntensors;
    while (--i>0 && tensors[i]->dtype == tensors[0]->dtype);
    if (i != 0) cfm_die(__LINE__, "cfm_tensor_cat cannot cat tensors with different cfm_dtype.");

    /* Tensors must have same number of dimensions. */
    i = ntensors;
    while (--i>0 && tensors[i]->ndims == tensors[0]->ndims);
    if (i != 0) cfm_die(__LINE__, "cfm_tensor_cat cannot cat tensors with different ndims.");

    /* Tensor must have the same shape (except in the concatening dimension cat_dim). */
    uint8_t dms = tensors[0]->ndims;
    for (int t = 1; t < ntensors; ++t) {
        for (uint8_t d = 0; d < dms; ++d) {
            if (d == cat_dim) continue;
            if (tensors[0]->shape[d] != tensors[t]->shape[d])
                cfm_die(__LINE__, "cfm_tensor_cat shape mismatch.");
        }
    }

    uint16_t cat_t_shape[dms];
    for (uint8_t d = 0; d < dms; ++d) {
        if (d == cat_dim) {
            cat_t_shape[d] = 0;
            for (int t = 0; t < ntensors; ++t) {
                cat_t_shape[d] += tensors[t]->shape[d];
            }
        } else {
            cat_t_shape[d] = tensors[0]->shape[d];
        }
    }

    cfm_tensor *t = cfm_tensor_new(name, tensors[0]->dtype, dms, cat_t_shape);
    CFM_ASSERT(t->shape[cat_dim] <= CFM_MAX_DIMS);

    /* Start and end of each source tensor. */
    uint16_t boundaries[ntensors+1];
    boundaries[0] = 0;
    for (int t = 0; t < ntensors; t++)
        boundaries[t+1] = boundaries[t] + tensors[t]->shape[cat_dim];

    /* Map a flat index i to the corresponding coords in a dim d. 
     * Find the source tensor index src given cooords. 
     * Calculate the src_idx of the element that needs to be inserted into out at i. */
    uint16_t coords[dms];
    switch (t->dtype) {
        case CFM_FLOAT32: {
            float *out = t->data;
            for (uint64_t i = 0; i < t->numel; i++) {
                cfm_tensor_unravel_index(coords, i, t->strides, t->ndims);

                int src = 0;
                while (coords[cat_dim] >= boundaries[src+1]) src++;

                coords[cat_dim] -= boundaries[src];

                uint64_t src_idx = cfm_tensor_ravel_index(coords, tensors[src]->strides, dms);

                out[i] = ((float*)tensors[src]->data)[src_idx];
            }
            break;
        }
        case CFM_FLOAT64: {
            double *out = t->data;
            for (uint64_t i = 0; i < t->numel; i++) {
                cfm_tensor_unravel_index(coords, i, t->strides, t->ndims);

                int src = 0;
                while (coords[cat_dim] >= boundaries[src+1]) src++;

                coords[cat_dim] -= boundaries[src];

                uint64_t src_idx = cfm_tensor_ravel_index(coords, tensors[src]->strides, dms);

                out[i] = ((double*)tensors[src]->data)[src_idx];
            }
            break;
        }
    }

    return t;
}

cfm_tensor *cfm_tensor_expand(const cfm_tensor *u, const uint8_t exp_ndims,
        const uint16_t *exp_shape) {
    int du = u->ndims-1;
    for (int d = exp_ndims-1; d >= 0; d--) {
        /* Treat the missing dimension as 1 in order to pass the second if. */
        uint16_t us = (du >= 0) ? u->shape[du] : 1;
        if (us != 1 && us != exp_shape[d])
            cfm_die(__LINE__, "cfm_tensor_expand cannot expand non-singleton dimension.");
        du--;
    }
    cfm_tensor *t = cfm_tensor_new(u->name->content, u->dtype, exp_ndims, exp_shape);

    /* Copy data to the new expanded tensor. */
    uint16_t coords[t->ndims];
    uint16_t u_coords[u->ndims];
    uint64_t offset = exp_ndims - u->ndims;
    switch (t->dtype) {
        case CFM_FLOAT32:
            float *f_data = t->data;
            for (uint64_t i = 0; i < t->numel; ++i) {
                cfm_tensor_unravel_index(coords, i, t->strides, t->ndims);
                for (uint8_t du = 0; du < u->ndims; ++du) {
                    uint64_t d = du + offset;
                    u_coords[du] = (u->shape[du] == 1) ? 0 : coords[d];
                }
                uint64_t src_idx = cfm_tensor_ravel_index(u_coords, u->strides, u->ndims);
                f_data[i] = ((float*)u->data)[src_idx];
            }
            break;
        case CFM_FLOAT64:
            double *d_data = t->data;
            for (uint64_t i = 0; i < t->numel; ++i) {
                cfm_tensor_unravel_index(coords, i, t->strides, t->ndims);
                for (uint8_t du = 0; du < u->ndims; ++du) {
                    uint64_t d = du + offset;
                    u_coords[du] = (u->shape[du] == 1) ? 0 : coords[d];
                }
                uint64_t src_idx = cfm_tensor_ravel_index(u_coords, u->strides, u->ndims);
                d_data[i] = ((double*)u->data)[src_idx];
            }
            break;
    }
    return t;
}

cfm_tensor *cfm_tensor_get_last(const cfm_tensor *t) {
    CFM_ASSERT(t->ndims > 0);
    uint16_t shape[t->ndims-1];
    for (uint8_t i = 0; i < t->ndims-1; ++i) shape[i] = t->shape[i+1];
    cfm_tensor *tt = cfm_tensor_new(t->name->content, t->dtype, t->ndims-1, shape);
    if (!tt) cfm_die(__LINE__, "Out of memory");
    uint64_t offset = (t->shape[0]-1) * t->strides[0];
    switch (tt->dtype) {
        case CFM_FLOAT32:
            float *tt_f_data = tt->data;
            float *t_f_data = t->data;
            for (uint64_t i = 0; i < tt->numel; ++i) {
                tt_f_data[i] = t_f_data[i+offset];
            }
            break;
        case CFM_FLOAT64:
            double *tt_d_data = tt->data;
            double *t_d_data = t->data;
            for (uint64_t i = 0; i < tt->numel; ++i) {
                tt_d_data[i] = t_d_data[i+offset];
            }
            break;
    }
    return tt;
}

void cfm_tensor_print_raw(const cfm_tensor *t, cfm_print_mode pm, int precision) {
    CFM_ASSERT(precision > 0 && precision <= 6);
#ifdef DEBUG
    printf("TENSOR:\n");
    printf("        name:       %s\n", t->name->content);
    printf("        ndims:      %u\n", t->ndims);
    printf("        shape:      "); CFM_D_VEC_PRINT(t->shape, t->ndims);
    printf("        strides:    "); CFM_D_VEC_PRINT(t->strides, t->ndims);
    printf("        numel:      %zu\n", t->numel);
    printf("        dtype:      %s\n", (t->dtype == CFM_FLOAT32) ? "CFM_FLOAT32" : "CFM_FLOAT64");
    printf("        elements:   ");
#else
    printf("%s(", t->name->content);
#endif
    switch (pm) {
        case CFM_H_PRINT:
            for (size_t i = 0; i < t->numel; ++i) {
                if (i == t->numel-1) printf("%.*f)\n", precision, cfm_tensor_get_element(t, i));
                else printf("%.*f, ", precision, cfm_tensor_get_element(t, i));
            }
            break;
        case CFM_V_PRINT:
#ifdef DEBUG
            int indent = 20;
#else
            int indent = (int)t->name->len;
#endif
            putchar('\n');
            for (size_t i = 0; i < t->numel; ++i) {
                if (i == t->numel-1) printf("%*s%.*f)\n", indent, "", precision, cfm_tensor_get_element(t, i));
                else printf("%*s%.*f,\n", indent, "", precision, cfm_tensor_get_element(t, i));
            }
            break;
    }
}

void cfm_tensor_print(const cfm_tensor *t, int precision) {
    CFM_ASSERT(precision > 0 && precision <= 6);
#ifdef DEBUG
    printf("TENSOR:\n");
    printf("        name:       %s\n", t->name->content);
    printf("        ndims:      %u\n", t->ndims);
    printf("        shape:      "); CFM_D_VEC_PRINT(t->shape, t->ndims);
    printf("        strides:    "); CFM_D_VEC_PRINT(t->strides, t->ndims);
    printf("        numel:      %zu\n", t->numel);
    printf("        dtype:      %s\n", (t->dtype == CFM_FLOAT32) ? "CFM_FLOAT32" : "CFM_FLOAT64");
    printf("        elements:   ");
#else
    /* Scalar. */
    if (t->ndims == 0) {
        printf("%s(%.*f)\n", t->name->content, precision, cfm_tensor_get_element(t, 0));
        return;
    }
    printf("%s(", t->name->content);
#endif
    for (uint8_t d = 0; d < t->ndims; d++) putchar('[');
    uint16_t coords[t->ndims];
    for (uint64_t i = 0; i < t->numel; i++) {
        int wrap = 0;
        cfm_tensor_unravel_index(coords, i, t->strides, t->ndims);
        for (int d = (int)t->ndims - 1; d >= 0; d--) {
            if (coords[d] == 0) wrap++;
            else break;
        }

        if (i > 0) {
            if (wrap > 0) {
                for (int w = 0; w < wrap; w++) putchar(']');
#ifdef DEBUG
                int indent = 20 + t->ndims - wrap;
#else
                int indent = t->name->len + t->ndims - wrap + 1;
#endif
                printf(",\n%*s", indent, "");
                for (int w = 0; w < wrap; w++) putchar('[');
            } else {
                printf(", ");
            }
        }

        printf("%.*f", precision, cfm_tensor_get_element(t, i));
   }
    for (uint8_t d = 0; d < t->ndims; d++) putchar(']');
    printf(")\n");
}

/* This function checks if two tensors are broadcastable and computes the resulting shape.
 * Broadcasting rules: https://docs.pytorch.org/docs/2.13/notes/broadcasting.html#broadcasting-semantics
 * The resulting cfm_tensor will have the same number of dimensions as the input with the
 * greatest number of dimensions. The size of each dimension is the element-wise maximum
 * among the corresponding input dimensions. Dimensions with size 1 are stretched.
 * Returns false if not broadcastable. On success, fills *out_ndims and out_shape. */
static bool cfm_tensor_broadcast(const cfm_tensor *u, const cfm_tensor *v,
                          uint8_t *out_ndims, uint16_t *out_shape) {
    int du = u->ndims-1;
    int dv = v->ndims-1;
    *out_ndims = CFM_MAX(u->ndims, v->ndims);
    int d = *out_ndims-1;
    while (du >= 0 || dv >= 0) {
        uint16_t su = (du >= 0) ? u->shape[du] : 1;
        uint16_t sv = (dv >= 0) ? v->shape[dv] : 1;
        if (su != sv && su != 1 && sv != 1) return false;
        out_shape[d] = CFM_MAX(su, sv);
        du--; dv--; d--;
    }
    return true;
}

cfm_tensor *cfm_tensor_add(const char *name, const cfm_tensor *u, const cfm_tensor *v) {
    if (u->dtype != v->dtype) cfm_die(__LINE__, "cfm_tensor_add cannot add tensors with differents dtype.");
    uint8_t ndims;
    uint16_t shape[CFM_MAX_DIMS] = {0};
    if (!cfm_tensor_broadcast(u, v, &ndims, shape))
        cfm_die(__LINE__, "cfm_tensor_add the two tensors are not broadcastable.");
    cfm_tensor *t = cfm_tensor_new(name, u->dtype, ndims, shape);
    cfm_tensor *u_exp = cfm_tensor_expand(u, ndims, shape); 
    cfm_tensor *v_exp = cfm_tensor_expand(v, ndims, shape); 
    
    switch (t->dtype) {
        case CFM_FLOAT32:
            float *t_f_data = t->data;
            float *u_f_data = u_exp->data;
            float *v_f_data = v_exp->data;
            for (uint64_t i = 0; i < t->numel; ++i) t_f_data[i] = u_f_data[i] + v_f_data[i];
            break;
        case CFM_FLOAT64:
            double *t_d_data = t->data;
            double *u_d_data = u_exp->data;
            double *v_d_data = v_exp->data;
            for (uint64_t i = 0; i < t->numel; ++i) t_d_data[i] = u_d_data[i] + v_d_data[i];
            break;
    }
    cfm_tensor_free(u_exp);
    cfm_tensor_free(v_exp);
    return t;
}

cfm_tensor *cfm_tensor_mul(const char *name, const cfm_tensor *u, const cfm_tensor *v) {
    if (u->dtype != v->dtype) cfm_die(__LINE__, "cfm_tensor_mul cannot mul tensors with differents dtype.");
    uint8_t ndims;
    uint16_t shape[CFM_MAX_DIMS] = {0};
    if (!cfm_tensor_broadcast(u, v, &ndims, shape))
        cfm_die(__LINE__, "cfm_tensor_mul the two tensors are not broadcastable.");
    cfm_tensor *t = cfm_tensor_new(name, u->dtype, ndims, shape);
    cfm_tensor *u_exp = cfm_tensor_expand(u, ndims, shape); 
    cfm_tensor *v_exp = cfm_tensor_expand(v, ndims, shape); 
    
    switch (t->dtype) {
        case CFM_FLOAT32:
            float *t_f_data = t->data;
            float *u_f_data = u_exp->data;
            float *v_f_data = v_exp->data;
            for (uint64_t i = 0; i < t->numel; ++i) t_f_data[i] = u_f_data[i] * v_f_data[i];
            break;
        case CFM_FLOAT64:
            double *t_d_data = t->data;
            double *u_d_data = u_exp->data;
            double *v_d_data = v_exp->data;
            for (uint64_t i = 0; i < t->numel; ++i) t_d_data[i] = u_d_data[i] * v_d_data[i];
            break;
    }
    cfm_tensor_free(u_exp);
    cfm_tensor_free(v_exp);
    return t;
}

cfm_tensor *cfm_tensor_dot(const char *name, const cfm_tensor *u,
        const cfm_tensor *v) {
    /* The check on the dtype may seems redundant if the caller of the function is cfm_tensor_matmul but is
     * needed if the caller is NOT cfm_tensor_matmul. The same goes for the check on the ndims. */
    if (u->dtype != v->dtype) cfm_die(__LINE__, "cfm_tensor_dot cannot compute the dot product between tensors with differents dtype.");
    if (u->ndims != 1 || v->ndims != 1) cfm_die(__LINE__, "cfm_tensor_dot both cfm_tensor needs to be 1D to compute the dot product.");
    if (u->numel != v->numel) cfm_die(__LINE__, "cfm_tensor_dot cannot compute the dot product between tensors with differents numel.");

    cfm_tensor *t = cfm_tensor_new(name, u->dtype, 1, (uint16_t[]){1});
    switch (t->dtype) {
        case CFM_FLOAT32:
            float *f_t_data = t->data;
            float *f_u_data = u->data;
            float *f_v_data = v->data;
            for (size_t i = 0; i < u->numel; ++i) f_t_data[0] += f_u_data[i] * f_v_data[i];
            break;
        case CFM_FLOAT64:
            double *d_t_data = t->data;
            double *d_u_data = u->data;
            double *d_v_data = v->data;
            for (size_t i = 0; i < u->numel; ++i) d_t_data[0] += d_u_data[i] * d_v_data[i];
            break;
    }
    return t;
}

/* Row-major indexing */
#define IDX(row, cols, col) (((row)*cols)+(col))

#ifdef __AVX2__
/* This is a fast matmul implementation done with AVX2 instructions that 
 * achieves the same performance as numpy in terms of GFLOPS (see ./mm/mm.c for more details).
 * Note that this is optimized for my CPU since the project is meant to be an educational project for me,
 * this means that the results and behavior on other CPUs are unknown to me.
 * Feel free to read ./mm/mm.c to explore in details the fast matmul implementation, try other micro-kernels
 * and run the benchmark on your CPU. */
#define kernel_16x6(A_start, B_start, C_start)          \
    _Generic((A_start),                                 \
            float:  kernel_16x6f,                       \
            double: kernel_16x6d,                       \
            )(A_start, B_start, C_start)
#if 0 // necessary until porting is finished
__attribute__((noinline))
static void kernel_16x6f(float *A_start, float *B_start, float *__restrict__ C_start) {
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

/* Since one double is 8bytes a single YMM can store 4 doubles. The logic of the micro-kernel
 * will be completely different.
 * Not planned as todo for now. */
__attribute__((noinline))
static void kernel_16x6d(double *A_start, double *B_start, double *__restrict__ C_start) {
    cfm_die(__LINE__, "kernel_16x6d not implemented.");
}

#define mm(A, B, C)         \
    _Generic((A),           \
            float:  mm_f,   \
            double: mm_d,   \
            )(A, B, C)

static void mm_f(float *A, float *B, float *__restrict__ C) {
    // A[M][K], B[K][N], C[M][N]
#ifdef _OPENMP
#define NTHREADS 12
    #pragma omp parallel for collapse(2) num_threads(NTHREADS)
#endif  /* _OPENMP */
    for (size_t i = 0; i < M; i+=MR) {
        for (size_t j = 0; j < N; j+=NR) {
            kernel_16x6(&A[i], &B[j*K], &C[idx(j,M,i)]);
        }
    }
}

static void mm_d(double *A, double *B, double *__restrict__ C) {
    cfm_die(__LINE__, "mm_d not implemented.");
}

#endif // if 0
#else /* no __AVX2__ */
/* If your CPU does not support AVX2 instructions then the multiplication 
 * between the two matrices will happen in the classic way.
 * Note: this can be optimized. */
#define mm_base(A, B, C)        \
    _Generic((A),               \
            float: mm_base_f,   \
            double: mm_base_d,  \
            )(A, B, C)

static void mm_base_f(float *__restrict__ C, uint16_t m, uint16_t n,
        const float *A, uint16_t k,
        const float *B) {
    // A[M][K], B[K][N], C[M][N]
    // u[M][K], v[K][N], C[M][N]
#ifdef _OPENMP
#define NTHREADS 12
    #pragma omp parallel for collapse(2) num_threads(NTHREADS)
#endif  /* _OPENMP */
    for (uint16_t i = 0; i < m; ++i) {
        for (uint16_t j = 0; j < n; ++j) {
            for (uint16_t p = 0; p < k; ++p) {
                C[IDX(i,n,j)] += A[IDX(i,k,p)] * B[IDX(p,n,j)];
            }
        }
    }
}

/* re-add static, -Wunused-function suppression */
void mm_base_d(double *__restrict__ C, uint16_t m, uint16_t n,
        const double *A, uint16_t k,
        const double *B) {
    (void)C; (void)m; (void)n; (void)A; (void)k; (void)B;
    cfm_die(__LINE__, "mm_base_d not implemented.");
}
#endif /* __AVX2__ */

cfm_tensor *cfm_tensor_matmul(const char *name, const cfm_tensor *u,
        const cfm_tensor *v) {
    if (u->dtype == CFM_FLOAT64 || v->dtype == CFM_FLOAT64) 
        cfm_die(__LINE__, "cfm_tensor_matmul can only perform matmul on CFM_FLOAT32 cfm_tensor for now.");

    /* Both 1D cfm_tensor, dot product. */
    if (u->ndims == 1 && v->ndims == 1) return cfm_tensor_dot(name, u, v);

    /* Both 2D cfm_tensor, matrix-matrix product. Note: efficient matmul implemented in mm/mm.c */
    if (u->ndims == 2 && v->ndims == 2) {
        if (u->shape[1] != v->shape[0]) cfm_die(__LINE__, "cfm_tensor_matmul incompatible inner dimensions.");
        uint16_t m = u->shape[0];
        uint16_t n = v->shape[1];
        uint16_t k = v->shape[0];
        uint16_t t_shape[2] = {m, n};
        cfm_tensor *t = cfm_tensor_new(name, u->dtype, 2, t_shape);
#ifdef __AVX2__
        cfm_tensor_free(t);
        cfm_die(__LINE__, "cfm_tensor_matmul AVX2 matmul not supported yet.");
        // mm()
#else
        mm_base_f((float*)t->data, m, n, (float*)u->data, k, (float*)v->data);
#endif /* __AVX2__ */
        return t;
    }

    /* Other cases */
    fprintf(stderr, "cfm_tensor_matmul on the provided u and v not supported yet.");
    return NULL;
}

cfm_tensor *cfm_tensor_exp(const char *name, const cfm_tensor *u) {
    cfm_tensor *t = cfm_tensor_new(name, u->dtype, u->ndims, u->shape);
    switch (t->dtype) {
        case CFM_FLOAT32:
            float *f_t_data = t->data;
            float *f_u_data = u->data;
            for (uint64_t i = 0; i < t->numel; ++i) {
                f_t_data[i] = expf(f_u_data[i]);
            }
            break;
        case CFM_FLOAT64:
            double *d_t_data = t->data;
            double *d_u_data = u->data;
            for (uint64_t i = 0; i < t->numel; ++i) {
                d_t_data[i] = exp(d_u_data[i]);
            }
            break;
    }
    return t;
}
