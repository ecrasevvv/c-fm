#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cfm.h"

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

#define CFM_EPS_FLOAT32 1e-10f
#define CFM_EPS_FLOAT64 1e-10

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

static void cfm_die(const char *msg) {
    fprintf(stderr, "cf-m: %s\n", msg);
    exit(EXIT_FAILURE);
}

cfm_string *cfm_string_new(const char *content) {
    size_t l = strlen(content);
    CFM_ASSERT(l > 0);
    cfm_string *s = (cfm_string*)malloc(sizeof(cfm_string));
    if (!s) cfm_die("Out of memory");
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

cfm_tensor *cfm_tensor_new(const char *name, cfm_dtype dtype,
        uint8_t ndims, const uint16_t *shape) {
    cfm_tensor *t = (cfm_tensor*)malloc(sizeof(cfm_tensor));
    if (!t) cfm_die("Out of memory"); 
    t->name = cfm_string_new(name);
    t->dtype = dtype;
    t->ndims = ndims;
    memcpy(t->shape, shape, sizeof(uint16_t) * ndims);
    t->numel = 1;
    for (size_t i = 0; i < t->ndims; ++i) t->numel *= t->shape[i]; 
    cfm_set_tensor_strides(t);
    if (!(t->data = calloc(t->numel, cfm_element_size(t->dtype))))  /* zero-init new cfm_tensor data. */
        cfm_die("Out of memory");
    return t;
}

cfm_tensor *cfm_tensor_from(const char *name, cfm_dtype dtype,
        uint8_t ndims, const uint16_t *shape, void *data) {
    cfm_tensor *t = (cfm_tensor*)malloc(sizeof(cfm_tensor));
    if (!t) cfm_die("Out of memory");
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

/* This function returns the position for a flat idx whitin a specific dimension dim. 
 * Maps a flat idx to a N-dimensional position. */
CFMDEF int cfm_tensor_get_position_whitin_dimension(uint64_t idx, int dim, 
        const uint16_t strides[], const uint16_t shape[]) {
    return (idx/strides[dim]) % shape[dim];
}

cfm_tensor *cfm_tensor_cat(const char *name, const cfm_tensor **tensors, 
        int ntensors, uint8_t cat_dim) {
    if (ntensors <= 1) cfm_die("cfm_tensor_cat not enough tensors to cat.");

    /* Promote a CFM_FLOAT32 to CFM_FLOAT64 will only add a bunch of useless zeros. */
    int i = ntensors;
    while (--i>0 && tensors[i]->dtype == tensors[0]->dtype);
    if (i != 0) cfm_die("cfm_tensor_cat cannot cat tensors with different cfm_dtype.");

    /* Tensors must have same number of dimensions. */
    i = ntensors;
    while (--i>0 && tensors[i]->ndims == tensors[0]->ndims);
    if (i != 0) cfm_die("cfm_tensor_cat cannot cat tensors with different ndims.");

    /* Tensor must have the same shape (except in the concatening dimension cat_dim). */
    uint8_t dms = tensors[0]->ndims;
    for (int t = 1; t < ntensors; ++t) {
        for (uint8_t d = 0; d < dms; ++d) {
            if (d == cat_dim) continue;
            if (tensors[0]->shape[d] != tensors[t]->shape[d])
                cfm_die("cfm_tensor_cat shape mismatch.");
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
                uint64_t rem = i;
                for (uint8_t d = 0; d < dms; d++) {
                    coords[d] = rem / t->strides[d];
                    rem %= t->strides[d];
                }

                int src = 0;
                while (coords[cat_dim] >= boundaries[src+1]) src++;

                coords[cat_dim] -= boundaries[src];

                uint64_t src_idx = 0;
                for (uint8_t d = 0; d < dms; d++)
                    src_idx += coords[d] * tensors[src]->strides[d];

                out[i] = ((float*)tensors[src]->data)[src_idx];
            }
            break;
        }
        case CFM_FLOAT64: {
            double *out = t->data;
            for (uint64_t i = 0; i < t->numel; i++) {
                uint64_t rem = i;
                for (uint8_t d = 0; d < dms; d++) {
                    coords[d] = rem / t->strides[d];
                    rem %= t->strides[d];
                }

                int src = 0;
                while (coords[cat_dim] >= boundaries[src+1]) src++;

                coords[cat_dim] -= boundaries[src];

                uint64_t src_idx = 0;
                for (uint8_t d = 0; d < dms; d++)
                    src_idx += coords[d] * tensors[src]->strides[d];

                out[i] = ((double*)tensors[src]->data)[src_idx];
            }
            break;
        }
    }

    return t;
}

cfm_tensor *cfm_tensor_get_last(const cfm_tensor *t) {
    CFM_ASSERT(t->ndims > 0);
    uint16_t shape[t->ndims-1];
    for (uint8_t i = 0; i < t->ndims-1; ++i) shape[i] = t->shape[i+1];
    cfm_tensor *tt = cfm_tensor_new(t->name->content, t->dtype, t->ndims-1, shape);
    if (!tt) cfm_die("Out of memory");
    int offset = (t->shape[0]-1) * t->strides[0];
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
    for (uint64_t i = 0; i < t->numel; i++) {
        int wrap = 0;
        for (int d = (int)t->ndims - 1; d >= 0; d--) {
            if (cfm_tensor_get_position_whitin_dimension(i, d, t->strides, t->shape) == 0) wrap++;
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

/* This function checks if two tensors are broadcastable. 
 * For now only tensors with exact same shape are considered as broadcastable, 
 * in reality there are other cases. https://docs.pytorch.org/docs/2.13/notes/broadcasting.html#broadcasting-semantics 
 * todo: add them later on. */

// broadcastable requisistes
// starting at trailing dimension: last dim shape[-1], TWO dimensions are compatibles if:
//  - they are equals
//  - one of them is 1
//
//  - one of them does not exists
static bool cfm_tensors_broadcastable(const cfm_tensor *u, const cfm_tensor *v) {
    for (uint8_t d = u->ndims; d > 0; d--)
        if (u->shape[d] != v->shape[d]) return false;
    return true;
}

cfm_tensor *cfm_tensor_add(const char *name, const cfm_tensor *u, const cfm_tensor *v) {
    if (u->dtype != v->dtype) cfm_die("cfm_tensor_add cannot add tensors with differents dtype.");
    if (!cfm_tensors_broadcastable(u, v)) cfm_die("cfm_tensor_add the two tensors are not broadcastable.");
    // if two tensors results broadcastable the new tensor t shape is calculatead as described 
    // in https://docs.pytorch.org/docs/2.13/notes/broadcasting.html#broadcasting-semantics
    // fow now t->shape = u->shape
    // todo: correct shape calculation
    cfm_tensor *t = cfm_tensor_new(name, u->dtype, u->ndims, u->shape);
    switch (t->dtype) {
        case CFM_FLOAT32:
            float *f_t_data = t->data;
            float *f_u_data = u->data;
            float *f_v_data = v->data;
            for (uint64_t i = 0; i < t->numel; ++i) {
                f_t_data[i] = f_u_data[i] + f_v_data[i];
            }
            break;
        case CFM_FLOAT64:
            double *d_t_data = t->data;
            double *d_u_data = u->data;
            double *d_v_data = v->data;
            for (uint64_t i = 0; i < t->numel; ++i) {
                d_t_data[i] = d_u_data[i] + d_v_data[i];
            }
            break;
    }
    return t;
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
