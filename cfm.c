#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "cfm.h"

#ifndef CFM_FREE
#define CFM_FREE free
#endif  /* CFM_FREE */

#ifndef CFM_ASSERT
#include <assert.h>
#define CFM_ASSERT assert
#endif  /* CFM_ASSERT */

/* Debug macro used to print a vector. */
#define CFM_D_VEC_PRINT(V, s)                                   \
    do {                                                        \
        CFM_ASSERT(V != NULL);                                  \
        for(size_t i = 0; i < s; ++i) printf("%d ", (int)V[i]); \
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
    CFM_FREE(t->data);
    CFM_FREE(t);
}

/* This function returns the size of the single element present in t->data given its cfm_dtype. */
static size_t cfm_element_size(cfm_dtype dtype) {
    return (dtype == CFM_FLOAT32) ? sizeof(float) : sizeof(double);
}

/* This function returns the stride (https://numpy.org/doc/2.1/reference/arrays.ndarray.html#arrays-ndarray
 * in the "Internal memory layout of an ndarray) which corresponds to the bytes to step in each dimension
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

cfm_tensor *cfm_tensor_new(cfm_string name, cfm_dtype dtype,
        uint8_t ndims, uint16_t *shape, bool requires_grad) {
    cfm_tensor *t = (cfm_tensor*)malloc(sizeof(cfm_tensor));
    if (!t) cfm_die("Out of memory"); 
    t->name = name;
    t->dtype = dtype;
    t->ndims = ndims;
    memcpy(t->shape, shape, sizeof(uint16_t) * ndims);
    t->numel = 1;
    for (size_t i = 0; i < t->ndims; ++i) t->numel *= t->shape[i]; 
    cfm_set_tensor_strides(t);
    if (!(t->data = calloc(t->numel, cfm_element_size(t->dtype))))  /* zero-init new cfm_tensor data. */
        cfm_die("Out of memory");
    t->requires_grad = requires_grad;
    return t;
}

cfm_tensor *cfm_tensor_from(cfm_string name, cfm_dtype dtype,
        uint8_t ndims, uint16_t *shape, void *data, bool requires_grad) {
    cfm_tensor *t = (cfm_tensor*)malloc(sizeof(cfm_tensor));
    if (!t) cfm_die("Out of memory"); 
    t->name = name;
    t->dtype = dtype;
    t->ndims = ndims;
    memcpy(t->shape, shape, sizeof(uint16_t) * ndims);
    t->data = data;
    t->numel = 1;
    for (size_t i = 0; i < t->ndims; ++i) t->numel *= t->shape[i]; 
    cfm_set_tensor_strides(t);
    size_t element_size = cfm_element_size(dtype);
    t->requires_grad = requires_grad;
    return t;
}

cfm_tensor *cfm_tensor_rand(cfm_string name, cfm_dtype dtype,
        uint8_t ndims, uint16_t *shape, bool requires_grad) {
    cfm_tensor *t = cfm_tensor_new(name, dtype, ndims, shape, requires_grad);
    CFM_ASSERT(t != NULL);

    if (t->dtype == CFM_FLOAT32) {
        float *d = t->data;
        for (size_t i = 0; i < t->numel; ++i) {
            d[i] = (float)rand()/(float)RAND_MAX;
        }
    } else {
        double *d = t->data;
        for (size_t i = 0; i < t->numel; ++i) {
            d[i] = (double)rand()/(double)RAND_MAX;
        }
    }
    return t;
}

cfm_tensor *cfm_tensor_randn(cfm_string name, cfm_dtype dtype, 
        uint8_t ndims, uint16_t *shape, bool requires_grad) {
    srand(time(NULL));
    cfm_tensor *t = cfm_tensor_new(name, dtype, ndims, shape, requires_grad);
    CFM_ASSERT(t != NULL);
    
    if (t->dtype == CFM_FLOAT32) {
        float *d = t->data;
        for (size_t i = 0; i < t->numel; ++i) {
            float u1 = (float)rand()/(float)RAND_MAX;
            float u2 = (float)rand()/(float)RAND_MAX;
            d[i] = sqrtf(-2.f*log((float)u1)) * cosf(CFM_2_PI*u2);
        }
    } else {
        double *d = t->data;
        for (size_t i = 0; i < t->numel; ++i) {
            double u1 = (double)rand()/(double)RAND_MAX;
            double u2 = (double)rand()/(double)RAND_MAX;
            d[i] = sqrt(-2.f*log(u1)) * cos(CFM_2_PI*u2);
        }
    }
    return t;
}

/* This function returns the element in the position specified by idx from a given cfm_tensor t. */
static double cfm_tensor_get_element(const cfm_tensor *t, uint64_t idx) {
    return (t->dtype == CFM_FLOAT32) ? ((float*)t->data)[idx] : ((double*)t->data)[idx];
}

void cfm_tensor_print_raw(const cfm_tensor *t, cfm_print_mode pm, int precision) {
    printf("%s(", t->name.content);
    switch (pm) {
        case CFM_H_PRINT:
            for (size_t i = 0; i < t->numel; ++i) {
                if (i == t->numel-1) printf("%.*f)\n", precision, cfm_tensor_get_element(t, i));
                else printf("%.*f, ", precision, cfm_tensor_get_element(t, i));
            }
            break;
        case CFM_V_PRINT:
            printf("\n");
            size_t indent = strlen(t->name.content);
            for (size_t i = 0; i < t->numel; ++i) {
                if (i == t->numel-1) printf("%*s%.*f)\n", (int)indent, "", precision, cfm_tensor_get_element(t, i));
                else printf("%*s%.*f,\n", (int)indent, "", precision, cfm_tensor_get_element(t, i));
            }
            break;
    }
}

void cfm_tensor_print(const cfm_tensor *t, int precision) {
    CFM_ASSERT(precision > 0 && precision <= 6);
    // (1, 2, 3, 4, 5, 6)       IN MEMORY, stored row major
    // [[1, 2, 3], [4, 5, 6]]   IN THIS cfm_tensor IMPLEMENTATION 
    /* for a shape[2, 3] tensor:
     *
     */
        
    for (uint64_t i = 0; i < t->numel; ++i) {

        printf("%.*f", precision, cfm_tensor_get_element(t, i));
    }
}
