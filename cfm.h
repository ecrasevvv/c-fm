#ifndef CFM_H_
#define CFM_H_

#include <stdint.h>
#include <stdbool.h>

#define CFM_MAX_DIMS        4
#define CFM_STRING_MAX_LEN  64

typedef enum {
    CFM_FLOAT32 = 0,
    CFM_FLOAT64 = 1,
} cfm_dtype;

typedef enum {
    CFM_H_PRINT = 0,    /* Horizontal print. */
    CFM_V_PRINT = 1,    /* Vertical print. */
} cfm_print_mode;

typedef struct {
    const char *content;
    uint64_t len;
} cfm_string;

typedef struct {
    cfm_string *name;
    uint8_t ndims;                  /* How many dimension;                      pytorch: t.dim() */
    uint16_t shape[CFM_MAX_DIMS];   /* How many elements for each dimension;    pytorch: t.shape */
    uint16_t strides[CFM_MAX_DIMS];
    uint64_t numel;
    cfm_dtype dtype;
    void *data;
    
    // grad..
    bool requires_grad;
} cfm_tensor;

/* This function allows the user to create a new cfm_string filled with content. */
cfm_string *cfm_string_new(const char *content);
void cfm_string_print(const cfm_string *str);
void cfm_string_free(cfm_string *str);

/* This function creates a new cfm_tensor. */
cfm_tensor *cfm_tensor_new(const char *name, cfm_dtype dtype,
        uint8_t ndims, uint16_t *shape, bool requires_grad);

/* This function creates a new cfm_tensor from existing data. */
cfm_tensor *cfm_tensor_from(const char *name, cfm_dtype dtype,
        uint8_t ndims, uint16_t *shape, void *data, bool requires_grad);

/* This function returns a new cfm_tensor filled with random numbers from a 
 * uniform distribution on the interval [0, 1). */
cfm_tensor *cfm_tensor_rand(const char *name, cfm_dtype dtype,
        uint8_t ndims, uint16_t *shape, bool requires_grad);

/* This function returns a new cfm_tensor filled with random 
 * numbers from a normal distribution with mean 0 and variance 1. 
 * Implemented following https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform */
cfm_tensor *cfm_tensor_randn(const char *name, cfm_dtype dtype, 
        uint8_t ndims, uint16_t *shape, bool requires_grad);

void cfm_tensor_free(cfm_tensor *t);

/* This function returns a new one-dimensional cfm_tensor of size end/step_size whose
 * values are evenly spaced from start to end. */
#define cfm_tensor_linspace(name, dtype, start, end, step_size, requires_grad)  \
    _Generic((start),                                                           \
            float:  cfm_tensor_linspace_float32,                                \
            double: cfm_tensor_linspace_float64                                 \
            )(name, dtype, start, end, step_size, requires_grad) 
cfm_tensor *cfm_tensor_linspace_float32(const char *name, cfm_dtype dtype,
        float start, float end, float step_size, bool requires_grad);
cfm_tensor *cfm_tensor_linspace_float64(const char *name, cfm_dtype dtype,
        double start, double end, double step_size, bool requires_grad);

/* This function returns a new cfm_tensor filled with fill_value. */
#define cfm_tensor_full(name, dtype, ndims, shape, fill_value, requires_grad)   \
    _Generic((fill_value),                                                      \
            float:  cfm_tensor_full_float32,                                    \
            double: cfm_tensor_full_float64                                     \
            )(name, dtype, ndims, shape, fill_value, requires_grad)
cfm_tensor *cfm_tensor_full_float32(const char *name, cfm_dtype dtype,
        uint8_t ndims, uint16_t *shape, float fill_value, bool requires_grad);
cfm_tensor *cfm_tensor_full_float64(const char *name, cfm_dtype dtype,
        uint8_t ndims, uint16_t *shape, double fill_value, bool requires_grad);

/* This function returns a new cfm_tensor filled with zeros. */
cfm_tensor *cfm_tensor_zeros(const char *name, cfm_dtype dtype,
        uint8_t ndims, uint16_t *shape, bool requires_grad);

/* This function returns a new cfm_tensor filled with ones. */
cfm_tensor *cfm_tensor_ones(const char *name, cfm_dtype dtype,
        uint8_t ndims, uint16_t *shape, bool requires_grad);

//cat
//expand
//the python xt[-1] equivalent for cfm_tensor

/* This function prints out the cfm_tensor t in the pytorch style. */
void cfm_tensor_print(const cfm_tensor *t, int precision);
void cfm_tensor_print_raw(const cfm_tensor *t, cfm_print_mode pm, int precision);

#endif
