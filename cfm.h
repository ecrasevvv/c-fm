#ifndef CFM_H_
#define CFM_H_

#include <stdint.h>
#include <stdbool.h>

#define CFM_MAX_DIMS        4
#define CFM_STRING_MAX_LEN  64

# define M_PI		3.14159265358979323846	/* pi */
#define _2_PI       6.28318530717958647692  /* pi*2 */

typedef enum {
    CFM_FLOAT32 = 0,
    CFM_FLOAT64 = 1,
} cfm_dtype;

typedef enum {
    CFM_H_PRINT = 0,    /* Horizontal print: (1, 2, 3, 4, ...) */
    CFM_V_PRINT = 1,    /* Vertical print: (1,
                                            2,
                                            ...) */
} cfm_print_mode;

typedef struct {
    const char *content;
    uint64_t len;
} cfm_string;

typedef struct {
    cfm_string name;
    uint8_t ndims;                  /* How many dimension;                      pytorch: t.dim() */
    uint16_t shape[CFM_MAX_DIMS];   /* How many elements for each dimension;    pytorch: t.shape */
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
cfm_tensor *cfm_tensor_new(cfm_string name, cfm_dtype dtype,
        uint8_t ndims, uint16_t *shape, bool requires_grad);
/* This function creates a new cfm_tensor from existing data. */
cfm_tensor *cfm_tensor_from(cfm_string name, cfm_dtype dtype,
        uint8_t ndims, uint16_t *shape, void *data, bool requires_grad);
/* This function returns a new cfm_tensor filled with random
 * numbers from a uniform distribution on the interval [0, 1). */
cfm_tensor *cfm_tensor_rand(cfm_string name, cfm_dtype dtype,
        uint8_t ndims, uint16_t *shape, bool requires_grad);
/* This function returns a new cfm_tensor filled with random 
 * numbers from a normal distribution with mean 0 and variance 1. 
 * Implemented following https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform */
cfm_tensor *cfm_tensor_randn(cfm_string name, cfm_dtype dtype, 
        uint8_t ndims, uint16_t *shape, bool requires_grad);
void cfm_tensor_free(cfm_tensor *t);
/* This function prints out the cfm_tensor t in the pytorch style. */
void cfm_tensor_print(const cfm_tensor *t);
void cfm_tensor_print_raw(const cfm_tensor *t, cfm_print_mode pm);

#endif
