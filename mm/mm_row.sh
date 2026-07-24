#!/bin/bash

# default
#clang -O3 -march=native -mno-avx512f -Wall -Wextra mm_row.c -o mm_row && ./mm_row
# multithread
clang -O3 -fopenmp=libomp -march=native -mno-avx512f -Wall -Wextra mm_row.c -o mm_row && ./mm_row
# debug
#clang -O3 -DCHECK -march=native -mno-avx512f -Wall -Wextra mm_row.c -o mm_row && ./mm_row

