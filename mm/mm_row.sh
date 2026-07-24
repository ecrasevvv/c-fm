#!/bin/bash

clang -O3 -march=native -mno-avx512f -Wall -Wextra mm_row.c -o mm_row && ./mm_row
#clang -O3 -DCHECK -march=native -mno-avx512f -Wall -Wextra mm_row.c -o mm_row && ./mm_row
