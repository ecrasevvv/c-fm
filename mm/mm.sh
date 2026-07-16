#!/bin/bash

RED='\033[0;31m'
RESET='\033[0m'

PERF_EVENTS="cycles,instructions,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses"

if [ "$1" == "--help" ]; then
    echo "./mm.sh 0     Compile and run mm.c without warmup and without vectorization."
    echo "./mm.sh 0 1   Compile and run mm.c with warmup and without vectorization."
    echo "./mm.sh 1     Compile and run mm.c without warmup with vectorization."
    echo "./mm.sh 1 1   Compile and run mm.c with warmup with vectorization."
    exit
fi

if [ "$1" -eq 0 ]; then
    if [ "$2" -eq 1 ]; then
        printf "${RED}WARMUP NO VECTORIZATION${RESET}\n"
        rm -f mm
        gcc -DDWARMUP -O3 -fno-tree-vectorize -Wall -Wextra mm.c -o mm && perf stat -e ${PERF_EVENTS} ./mm
    else
        printf "${RED}NO WARMUP NO VECTORIZATION${RESET}\n"
        rm -f mm
        gcc -O3 -fno-tree-vectorize -Wall -Wextra mm.c -o mm && perf stat -e ${PERF_EVENTS} ./mm
    fi
else
    if [ "$2" -eq 1 ]; then
        printf "${RED}WARMUP VECTORIZATION AVX2${RESET}\n"
        rm -f mm
        gcc -DDWARMUP -O3 -mavx2 -mfma -ffast-math -Wall -Wextra mm.c -o mm && perf stat -e ${PERF_EVENTS} ./mm
    else
        printf "${RED}NO WARMUP VECTORIZATION AVX2${RESET}\n"
        rm -f mm
        gcc -O3 -mavx2 -mfma -ffast-math -Wall -Wextra mm.c -o mm && perf stat -e ${PERF_EVENTS} ./mm
    fi
fi
