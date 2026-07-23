RED=\033[0;31m
RESET=\033[0m

CC=gcc

SOURCES=bench/bench.c cfm.c
B_TARGET=B_cfm
MT_B_TARGET=MT_B_cfm
D_TARGET=D_cfm
MT_D_TARGET=MT_D_cfm

NATIVE_CPU_FLAG=-march=native

BENCH_O=-O3
DEBUG_O=-O0

BENCH_CFLAGS=$(BENCH_O) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
BENCH_MULTITHREAD_CFLAGS=$(BENCH_O) $(NATIVE_CPU_FLAG) -fopenmp -Wall -Wextra -std=c99

DEBUG_CFLAGS=$(DEBUG_O) -DDEBUG -Wall -Wextra -std=c99 -g
DEBUG_MULTITHREAD_CFLAGS=$(DEBUG_O) -DDEBUG -fopenmp -Wall -Wextra -std=c99 -g

LDLIBS=-lm

help:
	@echo "CFM build targets:"
	@echo " - make bench		Build the bench version and run in."
	@echo " - make mtbench		Build the bench multi-thread version and run in."
	@echo " - make debug		Build the debug version and run in."
	@echo " - make mtdebug		Build the debug multi-thread version and run in."
	@echo " - make clean		Remove all the built version."
 
bench:
	@printf '${RED}BENCH${RESET}: '
	$(CC) $(BENCH_CFLAGS) $(SOURCES) $(LDLIBS) -o $(B_TARGET) && ./$(B_TARGET)

mtbench:
	@printf '${RED}MULTITHREAD BENCH${RESET}: '
	$(CC) $(BENCH_MULTITHREAD_CFLAGS) $(SOURCES) $(LDLIBS) -o $(MT_B_TARGET) && ./$(MT_B_TARGET)

debug:
	@printf '${RED}DEBUG${RESET}: '
	$(CC) $(DEBUG_CFLAGS) $(SOURCES) $(LDLIBS) -o $(D_TARGET) && ./$(D_TARGET)

mtdebug:
	@printf '${RED}MULTITHREAD DEBUG${RESET}: '
	$(CC) $(DEBUG_MULTITHREAD_CFLAGS) $(SOURCES) $(LDLIBS) -o $(MT_D_TARGET) && ./$(MT_D_TARGET)

clean:
	@rm -f $(B_TARGET) $(MT_B_TARGET) $(D_TARGET) $(MT_D_TARGET)

.PHONY: help bench mtbench debug mtdebug clean
