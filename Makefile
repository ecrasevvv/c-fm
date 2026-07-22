RED=\033[0;31m
RESET=\033[0m

CC=gcc

SOURCES=bench/bench.c cfm.c
D_TARGET=D_cfm
B_TARGET=B_cfm

NATIVE_CPU_FLAG=-march=native

BENCH_O=-O3
DEBUG_O=-O0

BENCH_CFLAGS=$(BENCH_O) -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
DEBUG_CFLAGS=$(DEBUG_O) -DDEBUG -Wall -Wextra -std=c99

LDLIBS=-lm

help:
	@echo "CFM build targets:"
	@echo "	make debug			Build the debug version and run in."
	@echo "	make bench			Build the bench version and run in."
	@echo "	make clean			Remove all the built version."
 
bench:
	@printf '${RED}BENCH${RESET}: '
	$(CC) $(BENCH_CFLAGS) $(SOURCES) $(LDLIBS) -o $(B_TARGET) && ./$(B_TARGET)

debug:
	@printf '${RED}DEBUG${RESET}: '
	$(CC) $(DEBUG_CFLAGS) $(SOURCES) $(LDLIBS) -o $(D_TARGET) && ./$(D_TARGET)

clean:
	@rm -f $(D_TARGET) $(B_TARGET)

.PHONY: help debug bench clean
