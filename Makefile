RED=\033[0;31m
RESET=\033[0m

CC=gcc

T_TARGET=T_cfm
D_TARGET=D_cfm
B_TARGET=B_cfm

NATIVE_CPU_FLAG=-march=native

BASE_O=-O3
DEBUG_O=-O0

# fully optmizes
BASE_CFLAGS=$(BASE_O) -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
# only for correctness 
DEBUG_CFLAGS=$(DEBUG_O) -DDEBUG -Wall -Wextra -std=c99

LDLIBS=-lm

help:
	@echo "CFM build targets:"
	@echo "	make test			Build the optmized (-march-native) test and run it."
	@echo "	make debug			Build the debug version and run in."
	@echo "	make bench			Build the bench version and run in."
	@echo "	make tc				Remove the built test version."
	@echo "	make dc				Remove the built debug version."
	@echo "	make bc				Remove the built bench version."
 
test:
	$(CC) $(BASE_CFLAGS) test.c cfm.c $(LDLIBS) -o $(T_TARGET) && ./$(T_TARGET)

debug:
	@printf '${RED}DEBUG${RESET}: '
	$(CC) $(DEBUG_CFLAGS) test.c cfm.c $(LDLIBS) -o $(D_TARGET) && ./$(D_TARGET)

bench:
	@printf '${RED}BENCH${RESET}: '
	$(CC) $(BASE_CFLAGS) bench.c cfm.c $(LDLIBS) -o $(B_TARGET) && ./$(B_TARGET)

tc: 
	@rm -f $(T_TARGET)

dc: 
	@rm -f $(D_TARGET)

bc: 
	@rm -f $(B_TARGET)

.PHONY: help test debug bench tc dc bc
