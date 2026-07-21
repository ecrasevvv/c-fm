CC = clang
SOURCES = mm.c
CFLAGS = -O3 -march=native -mno-avx512f -Wall -Wextra
TARGET = mm
D_CFLAGS = -O3 -march=native -mno-avx512f -Wall -Wextra -DDEBUG
D_TARGET = dmm
S_CFLAGS = -O3 -march=native -mno-avx512f -Wall -Wextra -DDEBUG -S
S_TARGET = smm
M_CFLAGS = -O3 -fopenmp=libomp -march=native -mno-avx512f -Wall -Wextra -DNTHREADS=12
M_TARGET = mmm
DM_CFLAGS = -O3 -fopenmp=libomp -march=native -mno-avx512f -Wall -Wextra -DNTHREADS=12 -DDEBUG
DM_TARGET = dmmm
PERF_FLAGS = cycles,instructions,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses

help:
	@echo "make mm: 		Compile, run and perf mm.c"
	@echo "make dmm: 		Compile, run and perf mm.c debug version (contains matrix print)"
	@echo "make mmm: 		Compile, run and perf mm.c multi-thread version"
	@echo "make dmmm: 		Compile, run and perf mm.c debug and multi-thread version"
	@echo "make smm: 		Get assembly code (with SIMD FMA instruction) from mm.c"
	@echo "make clean: 		Remove build outputs"

$(TARGET):
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) && perf stat -e $(PERF_FLAGS) ./$(TARGET)

$(D_TARGET):
	$(CC) $(D_CFLAGS) $(SOURCES) -o $(D_TARGET) && perf stat -e $(PERF_FLAGS) ./$(D_TARGET)

$(M_TARGET):
	$(CC) $(M_CFLAGS) $(SOURCES) -o $(M_TARGET) && perf stat -e $(PERF_FLAGS) ./$(M_TARGET)

$(DM_TARGET):
	$(CC) $(DM_CFLAGS) $(SOURCES) -o $(DM_TARGET) && perf stat -e $(PERF_FLAGS) ./$(DM_TARGET)

$(S_TARGET):
	$(CC) $(S_CFLAGS) $(SOURCES) -o $(S_TARGET)

clean: 
	rm -f $(TARGET) $(D_TARGET) $(M_TARGET) $(DM_TARGET) $(S_TARGET)

.PHONY: help $(TARGET) $(D_TARGET) $(M_TARGET) $(DM_TARGET) $(S_TARGET) clean
