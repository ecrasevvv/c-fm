CC = clang
SOURCES = mm.c
CFLAGS = -O3 -march=native -mno-avx512f -Wall -Wextra
TARGET = mm
D_CFLAGS = -O3 -march=native -mno-avx512f -Wall -Wextra -DDEBUG
D_TARGET = dmm
PERF_FLAGS = cycles,instructions,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses

help:
	@echo "make mm: compile, run and perf mm.c"

$(TARGET):
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) && perf stat -e $(PERF_FLAGS) ./$(TARGET)

$(D_TARGET):
	$(CC) $(D_CFLAGS) $(SOURCES) -o $(D_TARGET) && perf stat -e $(PERF_FLAGS) ./$(D_TARGET)

clean: 
	rm -f $(TARGET) $(D_TARGET)

.PHONY: help $(TARGET) $(D_TARGET) clean
