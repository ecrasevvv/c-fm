CC = clang
SOURCES = mm.c
CFLAGS = -O3 -march=native -mno-avx512f -Wall -Wextra
TARGET = mm
PERF_FLAGS = cycles,instructions,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses

help:
	@echo "make mm: compile, run and perf mm.c"

$(TARGET):
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) && perf stat -e $(PERF_FLAGS) ./mm

clean: 
	rm -f $(TARGET)

.PHONY: help $(TARGET) clean
