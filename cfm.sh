#!/bin/bash

if [[ "$1" == "d" ]]; then
    gcc "$2" cfm.c -DDEBUG -lm -Wall -Wextra -fsanitize=address -O2 -o cfm && ./cfm
else
    gcc test.c cfm.c -lm -Wall -Wextra -fsanitize=address -O2 -o cfm && ./cfm
fi
