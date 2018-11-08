#!/bin/bash

aarch64-linux-gnu-g++ ./test/feather/test_txt.cpp -I./build-linux-aarch64/install/feather/include/ -L ./build-linux-aarch64/install/feather/lib/ -lfeather -fopenmp -O3 -o feather_benchmark
aarch64-linux-gnu-g++ ./test/feather/test_bin.cpp -I./build-linux-aarch64/install/feather/include/ -L ./build-linux-aarch64/install/feather/lib/ -lfeather -fopenmp -O3 -o feather_benchmark
