#!/bin/bash

./build_scripts/build_linux_avx.sh
g++ ./test/test_txt.cpp -fopenmp -I./build-linux-avx/install/feather/include/ -L ./build-linux-avx/install/feather/lib/ -lfeather  -g -o feather_test_txt
g++ ./test/test_bin.cpp -fopenmp -I./build-linux-avx/install/feather/include/ -L ./build-linux-avx/install/feather/lib/ -lfeather  -g -o feather_test_bin
