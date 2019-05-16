#!/bin/bash

./build_scripts/build_linux_avx.sh
g++ -std=c++11 ./test/feather/test_txt.cpp -fopenmp -I./build-linux-avx/install/feather/include/ -L ./build-linux-avx/install/feather/lib/ -lfeather  -g -o feather_test_txt
#g++ ./test/feather/test_bin.cpp -fopenmp -I./build-linux-avx/install/feather/include/ -L ./build-linux-avx/install/feather/lib/ -lfeather  -g -o feather_test_bin
