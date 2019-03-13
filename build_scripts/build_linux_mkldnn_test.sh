#!/bin/bash

./build_scripts/build_linux_avx_mkldnn.sh
g++ -g -Og -std=c++11 ./test/feather/test_txt.cpp -fopenmp -I./build-linux-avx-mkldnn/install/booster/include/ -I./build-linux-avx-mkldnn/install/feather/include/ -L ./build-linux-avx-mkldnn/install/feather/lib/ -lfeather  -o feather_test_txt -lmkldnn
#g++ ./test/feather/test_bin.cpp -fopenmp -I./build-linux-avx/install/feather/include/ -L ./build-linux-avx/install/feather/lib/ -lfeather  -g -o feather_test_bin
