#!/bin/bash

./build_scripts/build_macos_avx.sh
clang++ -std=c++11 ./test/feather/test_txt.cpp -I./build-macos-avx/install/feather/include/ -L ./build-macos-avx/install/feather/lib/ -lfeather  -O3 -o feather_test_txt
#clang++ -std=c++11 ./test/feather/test_bin.cpp -I./build-macos-avx/install/feather/include/ -L ./build-macos-avx/install/feather/lib/ -lfeather  -O3 -o feather_test_bin
