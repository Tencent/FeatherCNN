#!/bin/bash

mkdir -p build-linux-avx
pushd build-linux-avx
cmake .. -DBOOSTER_AVX=1 -DCMAKE_BUILD_TYPE=Debug
make VERBOSE=1
make install
popd
