#!/bin/bash

mkdir -p build-linux-avx-mkldnn
pushd build-linux-avx-mkldnn
cmake .. -DBOOSTER_MKLDNN=1 -DBOOSTER_AVX=1
make -j4
make install
popd
