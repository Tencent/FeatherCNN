#!/bin/bash

mkdir -p build-linux-aarch64
pushd build-linux-aarch64
#cmake -DCMAKE_TOOLCHAIN_FILE=../build_scripts/linux-aarch64.toolchain.cmake .. -DFEATHER_ARM=true -DCOMPILE_OPENCL=false
cmake  .. -DBOOSTER_ARM=true -DCOMPILE_OPENCL=false -DCMAKE_BUILD_TYPE=Release
make -j4
make install
popd
