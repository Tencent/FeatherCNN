#!/bin/bash

export mycmake=/data1/andyao/android/cmake-3.11.1-Linux-x86_64/bin/cmake

mkdir -p build-linux-aarch64
pushd build-linux-aarch64
$mycmake -DCMAKE_TOOLCHAIN_FILE=../build_scripts/linux-aarch64.toolchain.cmake .. -DFEATHER_ARM=1
make -j4
make install
popd
