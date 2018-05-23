#!/bin/bash

mkdir -p build-android
pushd build-android
mkdir -p arm64-v8a
pushd arm64-v8a
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK_ROOT/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-22 -DFEATHER_ARM=1 ../..
make -j4
make install
popd
popd
