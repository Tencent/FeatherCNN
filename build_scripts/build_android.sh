#!/bin/bash

mkdir -p build-android
pushd build-android
mkdir -p arm64-v8a
pushd arm64-v8a
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK_ROOT/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-21 -DFEATHER_ARM=1 ../..
make -j4
make install
popd

mkdir -p armeabi-v7a
pushd armeabi-v7a
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK_ROOT/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=android-16 -DFEATHER_ARM=1 ../..
make -j4
make install
popd

#mkdir -p armeabi
#pushd armeabi
#cmake -DCMAKE_TOOLCHAIN_FILE=$NDK_ROOT/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi" -DANDROID_PLATFORM=android-16 -DFEATHER_ARM=0 ../..
#make -j4
#make install
#popd

mkdir -p feather
pushd feather
mkdir -p include
mkdir -p include/feather
cp ../arm64-v8a/install/feather/include/* ./include/feather/
mkdir -p arm64-v8a
cp ../arm64-v8a/install/feather/lib/* ./arm64-v8a/
mkdir -p armeabi-v7a
cp ../armeabi-v7a/install/feather/lib/* ./armeabi-v7a/
#mkdir -p armeabi
#cp ../armeabi/install/feather/lib/* ./armeabi/
#popd 
popd
