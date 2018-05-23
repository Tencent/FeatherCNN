#!/bin/bash

mycmake=/data1/andyao/android/cmake-3.11.1-Linux-x86_64/bin/cmake
NDK_ROOT=/data1/andyao/android/android-ndk-r15c

mkdir -p build-android
pushd build-android
mkdir -p arm64-v8a
pushd arm64-v8a
$mycmake -DCMAKE_TOOLCHAIN_FILE=$NDK_ROOT/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-21 -DFEATHER_ARM=1 ../..
make -j4
make install
popd

mkdir -p armeabi-v7a
pushd armeabi-v7a
$mycmake -DCMAKE_TOOLCHAIN_FILE=$NDK_ROOT/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=android-16 -DFEATHER_ARM=1 ../..
make -j4
make install
popd

mkdir -p armeabi
pushd armeabi
$mycmake -DCMAKE_TOOLCHAIN_FILE=$NDK_ROOT/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi" -DANDROID_PLATFORM=android-16 -DFEATHER_ARM=0 ../..
make -j4
make install
popd

mkdir -p feather
pushd feather
mkdir -p include
mkdir -p include/feather
cp ../arm64-v8a/install/feather/include/* ./include/feather/
mkdir -p arm64-v8a
cp ../arm64-v8a/install/feather/lib/* ./arm64-v8a/
mkdir -p armeabi-v7a
cp ../armeabi-v7a/install/feather/lib/* ./armeabi-v7a/
mkdir -p armeabi
cp ../armeabi/install/feather/lib/* ./armeabi/
popd 
zip -r feather.zip feather/
popd
