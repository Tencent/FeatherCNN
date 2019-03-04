#!/bin/bash

echo $(xcrun --sdk iphoneos --show-sdk-path)
mkdir -p build-ios
pushd build-ios
mkdir -p arm64 
pushd arm64
cmake -DCMAKE_TOOLCHAIN_FILE=../../build_scripts/ios.toolchain.cmake -DIOS_SDK_PATH=$(xcrun --sdk iphoneos --show-sdk-path) -DIOS_ARCH=arm64 -DBOOSTER_ARM=1 ../..
make -j4
make install
popd

mkdir -p armv7s
pushd armv7s
cmake -DCMAKE_TOOLCHAIN_FILE=../../build_scripts/ios.toolchain.cmake -DIOS_SDK_PATH=$(xcrun --sdk iphoneos --show-sdk-path) -DIOS_ARCH=armv7s -DBOOSTER_ARM=1 ../..
make -j4
make install
popd

#mkdir -p armv7
#pushd armv7
#cmake -DCMAKE_TOOLCHAIN_FILE=../../build_scripts/ios.toolchain.cmake -DIOS_SDK_PATH=$(xcrun --sdk iphoneos --show-sdk-path) -DIOS_ARCH=armv7 ../..
#make -j4
#make install
#popd

#mkdir -p x86_64
#pushd x86_64
#cmake -DCMAKE_TOOLCHAIN_FILE=../../build_scripts/ios.toolchain.cmake -DIOS_SDK_PATH=$(xcrun --sdk iphonesimulator --show-sdk-path) -DIOS_ARCH=x86_64 ../..
#make -j4
#make install
#popd

#mkdir -p i386
#pushd i386
#cmake -DCMAKE_TOOLCHAIN_FILE=../../build_scripts/ios.toolchain.cmake -DIOS_SDK_PATH=$(xcrun --sdk iphonesimulator --show-sdk-path) -DIOS_ARCH=i386 ../..
#make -j4
#make install
#popd

popd
bash ./build_scripts/pack_ios_framework.sh
