#!/usr/bin/env bash

cd ../..
#rm -rf build-android
bash build_scripts/build_android.sh
cd test/feather/
${NDK_ROOT}/ndk-build clean
${NDK_ROOT}/ndk-build
target_abi=armeabi-v7a
target_dir=/data/local/tmp/feather/${target_abi}
adb shell mkdir -p ${target_dir}
adb push ./libs/${target_abi}/feather_test ${target_dir}
