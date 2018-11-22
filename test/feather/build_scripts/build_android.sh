#!/bin/bash

cd ../..
bash build_scripts/build_android.sh
cd test/feather/
${NDK_ROOT}/ndk-build clean
${NDK_ROOT}/ndk-build
adb push ./libs/arm64-v8a/feather_test /data/local/tmp/booster_test/
#adb push ./libs/armeabi-v7a/feather_test /data/local/tmp/booster_test
