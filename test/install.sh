#!/bin/bash

${NDK_ROOT}/ndk-build clean
${NDK_ROOT}/ndk-build
adb push ./libs/arm64-v8a/test /data/local/tmp
#adb push ./libs/armeabi-v7a/test /data/local/tmp
