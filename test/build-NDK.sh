#!/bin/bash

${NDK_ROOT}/ndk-build clean
${NDK_ROOT}/ndk-build
cp ./libs/arm64-v8a/test /media/psf/Home/nfs/feather
