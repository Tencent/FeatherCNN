#!/usr/bin/env bash

target_abis=(armeabi-v7a) # default abi
if [ $# -gt 1 ]; then
    target_abis=( ${@:1} )
fi

## python and jinja2 required.
#pushd ../../src/booster/clhpp/python/tools/
#python clfile_converter.py
#popd

pushd ../..
rm -rf build-android
bash build_scripts/build_android.sh
feather_version=$(basename $(pwd))
popd

${NDK_ROOT}/ndk-build clean
${NDK_ROOT}/ndk-build

target_dir="/data/local/tmp/${feather_version}"
#pushd ~
#adb shell mkdir -p ${target_dir}
#adb push feathermodels ${target_dir}
#adb push test_models.sh ${target_dir}
#popd

for target_abi in "${target_abis[@]}"
do
    adb shell mkdir -p ${target_dir}/${target_abi}
    adb push ./libs/${target_abi}/feather_test ${target_dir}/${target_abi}

#    echo "testing models..."
#    adb shell "cd ${target_dir} && sh test_models.sh" &> ${target_abi}_result
done
