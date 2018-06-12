#!/bin/bash

./build_scripts/build_android.sh
pushd test
./test/install.sh
popd
