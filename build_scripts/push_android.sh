#!/bin/bash

./build_scripts/build_android.sh
pushd test
./install.sh
popd
