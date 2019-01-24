pushd ../../
bash ./build_scripts/build_macos_avx.sh
popd
g++ -o booster_test conv_test.cpp utils.cpp -I../../build-macos-avx/install/booster/include/ -L../../build-macos-avx/install/booster/lib/ -lbooster
