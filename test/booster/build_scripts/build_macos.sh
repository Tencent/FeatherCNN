pushd ../../
bash ./build_scripts/build_macos_avx_mkldnn.sh
popd

g++ -O3 -std=c++11 -o booster_test conv_test.cpp utils.cpp -DBOOSTER_USE_MKLDNN -I../../build-macos-avx-mkldnn/install/booster/include/ -L../../build-macos-avx-mkldnn/install/booster/lib/ -lbooster -lmkldnn
