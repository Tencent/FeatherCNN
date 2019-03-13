pushd ../../
bash ./build_scripts/build_linux_avx_mkldnn.sh
popd

g++ -fopenmp -O3 -std=c++11 -o booster_test conv_test.cpp utils.cpp -DBOOSTER_USE_MKLDNN -I../../build-linux-avx-mkldnn/install/booster/include/ -L../../build-linux-avx-mkldnn/install/booster/lib/ -lbooster -lmkldnn -I${MKLROOT}/include -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lm -ldl
