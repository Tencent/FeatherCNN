#!/bin/bash
#flatc -c flatbuffer_protocols/feather_simple.fbs && mv feather_simple_generated.h ../src/
echo "Compiling caffe proto..."
protoc --cpp_out=. ./caffe.proto
echo "Building FP32 convertor..."
g++ -g feather_convert_caffe.cc caffe.pb.cc -I/usr/include `pkg-config --cflags --libs protobuf` -o feather_convert_caffe -std=c++11 -I../src
echo "Building FP16 convertor..."
g++ -g feather_convert_caffe.cc caffe.pb.cc -I../src -I/usr/include `pkg-config --cflags --libs protobuf` -o feather_convert_caffe_fp16 -std=c++11 -I../src -DFP16_STORAGE
