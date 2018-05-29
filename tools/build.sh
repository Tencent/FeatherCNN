#!/bin/bash
flatc -c flatbuffer_protocols/feather_simple.fbs && mv feather_simple_generated.h ../src/
protoc --cpp_out=. ./caffe.proto
g++ -g feather_convert_caffe.cc caffe.pb.cc -I/usr/include `pkg-config --cflags --libs protobuf` -o feather_convert_caffe -std=c++11 -I../src
