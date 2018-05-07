#!/bin/bash

protoc --cpp_out=. ./caffe.proto
g++ -g feather_convert_caffe.cc caffe.pb.cc -I/usr/include `pkg-config --cflags --libs protobuf` -o feather_convert_caffe -std=c++11 -I../src
