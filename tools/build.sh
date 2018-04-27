#!/bin/bash

g++ -g feather_convert.cc caffe.o -I/usr/include `pkg-config --cflags --libs protobuf` -o feather_convert -std=c++11 -I../src
