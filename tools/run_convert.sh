#!/usr/bin/env bash

model_dir="/data1/andyao/android/caffemodel/"
if [ $# -ne 1 ]; then
    echo "usage: sh $(basename $0) model_name"
    echo "model_dir = ${model_dir}"
    exit 1
fi

model_name="$1"
proto="${model_dir}${model_name}.pt"
model="${model_dir}${model_name}.caffemodel"
prefix="./feathermodel/${model_name}"
./feather_convert_caffe \
    ${proto} \
    ${model} \
    ${prefix}
