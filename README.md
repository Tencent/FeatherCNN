# FeatherCNN Overview

[![license](http://img.shields.io/badge/license-BSD3-blue.svg?style=flat)](http://git.code.oa.com/haidonglan/Feather-dev/blob/master/LICENSE)
[![Release Version](https://img.shields.io/badge/release-0.1.0-red.svg)](http://git.code.oa.com/haidonglan/Feather-dev)

# Introduction

FeatherCNN, developed by Tencent TEG AI Platform, is an ultra fast lightweight CNN inference library targeting devices with ARM CPUs. 

Comparing with other libraries, Feather is 
- [**Ultrafast**](Ultrafast)   FeatherCNN along extracts rooftop performance from CPUs and provides comparative performance with other 
top AI chips for mobilephones (iOS/Android), embeded devices(Linux), and ARM servers(Linux). 

- [**Compatibility**](Compatibility) Google flatbuffer is employed to transform protocalbuf and trainded caffemodel before inference 
compuation. FeatherCNN currently supports 14 types of layers and these can inference most widely used networks, such as, VGG, MobileNet 
GoogleNet, ResNet, and etc. 

- [**Easy-usage**](Easy-usage) Allinone, no third party software dependency, download, compile and ready to go. Commonly used models can 
be download from [models](http://hpcc.siat.ac.cn/jintao/feathercnn/models/), self trained models must be first transformed with our tools [site]. 


# Installation
## Impatient Installation on ARM devices with Ubuntu/Federoa/debian 
### Install prerequired libraries
	sudo apt-get install cmake
	sudo apt-get install g++-aarch64-linux-gnu 

### Download source code
	git clone http://github.com/tencent/FeatherCNN

### Compiling and Install 
	cd FeatherCNN
	./build_scripts/build_linux.sh	
	./build_scripts/build_linux_test.sh

### Testing and Benchmark examples
	The following command will inference mobilenet 20 loops with 4 threads. 
	(usage: ./feather_benchmark [feathermodel] [input_data] [loops] [threads number])
	./feather_benchmark ./data/mobilenet.feathermodel ./data/input_2x3x224x224.txt 20 4	

# Benchmarks
##The following models are benchmarked:
|Network|Layers|Top-1 error|Top-5 error|Speed (ms)|Citation|
|---|---:|---:|---:|---:|---|
|[Inception-V1](#inception-v1)|22|-|10.07|39.14|[[2]](#inception-v1-paper)|
|[VGG-16](#vgg-16)|16|27.00|8.80|128.62|[[3]](#vgg-paper)|
|[VGG-19](#vgg-19)|19|27.30|9.00|147.32|[[3]](#vgg-paper)|
|[ResNet-18](#resnet-18)|18|30.43|10.76|31.54|[[4]](#resnet-cvpr)|
|[ResNet-34](#resnet-34)|34|26.73|8.74|51.59|[[4]](#resnet-cvpr)|
|[ResNet-50](#resnet-50)|50|24.01|7.02|103.58|[[4]](#resnet-cvpr)|
|[ResNet-101](#resnet-101)|101|22.44|6.21|156.44|[[4]](#resnet-cvpr)|
|[ResNet-152](#resnet-152)|152|22.16|6.16|217.91|[[4]](#resnet-cvpr)|
|[ResNet-200](#resnet-200)|200|21.66|5.79|296.51|[[5]](#resnet-eccv)|

### Advanced Installation on IOS/Android/Other Linuxs
####[Cross compiling on X86 Guide](http://git.code.oa.com/haidonglan/Feather-dev/wikis/Linux-aarch64-Guide)
####[iOS Guide](http://git.code.oa.com/haidonglan/Feather-dev/wikis/iOS-Guide)
####[Android Guide](http://git.code.oa.com/haidonglan/Feather-dev/wikis/Android-Guide)
####[Build From Source](http://git.code.oa.com/haidonglan/Feather-dev/wikis/Build-From-Source)

## Usage
### Model Format Conversion

Feather naturally accepts Caffe models. It merges the structure file (.prototxt)
and the weight file (.caffemodel) into a single binary feather model (.feathermodel).

The conversion tool depends on flatbuffers and protobuf, but you don't need them
for the library. We demonstrate tool building procedure from scratch on a freshly
installed Ubuntu 18.04 in [here](http://git.code.oa.com/haidonglan/Feather-dev/wikis/Model-Conversion-Tool-Builiding-Guide).

### Runtime Interfaces

The basic feather user interfaces are listed in feather/net.h.
Before inference, Feather needs two-step iniitalization
```cpp
feather::Net forward_net(num_threads);
forward_net.InitFromPath(FILE_PATH_TO_FEATHERMODEL);
```
The Net class can also initialize with raw buffers and FILE pointers.
After initialization, we can perform forward computation with raw buffer
```cpp
forward_net.Forward(PTR_TO_YOUR_INPUT_DATA);
```
Output results can be extracted from the net.
```cpp
forward_net.ExtractBlob(PTR_TO_YOUR_OUTPUT_BUFFER, BLOB_NAME);
```
or just extract the last blob.
```cpp
forward_net.ExtractLastBlob(PTR_TO_YOUR_OUTPUT_BUFFER);
```

## Play Under The Hood
[How To Customize A Layer](http://git.code.oa.com/haidonglan/Feather-dev/wikis/How-To-Customize-A-Layer)
[Standalone Backend Usage](http://git.code.oa.com/haidonglan/Feather-dev/wikis/Standalone-Backend-Usage)
[Performance Evaluation]()

## Known Issues
* CMake 3.9.1 which comes with Ubuntu 17.10.1 may have problems in finding OpenMP package.

## Authors


