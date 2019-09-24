<img width="420"  src="https://github.com/Tencent/FeatherCNN/wiki/Images/logo.png"/>

[![license](http://img.shields.io/badge/license-BSD3-blue.svg?style=flat)](https://github.com/Tencent/FeatherCNN/blob/master/LICENSE)
[![Release Version](https://img.shields.io/badge/release-0.1.0-red.svg)](https://github.com/Tencent/FeatherCNN/releases)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Tencent/FeatherCNN/pulls)

## Introduction

FeatherCNN is a high-performance lightweight CNN inference library, developed by Tencent AI Platform Department. 
FeatureCNN origins from our game AI project for King of Glory (Chinese: 王者荣耀), in which we aim to build a neural model for MOBA game AI and run it on mobile devices. 
FeatherCNN currently targets at ARM CPUs. 
We will extend it to cover other architecutures in the near future.

Comparing with other libraries, FeatherCNN has the following features: 

- **High Performance** FeatherCNN delivers state-of-the-art inference computing performance on a wide range of devices, including mobile phones (iOS/Android), embedded devices (Linux) as well as ARM-based servers (Linux). 

- **Easy Deployment** FeatherCNN packs everything in a single code base to get rid of third-party dependencies. Hence, it facilitates deployment on mobile platforms. 

- **Featherweight** The compiled FeatherCNN library is small-sized (hundreds of KBs). 

Please kindly open an issue in this repo for bug reports and enhancement suggests. We are grateful to user responses and will actively polish this library.

## Citation

FeatherCNN: Fast Inference Computation with TensorGEMM on ARM Architectures (TPDS September 2019, In press, DOI:10.1109/TPDS.2019.2939785)

## Clone hints
The FeatherCNN repository has a heavy development history, please only clone the master branch as follows:
```
git clone -b master --single-branch https://github.com/tencent/FeatherCNN.git
```

## Detailed Instructions for iOS/Android/Linux

[**Build From Source**](https://github.com/Tencent/FeatherCNN/wikis/Build-From-Source)

[**iOS Guide**](https://github.com/Tencent/FeatherCNN/wikis/iOS-Guide)

[**Android Guide**](https://github.com/Tencent/FeatherCNN/wiki/Android-Guide)

[**Android ADB Guide**](https://github.com/Tencent/FeatherCNN/wiki/Android-ADB-Guide)

## Usage

### Model Format Conversion

FeatherCNN accepts Caffemodels. It merges the structure file (.prototxt) and the weight file (.caffemodel) into a single binary model (.feathermodel). The convert tool requires protobuf, but you don't need them for the library. 

[**Model Convert Guide**](https://github.com/Tencent/FeatherCNN/wikis/Model-Convert-Guide).

### Runtime Interfaces

The basic user interfaces are listed in feather/net.h. Currently we are using raw pointers to reference data.
We may provide more convenient interfaces in the near future.

Before inference, FeatherCNN requires two steps to initialize the network.
```cpp
feather::Net forward_net(num_threads);
forward_net.InitFromPath(FILE_PATH_TO_FEATHERMODEL);
```
The net can also be initialized with raw buffers and FILE pointers.
We can perform forward computation with raw `float*` buffer consequently. 
```cpp
forward_net.Forward(PTR_TO_YOUR_INPUT_DATA);
```
The output can be extracted from the net by the name of blobs. The blob names are kept consistent with caffe prototxt.
```cpp
forward_net.ExtractBlob(PTR_TO_YOUR_OUTPUT_BUFFER, BLOB_NAME);
```
BTW, you can also get the blob's data size by calling
```cpp
size_t data_size = 0;
forward_net.GetBlobDataSize(&data_size, BLOB_NAME);
```

## Performance Benchmarks
We have tested FeatherCNN on a bunch of devices, see [**this page**](https://github.com/Tencent/FeatherCNN/wikis/Benchmarks) for details.

## User Groups

Telegram: https://t.me/FeatherCNN

QQ: 728147343
