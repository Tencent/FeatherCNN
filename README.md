<img width="420"  src="https://github.com/Tencent/FeatherCNN/wiki/Images/logo.png"/>

[![license](http://img.shields.io/badge/license-BSD3-blue.svg?style=flat)](https://github.com/Tencent/FeatherCNN/blob/master/LICENSE)
[![Release Version](https://img.shields.io/badge/release-0.1.0-red.svg)](https://github.com/Tencent/FeatherCNN/releases)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Tencent/FeatherCNN/pulls)

## Introduction

FeatherCNN, developed by Tencent TEG AI Platform, is a high performance lightweight CNN inference library. FeatherCNN is currently targeting at ARM CPUs, and is capable to extend to other devices in the future.

Comparing with other libraries, FeatherCNN is 

- **Highly Performant** FeatherCNN delievers state-of-the-art inference computing performance on a wide range of devices, including mobile phones (iOS/Android), embeded devices (Linux) as well as ARM-based servers (Linux). 

- **Easily Deployable** FeatherCNN packs everything in a single code base to get rid of third party dependencies. Hence, it facilatites deployment on mobile platforms. FeatherCNN's own model format is fully compatible to Caffe models. We are working to provide compatibility with other pre-trained models.

- **Featherweight** The compiled FeatherCNN library is in small size of several hundred KBs. 

Please kindly open an issue in this repo for bug reports and enhancement suggests. We are grateful to user responses and will actively polish this library.

## Quick guide on Ubuntu host and ARM-Linux targets.
If you are using Ubuntu and want to test on an ARM-Linux devices, here's an quick guide.
#### Host side compilation
- Install compilers
```
sudo apt-get install cmake
sudo apt-get install g++-aarch64-linux-gnu
```
- Download source code
```
git clone http://github.com/tencent/FeatherCNN
```
- Compiling and Install 
```
cd FeatherCNN
./build_scripts/build_linux.sh	
./build_scripts/build_linux_test.sh
```

#### Devide-side test example
The following command will run a benchmark with respect to specific network, input data, loop count and thread numbers. 
You can also check results with this program.
```
./feather_benchmark [feathermodel] [input_data] [loops] [threads number]
```
An example:
```
./feather_benchmark ./data/mobilenet.feathermodel ./data/input_3x224x224.txt 20 4	
```

## Detailed Instructions for iOS/Android/Linux

[**Build From Source**](https://github.com/Tencent/FeatherCNN/wikis/Build-From-Source)

[**iOS Guide**](https://github.com/Tencent/FeatherCNN/wikis/iOS-Guide)

[**Android Guide**](https://github.com/Tencent/FeatherCNN/wiki/Android-Guide)

## Usage

### Model Format Conversion

FeatherCNN accepts Caffemodels. It merges the structure file (.prototxt) and the weight file (.caffemodel) into a single binary model (.feathermodel).The convert tool requires protobuf, but you don't need them for the library. 

[**Model Convert Guide**](https://github.com/Tencent/FeatherCNN/wikis/Model-Convert-Guide).

### Runtime Interfaces

The basic user interfaces are listed in feather/net.h. Currently we are using raw pointers to reference data.
We may provide more convienent interfaces in the near future.

Before inference, FeatherCNN reqiures two steps to initialize the network.
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
