//Tencent is pleased to support the open source community by making FeatherCNN available.

//Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.

//Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
//in compliance with the License. You may obtain a copy of the License at
//
//https://opensource.org/licenses/BSD-3-Clause
//
//Unless required by applicable law or agreed to in writing, software distributed
//under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//CONDITIONS OF ANY KIND, either express or implied. See the License for the
//specific language governing permissions and limitations under the License.

// Booster is the standalone backend of FeatherCNN, in order to facilitate unit testing
// and multi-purpose deployment. I am currently focusing on the fast convolution kernels,
// and will pack other operators as well. This backend library is now supporting
// AVX and Neon, and is going to supoort OpenCL/GLES in the future. 
// Booster won't grow up into a hugh and abstract lib. I'll keep it simple and stupid.
// -- Haidong Lan @ Tencent AI Platform, 08/30/2018

#pragma once

#include <stdlib.h>
#include <stdio.h>

namespace booster{

enum ConvAlgo{
    NAIVE,
    IM2COL,
    SGECONV,
    DEPTHWISE,
    WINOGRADF63,
    WINOGRADF23,
};

enum ActivationType{
    None,
    ReLU,
};

struct ConvParam{
    int output_channels;
    int input_channels;
    int input_h;
    int input_w;
    int kernel_h;
    int kernel_w;
    int output_h;
    int output_w;
    int stride_h;
    int stride_w;
    int pad_left;
    int pad_bottom;
    int pad_right;
    int pad_top;
    int group;
    bool bias_term;
    ActivationType activation;
    void AssignOutputDim()
    {
        //Validate default values
        if(group == 0)	group = 1;
        if(stride_h == 0) stride_h = 1;
        if(stride_w == 0) stride_w = 1;
        output_h = (input_h + pad_top + pad_bottom - kernel_h) / stride_h + 1;
        output_w = (input_w + pad_left + pad_right - kernel_w) / stride_w + 1;
        if(group == input_channels)
        {
            output_channels = input_channels;
        }
    }
    void AssignPaddedDim()
    {
        input_h = input_h + pad_left + pad_right;
        input_w = input_w + pad_top + pad_bottom;
        pad_left = 0;
        pad_bottom = 0;
        pad_right = 0;
        pad_top = 0;
    }
    void LogParams(const char* layer_name)
    {
        printf("-----Layer %s ConvParam----\n", layer_name);
        printf("Input CxHxW=(%d, %d, %d)\n", input_channels, input_h, input_w);
        printf("Output CxHxW=(%d, %d, %d)\n", output_channels, output_h, output_w);
        printf("Kernel HxW=(%d, %d)\n", kernel_h, kernel_w);
        printf("Stride HxW=(%d, %d)\n", stride_h, stride_w);
        printf("Paddings (%d %d %d %d)\n", pad_left, pad_bottom, pad_right, pad_top);
    }
};

typedef int (*GET_BUFFER_SIZE_FUNC)(ConvParam *param, int* buffer_size, int* processed_kernel_size);
typedef int (*INIT_FUNC)(ConvParam *param, float* processed_kernel, float* kernel); 
typedef int (*FORWARD_FUNC)(ConvParam *param, float* output, float* input, float* kernel, float* buffer, float* bias_arr);

//ConvBooster doesn't allocate any memory.
class ConvBooster
{
public:
    ConvBooster();
    ~ConvBooster(){}
    int SelectAlgo(ConvParam* param);
    int ForceSelectAlgo(ConvAlgo algo);
    int SetFuncs();
    GET_BUFFER_SIZE_FUNC GetBufferSize;
    INIT_FUNC Init;
    FORWARD_FUNC Forward;

private:
    ConvAlgo algo;
};
};