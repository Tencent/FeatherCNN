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

#pragma once
#include <stdio.h>

enum WinogradOutType
{
    None, ReLU, Bias, BiasReLU
};

//UT larger than 16 * inChannels * outChannels
void transformKernel(float* UT, float* kernel, int inChannels, int outChannels, float *ST);

//VT larger than 16 * (inputw / 2 - 1) * (inputh / 2 - 1) * inChannels
//WT larger than 16 * (inputw / 2 - 1) * (inputh / 2 - 1) * outChannels
void winogradNonFusedTransform(float *output, int outChannels, float* WT, float* VT, float* UT, float* input, int inChannels, int inputw, int inputh, WinogradOutType outType, float* biasArr, int num_threads);

size_t getPackArraySize_F6x6_3x3(int inChannels, int num_threads);
void transformKernel_F6x6_3x3(float* UT, float* kernel, int inChannels, int outChannels);
void winogradNonFusedTransform_F6x6_3x3(float *output, int outChannels, float* WT, float* VT, float* UT, float* input, int inChannels, int inputw, int inputh, WinogradOutType outType, float* biasArr, float* pack_array, int num_threads);
