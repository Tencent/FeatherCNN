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
#ifndef TCNN_GENERIC_KERNELS_H_
#define TCNN_GENERIC_KERNELS_H_

#include <stdio.h>

void pad_input(float* padded, const float* input, const size_t input_channels, const size_t input_width, const size_t input_height, const size_t padding_left, const size_t padding_top, const size_t padding_right, const size_t padding_bottom);

template<bool fuse_relu>
void add_relu(float* dst, const float* A, const float* B, const size_t len, const size_t num_threads);

template<bool has_bias>
void scale(const size_t channels, const size_t stride, const float* bias_data, const float* scale_data, const float* input, float* output, const size_t num_threads);

template<bool has_bias, bool has_scale, bool has_relu>
void batchnorm(const size_t channels, const size_t stride, const float* alpha, const float* beta, const float* bias_data, const float* scale_data, const float* input, float* output, const size_t num_threads);

//void dwConv(float* output, const float* input, const int inw, const int inh, const int stridew, const int strideh, const float* kernel, const int kw, const int kh, const int group, const int nThreads);
void softmax(float* input, float n);
bool pooling(float *A, float *B, const char *type, int input_channels, size_t kernelw, size_t kernelh, size_t outputw, size_t outputh, int output_channels);

void naive_gemm(int M, int N, int L, float *A, float *B, float *C);

void relu(float* arr, int len);
void biasRelu(float* arr, int len, float bias);
void reluVec(float* arr, int len);
void biasVec(float* arr, int len, float bias);
void biasReluVec(float* arr, int len, float bias);
void reluVecOpenmp(float* arr, int len, int nThreads);
void biasVecOpenmp(float* arr, int len, float bias, int nThreads);
void biasReluVecOpenmp(float* arr, int len, float bias, int nThreads);
#endif /* TCNN_GENERIC_KERNELS_H_ */
