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

#include "sgemv.h"

void matrixTranspose(float* array, size_t m, size_t n, float *buffer) {}
void fully_connected_inference_direct(const int input_size, const int output_size, const float *x, const float *y, float *z, const int num_threads) {}
void fully_connected_transpose_inference_neon8(const int input_size, const int output_size, const float *x, const float *y, float *z, const int num_threads) {}
void fully_connected_inference_direct_BiasReLU(int input_size, int output_size, float *x, float *y, float *z, float* biasArr, int num_threads) {}
void fully_connected_transpose_inference_neon8_BiasReLU(int input_size, int output_size, float *x, float *y, float *z, float* biasArr, int num_threads) {}