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

#include <booster/booster.h>
#include <booster/depthwise.h>
#include <booster/generic_kernels.h>
#include <booster/sgemm.h>
#include <booster/sgeconv.h>
#include <booster/helper.h>
#include <booster/winograd_kernels.h>

#include <string.h>

namespace booster
{
//NAIVE Methods
int NAIVE_GetBufferSize(ConvParam *param, int* buffer_size, int* processed_kernel_size)
{
    *buffer_size = param->input_channels * param->output_h * param->output_w * param->kernel_h * param->kernel_w;
    *processed_kernel_size = param->input_channels * param->output_channels * param->kernel_h * param->kernel_w;
    return 0;
}

int NAIVE_Init(ConvParam *param, float* processed_kernel, float* kernel)
{
    memcpy(processed_kernel, kernel, sizeof(float) * param->output_channels * param->input_channels * param->kernel_h * param->kernel_w);
    return 0;
}

int NAIVE_Forward(ConvParam *param, float* output, float* input, float* processed_kernel, float* buffer, float* bias_arr)
{
    const int M = param->output_channels;
    const int N = param->output_h * param->output_w;
    const int K = param->input_channels * param->kernel_h * param->kernel_w;
    im2col(param, buffer, input);
    naive_sgemm(M, N, K, processed_kernel, buffer, output);
    if (param->bias_term)
    {
        size_t out_stride = param->output_w * param->output_h;
        for (int i = 0; i < param->output_channels; ++i)
        {
            float bias = bias_arr[i];
            for (int j = 0; j < out_stride; ++j)
            {
                output[out_stride * i + j] = output[out_stride * i + j] + bias;
            }
        }
    }
    return 0;
}

//IM2COL Methods
int IM2COL_GetBufferSize(ConvParam *param, int* buffer_size, int* processed_kernel_size)
{
    *buffer_size = param->input_channels * param->output_h * param->output_w * param->kernel_h * param->kernel_w;
    *processed_kernel_size = param->input_channels * param->output_channels * param->kernel_h * param->kernel_w;
    return 0;
}

int IM2COL_Init(ConvParam *param, float* processed_kernel, float* kernel)
{
    const int M = param->output_channels;
    const int K = param->input_channels * param->kernel_h * param->kernel_w;
    packed_sgemm_init<6>(M, K, 320, processed_kernel, kernel, K);
    return 0;
}

int IM2COL_Forward(ConvParam *param, float* output, float* input, float* processed_kernel, float* buffer, float* bias_arr)
{
    const int M = param->output_channels;
    const int N = param->output_h * param->output_w;
    const int K = param->input_channels * param->kernel_h * param->kernel_w;
    im2col(param, buffer, input);
    if ((!param->bias_term) && (param->activation == None))
        packed_sgemm_activation<false, false>(M, N, K, processed_kernel, buffer, N, output, N, 160, 320, bias_arr, 1, NULL);
    else if ((param->bias_term) && (param->activation == None))
        packed_sgemm_activation<true,  false>(M, N, K, processed_kernel, buffer, N, output, N, 160, 320, bias_arr, 1, NULL);
    else if ((!param->bias_term) && (param->activation == ReLU))
        packed_sgemm_activation<false,  true>(M, N, K, processed_kernel, buffer, N, output, N, 160, 320, bias_arr, 1, NULL);
    else if ((param->bias_term) && (param->activation == ReLU))
        packed_sgemm_activation<true,   true>(M, N, K, processed_kernel, buffer, N, output, N, 160, 320, bias_arr, 1, NULL);
    return 0;
}

//SGECONV Methods
int SGECONV_GetBufferSize(ConvParam *param, int* buffer_size, int* processed_kernel_size)
{
    return 0;
}

int SGECONV_Init(ConvParam *param, float* processed_kernel, float* kernel)
{
    return 0;
}

int SGECONV_Forward(ConvParam *param, float* output, float* input, float* processed_kernel, float* buffer, float* bias_arr)
{
    return 0;
}

//DEPTHWISE Methods
int DEPTHWISE_GetBufferSize(ConvParam *param, int* buffer_size, int* processed_kernel_size)
{
    *buffer_size = param->input_channels * param->input_h * param->input_w;
    *processed_kernel_size = param->group * param->kernel_h * param->kernel_w;
    return 0;
}

int DEPTHWISE_Init(ConvParam *param, float* processed_kernel, float* kernel)
{
    memcpy(processed_kernel, kernel, sizeof(float) * param->group * param->kernel_h * param->kernel_w);
    return 0;
}

int DEPTHWISE_Forward(ConvParam *param, float* output, float* input, float* processed_kernel, float* buffer, float* bias_arr)
{
    void (*dwConv)(float *, float *, int, int, int, int, int, float *, int, int, int, int, float *);
    if (param->bias_term && (param->activation == ReLU))
        dwConv = dwConv_template<true, true>;
    else if (param->bias_term && !(param->activation == ReLU))
        dwConv = dwConv_template<true, false>;
    else if (!param->bias_term && (param->activation == ReLU))
        dwConv = dwConv_template<false, true>;
    else if (!param->bias_term && !(param->activation == ReLU))
        dwConv = dwConv_template<false, false>;

    if (param->pad_left > 0 || param->pad_right > 0 || param->pad_top > 0 || param->pad_bottom > 0)
    {
        pad_input(buffer, input, param->input_channels, param->input_w, param->input_h, param->pad_left,
                  param->pad_top, param->pad_right, param->pad_bottom);
        dwConv(output, buffer, param->input_channels, param->input_w, param->input_h, param->stride_w, param->stride_h, processed_kernel, param->kernel_w, param->kernel_h, param->group, 1, bias_arr);
    }
    else
        dwConv(output, input, param->input_channels, param->input_w, param->input_h, param->stride_w, param->stride_h, processed_kernel, param->kernel_w, param->kernel_h, param->group, 1, bias_arr);
    return 0;
}
//WINOGRADF23 Methods
int WINOGRADF23_GetBufferSize(ConvParam *param, int* buffer_size, int* processed_kernel_size)
{
    return 0;
}

int WINOGRADF23_Init(ConvParam *param, float* processed_kernel, float* kernel)
{
    return 0;
}

int WINOGRADF23_Forward(ConvParam *param, float* output, float* input, float* processed_kernel, float* buffer, float* bias_arr)
{
    return 0;
}

//WINOGRADF63 Methods
int WINOGRADF63_GetBufferSize(ConvParam *param, int* buffer_size, int* processed_kernel_size)
{
    int num_threads = 1;
    ConvParam padded_param = *param;
    padded_param.AssignPaddedDim();
    size_t packArraySize = getPackArraySize_F6x6_3x3(padded_param.input_channels, num_threads);
    int nRowBlocks = (padded_param.input_w + 3) / 6;
    int nColBlocks = (padded_param.input_h + 3) / 6;
    int nBlocks = nRowBlocks * nColBlocks;

    size_t winograd_mem_size = 0;
    winograd_mem_size += 64 * nBlocks * padded_param.input_channels;    //VT
    winograd_mem_size += 64 * nBlocks * padded_param.output_channels;   //WT
    winograd_mem_size += packArraySize;                    //WT
    winograd_mem_size += padded_param.input_w * padded_param.input_h * padded_param.input_channels; //Padded Input

    *buffer_size = winograd_mem_size;
    *processed_kernel_size = 64 * padded_param.input_channels * padded_param.output_channels;
    return 0;
}

int WINOGRADF63_Init(ConvParam *param, float* processed_kernel, float* kernel)
{
    transformKernel_F6x6_3x3(processed_kernel, kernel, param->input_channels, param->output_channels);
    return 0;
}

int WINOGRADF63_Forward(ConvParam *param, float* output, float* input, float* processed_kernel, float* buffer, float* bias_arr)
{
    const size_t inputw = param->input_w + param->pad_left + param->pad_right;
    const size_t inputh = param->input_h + param->pad_top + param->pad_bottom;
    const int nRowBlocks = (inputw + 3) / 6;
    const int nColBlocks = (inputh + 3) / 6;
    const int nBlocks = nRowBlocks * nColBlocks;

    //Get addresses
    float *VT = buffer;
    float *WT = VT + 64 * nBlocks * param->input_channels;                      //Offset by sizeof VT
    float *padded_input = WT + 64 * nBlocks * param->output_channels;           //Offset by sizeof WT
    float *pack_array = padded_input + inputw * inputh * param->input_channels; //Offset by sizeof WT
    pad_input(padded_input, input, param->input_channels, param->input_w, param->input_h, param->pad_left, param->pad_top, param->pad_right, param->pad_bottom);
    WinogradOutType out_type;
    if ((!param->bias_term) && (param->activation == None))
        out_type = Nothing;
    else if ((param->bias_term) && (param->activation == None))
        out_type = Bias;
    else if ((!param->bias_term) && (param->activation == ReLU))
        out_type = Relu;
    else if ((param->bias_term) && (param->activation == ReLU))
        out_type = BiasReLU;
    winogradNonFusedTransform_F6x6_3x3(output, param->output_channels, WT, VT, processed_kernel, padded_input, param->input_channels, inputh, inputw, out_type, bias_arr, pack_array, 1);
    return 0;
}

//Class wrappers
ConvBooster::ConvBooster()
    : GetBufferSize(NULL), Init(NULL), Forward(NULL)
{
}

//Conditional algo selecter
int ConvBooster::SelectAlgo(ConvParam* param)
{
    if (param->group == param->input_channels)
    {
        this->algo = DEPTHWISE;
    }
    else if (param->group == 1 && param->kernel_h == 3 && param->kernel_w == 3 && param->stride_h == 1 && param->stride_w == 1  && param->output_channels < 1024 && param->output_channels % 4 == 0)
    {
        this->algo = WINOGRADF63;
    }
    else if (param->group == 1 && param->kernel_w > 1 && param->kernel_h > 1)
    {
       this->algo = SGECONV;
    }
    else if (param->group == 1)
    {
       this->algo = IM2COL;
    }
    else
    {
        LOGE("Partial group conv is not yet supported. If you need it, try develop your own im2col method.");
        return -1;
    }
    return this->SetFuncs();
}

//Force algo selecter
int ConvBooster::ForceSelectAlgo(ConvAlgo algo)
{
    this->algo = algo;
    return this->SetFuncs();
}

int ConvBooster::SetFuncs()
{
    switch (this->algo)
    {
    case NAIVE:
        this->GetBufferSize = NAIVE_GetBufferSize;
        this->Init = NAIVE_Init;
        this->Forward = NAIVE_Forward;
        return 0;
    case IM2COL:
        this->GetBufferSize = IM2COL_GetBufferSize;
        this->Init = IM2COL_Init;
        this->Forward = IM2COL_Forward;
        return 0;
    case WINOGRADF63:
        this->GetBufferSize = WINOGRADF63_GetBufferSize;
        this->Init = WINOGRADF63_Init;
        this->Forward = WINOGRADF63_Forward;
        return 0;
    case DEPTHWISE:
        this->GetBufferSize = DEPTHWISE_GetBufferSize;
        this->Init = DEPTHWISE_Init;
        this->Forward = DEPTHWISE_Forward;
    default:
        LOGE("This algo is not supported on AVX2.");
        this->GetBufferSize = NULL;
        this->Init = NULL;
        this->Forward = NULL;
        return -1;
    }
}
}; // namespace booster