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

#include "generic_kernels.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <arm_neon.h>

#ifdef __APPLE__
#else
#include <omp.h>
#endif

void pad_input(float* padded, const float* input, const size_t input_channels, const size_t input_width, const size_t input_height, const size_t padding_left, const size_t padding_top, const size_t padding_right, const size_t padding_bottom)
{
    int paddedWidth  = (int)(input_width + padding_left + padding_right);
    int paddedHeight = (int)(input_height + padding_top + padding_bottom);
    memset(padded, 0, paddedWidth * paddedHeight * sizeof(float) * input_channels);
    for (int channelId = 0; channelId < input_channels; ++channelId)
    {
        //Beginning position in each input channel.
        float* padPtr = padded + channelId * paddedWidth * paddedHeight + paddedWidth * padding_top + padding_left;
        const float* inPtr  = input  + channelId * input_height * input_width;
        for (int i = 0; i < input_height; ++i)
        {
            memcpy(padPtr, inPtr, sizeof(float) * input_width);
            padPtr += paddedWidth;
            inPtr  += input_width;
        }
    }
}


/*
 * Elementwise operations
 */
void add_coeff(float* dst, float* A, float* coffA, float* B, float* coffB, size_t len, size_t num_threads)
{
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < len; i += 4)
    {
        float32x4_t vA = vld1q_f32(A + i);
        float32x4_t vB = vld1q_f32(B + i);
        float32x4_t vAc = vld1q_f32(coffA + i);
        float32x4_t vBc = vld1q_f32(coffB + i);
        vst1q_f32(dst + i, vaddq_f32(vmulq_f32(vA, vAc), vmulq_f32(vB, vBc)));
    }
    for (int i = len - len % 4; i < len; ++i)
    {
        dst[i] = A[i] * coffA[i] + B[i] * coffB[i];
    }
}
void add(float* dst, float* A, float* B, size_t len, size_t num_threads)
{
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < len; i += 4)
    {
        float32x4_t vA = vld1q_f32(A + i);
        float32x4_t vB = vld1q_f32(B + i);
        vst1q_f32(dst + i, vaddq_f32(vA, vB));
    }
    for (int i = len - len % 4; i < len; ++i)
    {
        dst[i] = A[i] + B[i];
    }
}

template<bool fuse_relu>
void add_relu(float* dst, const float* A, const float* B, const size_t len, const size_t num_threads)
{
    float32x4_t vZero = vdupq_n_f32(0.0f);
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < len; i += 4)
    {
        float32x4_t vA = vld1q_f32(A + i);
        float32x4_t vB = vld1q_f32(B + i);
        float32x4_t vS = vaddq_f32(vA, vB);
        if (fuse_relu)
        {
            vst1q_f32(dst + i, vmaxq_f32(vS, vZero));
        }
        else
        {
            vst1q_f32(dst + i, vS);
        }
    }
    for (int i = len - len % 4; i < len; ++i)
    {
        float S = A[i] + B[i];
        if (fuse_relu)
        {
            dst[i] = S > 0.0f ? S : 0.0f;
        }
        else
        {
            dst[i] = S;
        }
    }
}
template void add_relu<true>(float* dst, const float* A, const float* B, const size_t len, const size_t num_threads);
template void add_relu<false>(float* dst, const float* A, const float* B, const size_t len, const size_t num_threads);

void vsub(float* dst, float* A, float* B, size_t len, size_t num_threads)
{
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < len - 4; ++i)
    {
        float32x4_t vA = vld1q_f32(A + i);
        float32x4_t vB = vld1q_f32(B + i);
        vst1q_f32(dst + i, vsubq_f32(vA, vB));
    }
    for (int i = len - len % 4; i < len; ++i)
    {
        dst[i] = A[i] - B[i];
    }
}

void vmul(float* dst, float* A, float* B, size_t len, size_t num_threads)
{
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < len - 4; ++i)
    {
        float32x4_t vA = vld1q_f32(A + i);
        float32x4_t vB = vld1q_f32(B + i);
        vst1q_f32(dst + i, vmulq_f32(vA, vB));
    }
    for (int i = len - len % 4; i < len; ++i)
    {
        dst[i] = A[i] * B[i];
    }
}

template<bool has_bias>
void scale(const size_t channels, const size_t stride, const float* bias_data, const float* scale_data, const float* input, float* output, const size_t num_threads)
{
    #pragma omp parallel for num_threads(num_threads) schedule(guided)
    for (int i = 0; i < channels; i++)
    {
        float32x4_t v_scale = vdupq_n_f32(scale_data[i]);
        float32x4_t v_bias = vdupq_n_f32(0.f);
        float32x4_t v_zero = vdupq_n_f32(0.f);
        if (has_bias)
            v_bias = vdupq_n_f32(bias_data[i]);
        int j = 0;
        for (; j + 4 < stride; j += 4)
        {
            float32x4_t v_input = vld1q_f32(input + i * stride + j);
            float32x4_t v_out = vmulq_f32(v_input,  v_scale);
            if (has_bias)
                v_out = vaddq_f32(v_out, v_bias);
            vst1q_f32(output + i * stride + j, v_out);
        }
        for (; j < stride; j++)
        {
            float scale = input[i * stride + j] * scale_data[i];
            if (has_bias)
                scale = scale + bias_data[i];
            output[i * stride + j] = scale;
        }
    }
}
template void scale<true>(const size_t, const size_t, const float*, const float*, const float*, float*, const size_t);
template void scale<false>(const size_t, const size_t, const float*, const float*, const float*, float*, const size_t);

template<bool has_bias, bool has_scale, bool has_relu>
void batchnorm(const size_t channels, const size_t stride, const float* alpha, const float* beta, const float* bias_data, const float* scale_data, const float* input, float* output, const size_t num_threads)
{
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < channels; i++)
    {
        int j = 0;
#if 0
        float32x4_t v_alpha = vdupq_n_f32(alpha[i]);
        float32x4_t v_beta  = vdupq_n_f32(beta[i]);
        float32x4_t v_scale = vdupq_n_f32(0.f);
        float32x4_t v_bias = v_scale;
        float32x4_t v_zero = v_scale;

        if (has_scale) v_scale = vdupq_n_f32(scale_data[i]);
        if (has_bias) v_bias = vdupq_n_f32(bias_data[i]);

        const float *inputCur = input + i * stride;
        float *outputCur = output + i * stride;
        for (; j < stride; j += 4)
        {
            float32x4_t v_input = vld1q_f32(inputCur + j);
#ifdef __aarch64__
            float32x4_t v_norm = vfmaq_f32(v_alpha, v_beta, v_input);
#else
            float32x4_t v_norm = vmlaq_f32(v_alpha, v_beta, v_input);
#endif
            if (has_scale)
                v_norm = v_norm * v_scale;
            if (has_bias)
                v_norm = v_norm + v_bias;
            if (has_relu)
                v_norm = vmaxq_f32(v_norm, v_zero);
            vst1q_f32(outputCur + j, v_norm);
        }
        if (j > stride) j -= 4;
#endif
        for (; j < stride; j++)
        {
            float norm = beta[i] * input[i * stride + j] + alpha[i];
            if (has_scale)
                norm = norm * scale_data[i];
            if (has_bias)
                norm = norm + bias_data[i];
            if (has_relu)
                norm = (norm > 0) ? norm : 0;
            output[i * stride + j] = norm;
        }
    }
}

template void batchnorm<true, true, true>(const size_t, const size_t, const float*, const float*, const float*, const float*, const float*, float*, const size_t);
template void batchnorm<false, true, true>(const size_t, const size_t, const float*, const float*, const float*, const float*, const float*, float*, const size_t);
template void batchnorm<true, false, true>(const size_t, const size_t, const float*, const float*, const float*, const float*, const float*, float*, const size_t);
template void batchnorm<true, true, false>(const size_t, const size_t, const float*, const float*, const float*, const float*, const float*, float*, const size_t);
template void batchnorm<true, false, false>(const size_t, const size_t, const float*, const float*, const float*, const float*, const float*, float*, const size_t);
template void batchnorm<false, true, false>(const size_t, const size_t, const float*, const float*, const float*, const float*, const float*, float*, const size_t);
template void batchnorm<false, false, true>(const size_t, const size_t, const float*, const float*, const float*, const float*, const float*, float*, const size_t);
template void batchnorm<false, false, false>(const size_t, const size_t, const float*, const float*, const float*, const float*, const float*, float*, const size_t);

void softmax(float* input, float n)
{
    float sum = 0;
    for (int i = 0; i < n; i++)    sum += exp(input[i]);
    for (int i = 0; i < n; i++)    input[i] = exp(input[i]) / sum;
}

void naive_gemm(int M, int N, int L, float *A, float *B, float *C)
{
//    matrixTranspose(B, N, L);
    for (int i = 0; i < M; i++)  for (int j = 0; j < L; j++)  C[i * L + j] = 0;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < L; j++)
        {
            for (int k = 0; k < N; k++)
                C[i * L + j] += A[i * N + k] * B[k * L + j];
        }
    }
}

void relu(float* arr, int len)
{
    for (int i = 0; i < len; i++)
        if (arr[i] < 0)
            arr[i] = 0;
}

//The bias ReLU function is strangely faster than the basic relu.
void biasRelu(float* arr, int len, float bias)
{
    for (int i = 0; i < len; i++)
    {
        arr[i] += bias;
        if (arr[i] < 0)
            arr[i] = 0;
    }
}

void reluVec(float* arr, int len)
{
    int aLen = len - len % 4;
    float32x4_t vzero = {0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < aLen; i += 4)
    {
        float32x4_t vl = vld1q_f32(arr + i);
        float32x4_t vs = vmaxq_f32(vl, vzero);
        vst1q_f32(arr + i, vs);
    }
    for (int i = aLen; i < len; i++)
        if (arr[i] < 0)
            arr[i] = 0;
}

void biasVec(float* arr, int len, float bias)
{
    int aLen = len - len % 4;
    float32x4_t vzero = {0.0, 0.0, 0.0, 0.0};
    float32x4_t vbias = vdupq_n_f32(bias);
    for (int i = 0; i < aLen; i += 4)
    {
        float32x4_t vl = vld1q_f32(arr + i);
        vl = vaddq_f32(vl, vbias);
        vst1q_f32(arr + i, vl);
    }
    for (int i = aLen; i < len; i++)
    {
        arr[i] += bias;
    }
}
void biasReluVec(float* arr, int len, float bias)
{
    int aLen = len - len % 4;
    float32x4_t vzero = {0.0, 0.0, 0.0, 0.0};
    float32x4_t vbias = vdupq_n_f32(bias);
    for (int i = 0; i < aLen; i += 4)
    {
        float32x4_t vl = vld1q_f32(arr + i);
        vl = vaddq_f32(vl, vbias);
        float32x4_t vs = vmaxq_f32(vl, vzero);
        vst1q_f32(arr + i, vs);
    }
    for (int i = aLen; i < len; i++)
    {
        arr[i] += bias;
        if (arr[i] < 0)
            arr[i] = 0;
    }
}

void biasReluVecOpenmp(float* arr, int len, float bias, int nThreads)
{
    //Don't use too many threads.
    nThreads = (nThreads > 4) ? 4 : nThreads;
    int aLen = len - len % 16;
    float32x4_t vzero = {0.0, 0.0, 0.0, 0.0};
    float32x4_t vbias = vdupq_n_f32(bias);
    #pragma omp parallel for num_threads(nThreads)
    for (int i = 0; i < aLen; i += 16)
    {
        float32x4_t v0 = vld1q_f32(arr + i);
        float32x4_t v1 = vld1q_f32(arr + i + 4);
        float32x4_t v2 = vld1q_f32(arr + i + 8);
        float32x4_t v3 = vld1q_f32(arr + i + 12);
        v0 = vaddq_f32(v0, vbias);
        v1 = vaddq_f32(v1, vbias);
        v2 = vaddq_f32(v2, vbias);
        v3 = vaddq_f32(v3, vbias);
        vst1q_f32(arr + i, vmaxq_f32(v0, vzero));
        vst1q_f32(arr + i + 4, vmaxq_f32(v1, vzero));
        vst1q_f32(arr + i + 8, vmaxq_f32(v2, vzero));
        vst1q_f32(arr + i + 12, vmaxq_f32(v3, vzero));
    }
    for (int i = aLen; i < len; i++)
    {
        arr[i] += bias;
        if (arr[i] < 0)
            arr[i] = 0;
    }
}
void biasVecOpenmp(float* arr, int len, float bias, int nThreads)
{
    //Don't use too many threads.
    nThreads = (nThreads > 4) ? 4 : nThreads;
    int aLen = len - len % 16;
    float32x4_t vzero = {0.0, 0.0, 0.0, 0.0};
    float32x4_t vbias = vdupq_n_f32(bias);
    #pragma omp parallel for num_threads(nThreads)
    for (int i = 0; i < aLen; i += 16)
    {
        float32x4_t v0 = vld1q_f32(arr + i);
        float32x4_t v1 = vld1q_f32(arr + i + 4);
        float32x4_t v2 = vld1q_f32(arr + i + 8);
        float32x4_t v3 = vld1q_f32(arr + i + 12);
        v0 = vaddq_f32(v0, vbias);
        v1 = vaddq_f32(v1, vbias);
        v2 = vaddq_f32(v2, vbias);
        v3 = vaddq_f32(v3, vbias);
        vst1q_f32(arr + i, v0);
        vst1q_f32(arr + i + 4,  v1);
        vst1q_f32(arr + i + 8,  v2);
        vst1q_f32(arr + i + 12, v3);
    }
    for (int i = aLen; i < len; i++)
        arr[i] += bias;
}
void reluVecOpenmp(float* arr, int len, int nThreads)
{
    //Don't use too many threads.
    nThreads = (nThreads > 4) ? 4 : nThreads;
    int aLen = len - len % 16;
    float32x4_t vzero = {0.0, 0.0, 0.0, 0.0};
    #pragma omp parallel for num_threads(nThreads)
    for (int i = 0; i < aLen; i += 16)
    {
        float32x4_t v0 = vld1q_f32(arr + i);
        float32x4_t v1 = vld1q_f32(arr + i + 4);
        float32x4_t v2 = vld1q_f32(arr + i + 8);
        float32x4_t v3 = vld1q_f32(arr + i + 12);
        vst1q_f32(arr + i, vmaxq_f32(v0, vzero));
        vst1q_f32(arr + i + 4, vmaxq_f32(v1, vzero));
        vst1q_f32(arr + i + 8, vmaxq_f32(v2, vzero));
        vst1q_f32(arr + i + 12, vmaxq_f32(v3, vzero));
    }
    for (int i = aLen; i < len; i++)
        if (arr[i] < 0)    arr[i] = 0;
}
