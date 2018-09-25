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
#include <booster/generic_kernels.h>

#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <immintrin.h>

#ifdef __APPLE__
#else
#include <omp.h>
#endif

namespace booster{
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

void im2col(ConvParam *conv_param, float *img_buffer, float *input)
{
    const int stride = conv_param->kernel_h * conv_param->kernel_w * conv_param->output_h * conv_param->output_w;
    float *ret = img_buffer;
    // #pragma omp parallel for num_threads(num_threads)
    for (int k = 0; k < conv_param->input_channels; k++)
    {
        int retID = stride * k;
        for (int u = 0; u < conv_param->kernel_h; u++)
        {
            for (int v = 0; v < conv_param->kernel_w; v++)
            {
                for (int i = 0; i < conv_param->output_h; i++)
                {
                    for (int j = 0; j < conv_param->output_w; j++)
                    {
                        //calculate each row
                        int row = u - conv_param->pad_top + i * conv_param->stride_h;
                        int col = v - conv_param->pad_left + j * conv_param->stride_w;
                        //printf("row %d, col %d\n", row, col);
                        if (row < 0 || row >= conv_param->input_h || col < 0 || col >= conv_param->input_w)
                        {
                            ret[retID] = 0;
                        }
                        else
                        {
                            size_t index = k * conv_param->input_w * conv_param->input_h + row * conv_param->input_w + col; //(i+u)*input_width+j+v;
                            ret[retID] = input[index];
                        }
                        retID++;
                    }
                }
            }
        }
    }
}

void naive_sgemm(int M, int N, int L, float *A, float *B, float *C)
{
    for (int i = 0; i < M; ++i) //loop over rows in C
    {
        for (int j = 0; j < N; ++j) //loop over columns in C
        {
            float sigma = 0;
            for (int k = 0; k < L; ++k)
            {
                sigma += A[i * L + k] * B[k * N + j];
            }
            C[i * N + j] = sigma;
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
        __m128 vA = _mm_load_ps(A + i);
        __m128 vB = _mm_load_ps(B + i);
        __m128 vAc = _mm_load_ps(coffA + i);
        __m128 vBc = _mm_load_ps(coffB + i);
        _mm_store_ps(dst + i, _mm_add_ps(_mm_mul_ps(vA, vAc), _mm_mul_ps(vB, vBc)));
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
        __m128 vA = _mm_load_ps(A + i);
        __m128 vB = _mm_load_ps(B + i);
        _mm_store_ps(dst + i, _mm_add_ps(vA, vB));
    }
    for (int i = len - len % 4; i < len; ++i)
    {
        dst[i] = A[i] + B[i];
    }
}

template<bool fuse_relu>
void add_relu(float* dst, const float* A, const float* B, const size_t len, const size_t num_threads)
{
    __m128 vZero = _mm_set1_ps(0.0f);
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < len; i += 4)
    {
        __m128 vA = _mm_load_ps(A + i);
        __m128 vB = _mm_load_ps(B + i);
        __m128 vS = _mm_add_ps(vA, vB);
        if (fuse_relu)
        {
            _mm_store_ps(dst + i, _mm_max_ps(vS, vZero));
        }
        else
        {
            _mm_store_ps(dst + i, vS);
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
        __m128 vA = _mm_load_ps(A + i);
        __m128 vB = _mm_load_ps(B + i);
        _mm_store_ps(dst + i, _mm_sub_ps(vA, vB));
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
        __m128 vA = _mm_load_ps(A + i);
        __m128 vB = _mm_load_ps(B + i);
        _mm_store_ps(dst + i, _mm_mul_ps(vA, vB));
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
        __m128 v_scale = _mm_set1_ps(scale_data[i]);
        __m128 v_bias = _mm_set1_ps(0.f);
        __m128 v_zero = _mm_set1_ps(0.f);
        if (has_bias)
            v_bias = _mm_set1_ps(bias_data[i]);
        int j = 0;
        for (; j + 4 < stride; j += 4)
        {
            __m128 v_input = _mm_load_ps(input + i * stride + j);
            __m128 v_out = _mm_mul_ps(v_input,  v_scale);
            if (has_bias)
                v_out = _mm_add_ps(v_out, v_bias);
            _mm_store_ps(output + i * stride + j, v_out);
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
        __m128 v_alpha = _mm_set1_ps(alpha[i]);
        __m128 v_beta  = _mm_set1_ps(beta[i]);
        __m128 v_scale = _mm_set1_ps(0.f);
        __m128 v_bias = v_scale;
        __m128 v_zero = v_scale;

        if (has_scale) v_scale = _mm_set1_ps(scale_data[i]);
        if (has_bias) v_bias = _mm_set1_ps(bias_data[i]);

        const float *inputCur = input + i * stride;
        float *outputCur = output + i * stride;
        int j = 0;
        for (; j < ((int)stride) - 4; j += 4)
        {
            __m128 v_input = _mm_loadu_ps(inputCur + j);
            __m128 v_norm = _mm_fmadd_ps(v_beta, v_input, v_alpha);
            if (has_scale)
                v_norm = v_norm * v_scale;
            if (has_bias)
                v_norm = v_norm + v_bias;
            if (has_relu)
                v_norm = _mm_max_ps(v_norm, v_zero);
            _mm_storeu_ps(outputCur + j, v_norm);
        }
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
    __m128 vzero = {0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < aLen; i += 4)
    {
        __m128 vl = _mm_load_ps(arr + i);
        __m128 vs = _mm_max_ps(vl, vzero);
        _mm_store_ps(arr + i, vs);
    }
    for (int i = aLen; i < len; i++)
        if (arr[i] < 0)
            arr[i] = 0;
}

void biasVec(float* arr, int len, float bias)
{
    int aLen = len - len % 4;
    __m128 vzero = {0.0, 0.0, 0.0, 0.0};
    __m128 vbias = _mm_set1_ps(bias);
    for (int i = 0; i < aLen; i += 4)
    {
        __m128 vl = _mm_load_ps(arr + i);
        vl = _mm_add_ps(vl, vbias);
        _mm_store_ps(arr + i, vl);
    }
    for (int i = aLen; i < len; i++)
    {
        arr[i] += bias;
    }
}
void biasReluVec(float* arr, int len, float bias)
{
    int aLen = len - len % 4;
    __m128 vzero = {0.0, 0.0, 0.0, 0.0};
    __m128 vbias = _mm_set1_ps(bias);
    for (int i = 0; i < aLen; i += 4)
    {
        __m128 vl = _mm_load_ps(arr + i);
        vl = _mm_add_ps(vl, vbias);
        __m128 vs = _mm_max_ps(vl, vzero);
        _mm_store_ps(arr + i, vs);
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
    __m128 vzero = {0.0, 0.0, 0.0, 0.0};
    __m128 vbias = _mm_set1_ps(bias);
    #pragma omp parallel for num_threads(nThreads)
    for (int i = 0; i < aLen; i += 16)
    {
        __m128 v0 = _mm_load_ps(arr + i);
        __m128 v1 = _mm_load_ps(arr + i + 4);
        __m128 v2 = _mm_load_ps(arr + i + 8);
        __m128 v3 = _mm_load_ps(arr + i + 12);
        v0 = _mm_add_ps(v0, vbias);
        v1 = _mm_add_ps(v1, vbias);
        v2 = _mm_add_ps(v2, vbias);
        v3 = _mm_add_ps(v3, vbias);
        _mm_store_ps(arr + i, _mm_max_ps(v0, vzero));
        _mm_store_ps(arr + i + 4, _mm_max_ps(v1, vzero));
        _mm_store_ps(arr + i + 8, _mm_max_ps(v2, vzero));
        _mm_store_ps(arr + i + 12, _mm_max_ps(v3, vzero));
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
    __m128 vzero = {0.0, 0.0, 0.0, 0.0};
    __m128 vbias = _mm_set1_ps(bias);
    #pragma omp parallel for num_threads(nThreads)
    for (int i = 0; i < aLen; i += 16)
    {
        __m128 v0 = _mm_load_ps(arr + i);
        __m128 v1 = _mm_load_ps(arr + i + 4);
        __m128 v2 = _mm_load_ps(arr + i + 8);
        __m128 v3 = _mm_load_ps(arr + i + 12);
        v0 = _mm_add_ps(v0, vbias);
        v1 = _mm_add_ps(v1, vbias);
        v2 = _mm_add_ps(v2, vbias);
        v3 = _mm_add_ps(v3, vbias);
        _mm_store_ps(arr + i, v0);
        _mm_store_ps(arr + i + 4,  v1);
        _mm_store_ps(arr + i + 8,  v2);
        _mm_store_ps(arr + i + 12, v3);
    }
    for (int i = aLen; i < len; i++)
        arr[i] += bias;
}
void reluVecOpenmp(float* arr, int len, int nThreads)
{
    //Don't use too many threads.
    nThreads = (nThreads > 4) ? 4 : nThreads;
    int aLen = len - len % 16;
    __m128 vzero = {0.0, 0.0, 0.0, 0.0};
    #pragma omp parallel for num_threads(nThreads)
    for (int i = 0; i < aLen; i += 16)
    {
        __m128 v0 = _mm_load_ps(arr + i);
        __m128 v1 = _mm_load_ps(arr + i + 4);
        __m128 v2 = _mm_load_ps(arr + i + 8);
        __m128 v3 = _mm_load_ps(arr + i + 12);
        _mm_store_ps(arr + i, _mm_max_ps(v0, vzero));
        _mm_store_ps(arr + i + 4, _mm_max_ps(v1, vzero));
        _mm_store_ps(arr + i + 8, _mm_max_ps(v2, vzero));
        _mm_store_ps(arr + i + 12, _mm_max_ps(v3, vzero));
    }
    for (int i = aLen; i < len; i++)
        if (arr[i] < 0)    arr[i] = 0;
}
};
