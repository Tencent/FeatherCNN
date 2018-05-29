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

#include <assert.h>
#include <arm_neon.h>
#include <string.h>

void fully_connected_inference_direct(const int input_size, const int output_size, const float *x, const float *y, float *z, const int num_threads)
{
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int i = 0; i < output_size; i++)
    {
        float sum = 0;
        for (int j = 0; j < input_size; j++)
            sum += x[j] * y[i * input_size + j];
        z[i] = sum;
    }
}

void fully_connected_transpose_inference_neon8(const int input_size, const int output_size, const float *x, const float *y, float *z, const int num_threads)
{
    assert(input_size % 8 == 0);
    assert(output_size % 8 == 0);
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int k = 0; k < output_size / 8; k++)
    {
        const float *yPtr = y + k * 8 * input_size;
        float32x4_t res = {0.0, 0.0, 0.0, 0.0};
        float32x4_t res1 = {0.0, 0.0, 0.0, 0.0};
        float32x4_t va, vb0, vb1, vb2, vb3, vb4, vb5, vb6, vb7;
        for (int i = 0; i < input_size; i += 4)
        {
            //          float32x4_t v1, v2;
            va = vld1q_f32(x + i);

            vb0 = vld1q_f32(yPtr);
            vb1 = vld1q_f32(yPtr + 4);
            vb2 = vld1q_f32(yPtr + 8);
            vb3 = vld1q_f32(yPtr + 12);
            vb4 = vld1q_f32(yPtr + 16);
            vb5 = vld1q_f32(yPtr + 20);
            vb6 = vld1q_f32(yPtr + 24);
            vb7 = vld1q_f32(yPtr + 28);

#if __aarch64__
            res = vfmaq_laneq_f32(res, vb0, va, 0);
            res1 = vfmaq_laneq_f32(res1, vb1, va, 0);
            res = vfmaq_laneq_f32(res, vb2, va, 1);
            res1 = vfmaq_laneq_f32(res1, vb3, va, 1);
            res = vfmaq_laneq_f32(res, vb4, va, 2);
            res1 = vfmaq_laneq_f32(res1, vb5, va, 2);
            res = vfmaq_laneq_f32(res, vb6, va, 3);
            res1 = vfmaq_laneq_f32(res1, vb7, va, 3);
#else
            res = vmlaq_f32(res, vb0, vld1q_dup_f32(x + i + 0));
            res1 = vmlaq_f32(res1, vb1, vld1q_dup_f32(x + i + 0));
            res = vmlaq_f32(res, vb2, vld1q_dup_f32(x + i + 1));
            res1 = vmlaq_f32(res1, vb3, vld1q_dup_f32(x + i + 1));
            res = vmlaq_f32(res, vb4, vld1q_dup_f32(x + i + 2));
            res1 = vmlaq_f32(res1, vb5, vld1q_dup_f32(x + i + 2));
            res = vmlaq_f32(res, vb6, vld1q_dup_f32(x + i + 3));
            res1 = vmlaq_f32(res1, vb7, vld1q_dup_f32(x + i + 3));
#endif

            yPtr += 32;
        }
        vst1q_f32((float32_t *)(z + 8 * k), res);
        vst1q_f32((float32_t *)(z + 8 * k + 4), res1);
    }
}

void fully_connected_inference_direct_BiasReLU(int input_size, int output_size, float *x, float *y, float *z, float* biasArr, int num_threads)
{
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int i = 0; i < output_size; i++)
    {
        float sum = 0.f;
        for (int j = 0; j < input_size; j++)
            sum += x[j] * y[i * input_size + j];

        sum += biasArr[i];
        if (sum < 0.f) sum = 0.f;
        z[i] = sum;
    }
}

void fully_connected_transpose_inference_neon8_BiasReLU(int input_size, int output_size, float *x, float *y, float *z, float* biasArr, int num_threads)
{
    assert(input_size % 8 == 0);
    assert(output_size % 8 == 0);
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int k = 0; k < output_size / 8; k++)
    {
        float *yPtr = y + k * 8 * input_size;
        const float32x4_t vzero = vdupq_n_f32(0.f);

        float32x4_t res  = vld1q_f32(biasArr + k * 8);
        float32x4_t res1 = vld1q_f32(biasArr + k * 8 + 4);

        float32x4_t va, vb0, vb1, vb2, vb3, vb4, vb5, vb6, vb7;
        for (int i = 0; i < input_size; i += 4)
        {
            va = vld1q_f32(x + i);

            vb0 = vld1q_f32(yPtr);
            vb1 = vld1q_f32(yPtr + 4);
            vb2 = vld1q_f32(yPtr + 8);
            vb3 = vld1q_f32(yPtr + 12);
            vb4 = vld1q_f32(yPtr + 16);
            vb5 = vld1q_f32(yPtr + 20);
            vb6 = vld1q_f32(yPtr + 24);
            vb7 = vld1q_f32(yPtr + 28);

#if __aarch64__
            res = vfmaq_laneq_f32(res, vb0, va, 0);
            res1 = vfmaq_laneq_f32(res1, vb1, va, 0);
            res = vfmaq_laneq_f32(res, vb2, va, 1);
            res1 = vfmaq_laneq_f32(res1, vb3, va, 1);
            res = vfmaq_laneq_f32(res, vb4, va, 2);
            res1 = vfmaq_laneq_f32(res1, vb5, va, 2);
            res = vfmaq_laneq_f32(res, vb6, va, 3);
            res1 = vfmaq_laneq_f32(res1, vb7, va, 3);
#else
            res = vmlaq_f32(res, vb0, vld1q_dup_f32(x + i + 0));
            res1 = vmlaq_f32(res1, vb1, vld1q_dup_f32(x + i + 0));
            res = vmlaq_f32(res, vb2, vld1q_dup_f32(x + i + 1));
            res1 = vmlaq_f32(res1, vb3, vld1q_dup_f32(x + i + 1));
            res = vmlaq_f32(res, vb4, vld1q_dup_f32(x + i + 2));
            res1 = vmlaq_f32(res1, vb5, vld1q_dup_f32(x + i + 2));
            res = vmlaq_f32(res, vb6, vld1q_dup_f32(x + i + 3));
            res1 = vmlaq_f32(res1, vb7, vld1q_dup_f32(x + i + 3));
#endif
            yPtr += 32;
        }

        //res  = vaddq_f32(res, vBias);
        //res1 = vaddq_f32(res, vBias1);

        res  = vmaxq_f32(res, vzero);
        res1 = vmaxq_f32(res1, vzero);

        vst1q_f32((float32_t *)(z + 8 * k), res);
        vst1q_f32((float32_t *)(z + 8 * k + 4), res1);
    }
}
/*
void fully_connected_transpose_inference_neon(int input_size, int output_size, float *x, float *y, float *z)
{
    assert(input_size %4==0);
    assert(output_size%4==0);
//#pragma omp parallel for num_threads(32) schedule(static)
    for(int k=0; k<output_size/4; k++)
    {
        float *yPtr = y + k*4*input_size;
        float32x4_t res = {0.0,0.0,0.0,0.0};

        for(int i=0; i<input_size; i+=4)
        {
            float32x4_t v1, v2;
            v2 = vld1q_f32(x + i);

#if __aarch64__
            v1 = vld1q_f32(yPtr);
            res = vfmaq_laneq_f32(res, v1, v2, 0);

            v1 = vld1q_f32(yPtr + 4);
            res = vfmaq_laneq_f32(res, v1, v2, 1);

            v1 = vld1q_f32(yPtr + 8);
            res = vfmaq_laneq_f32(res, v1, v2, 2);

            v1 = vld1q_f32(yPtr + 12);
            res = vfmaq_laneq_f32(res, v1, v2, 3);
#else
            v1 = vld1q_f32(yPtr);
            res = vmlaq_f32(res, v1, vld1q_dup_f32(x + i + 0));

            v1 = vld1q_f32(yPtr + 4);
            res = vmlaq_f32(res, v1, vld1q_dup_f32(x + i + 1));

            v1 = vld1q_f32(yPtr + 8);
            res = vmlaq_f32(res, v1, vld1q_dup_f32(x + i + 2));

            v1 = vld1q_f32(yPtr + 12);
            res = vmlaq_f32(res, v1, vld1q_dup_f32(x + i + 3));
#endif
            yPtr += 16;
        }
        vst1q_f32((float32_t *) (z+4*k), res);
    }
}
*/

void matrixTranspose(float* array, size_t m, size_t n, float *buffer)//  A[m][n] -> A[n][m]
{
    for (int i = 0; i < m; i++)    for (int j = 0; j < n; j++)
            buffer[j * m + i] = array[i * n + j];
    memcpy(array, buffer, m * n * sizeof(float));
}
