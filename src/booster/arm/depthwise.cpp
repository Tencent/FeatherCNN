//Tencent is pleased to support the open source community by making FeatherCNN available.

//Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.

//Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
//in compliance with the License. You may obtain a copy of the License at
//
//https://opensource.org/licenses/BSD-3-Clause
//
//Unless required by applicable law or agreed to in writing, software distributed
//under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//CONDITIONS OF ANY KIND, either express or implied. See the License for the
//specific language governing permissions and limitations under the License.

#include <booster/depthwise.h>
#include <booster/helper.h>

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <arm_neon.h>

#ifdef __APPLE__
#else
#include <omp.h>
#endif
/*
 * For global depthwise convolution
 */

template <bool fuseBias, bool fuseRelu>
void globalDwConv(float *output, const float *input, int input_channels, int inw, int inh, float *kernel, int group, int nThreads, float *bias_arr)
{
    assert(group > 0 || input_channels % group == 0);
    int step = inw * inh;
    int block = input_channels / group;
    int groupKernelSize = inw * inh * group;

    for (int i = 0; i < input_channels; i++)
    {
        int k = i / group, u = i % group;
        output[i] = 0;
        for (int j = 0; j < step; j++)
        {
            output[i] += input[i * step + j] * kernel[k * groupKernelSize + u * step + j];
        }
        if (fuseBias)
        {
            output[i] += bias_arr[i];
        }
        if (fuseRelu)
        {
            output[i] = (output[i] > 0.f) ? output[i] : 0.f;
        }
    }

    /*
    int kw = inw, kh = inh;
    int width = kw * kh;
    int widthAligned = width & 0xFFFFFFFC;
    int widthRem = width & 0x03; // int widthRem = width & 0x11;
    int height = group;
    int heightAligned = group & 0xFFFFFFFC;
    int heightRem = height & 0x03; // int heightRem = height & 0x11;
    float ext[8];
    for(int i = 0; i < heightAligned; i += 4)
    {
        float32x4_t sum = vdupq_n_f32(0.f);
        float* p0 = const_cast<float *>(input) + width * i;
        float* p1 = p0 + width;
        float* p2 = p1 + width;
        float* p3 = p2 + width;
        float* k0 = kernel + width * i;
        float* k1 = k0 + width;
        float* k2 = k1 + width;
        float* k3 = k2 + width;

        for(int j = 0; j < widthAligned; j += 4)
        {
            float32x4_t v0 = vld1q_f32(p0);
            p0 += 4;
            float32x4_t v1 = vld1q_f32(p1);
            p1 += 4;
            float32x4_t v2 = vld1q_f32(p2);
            p2 += 4;
            float32x4_t v3 = vld1q_f32(p3);
            p3 += 4;

            float32x4_t r0 = vld1q_f32(k0);
            k0 += 4;
            float32x4_t r1 = vld1q_f32(k1);
            k1 += 4;
            float32x4_t r2 = vld1q_f32(k2);
            k2 += 4;
            float32x4_t r3 = vld1q_f32(k3);
            k3 += 4;

            float32x4x2_t row01 = vtrnq_f32(v0, v1);
            float32x4x2_t row23 = vtrnq_f32(v2, v3);

    //           * row0 = ( x00 x10 x20 x30 )
    //           * row1 = ( x01 x11 x21 x31 )
    //           * row2 = ( x02 x12 x22 x32 )
    //           * row3 = ( x03 x13 x23 x33 )

            v0 = vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0]));
            v1 = vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1]));
            v2 = vcombine_f32(vget_high_f32(row01.val[0]), vget_high_f32(row23.val[0]));
            v3 = vcombine_f32(vget_high_f32(row01.val[1]), vget_high_f32(row23.val[1]));
            row01 = vtrnq_f32(r0, r1);
            row23 = vtrnq_f32(r2, r3);
            r0 = vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0]));
            r1 = vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1]));
            r2 = vcombine_f32(vget_high_f32(row01.val[0]), vget_high_f32(row23.val[0]));
            r3 = vcombine_f32(vget_high_f32(row01.val[1]), vget_high_f32(row23.val[1]));
    #ifdef __aarch64__
            sum = vfmaq_f32(sum, v0, r0);
            sum = vfmaq_f32(sum, v1, r1);
            sum = vfmaq_f32(sum, v2, r2);
            sum = vfmaq_f32(sum, v3, r3);
    #else
            sum = vmlaq_f32(sum, v0, r0);
            sum = vmlaq_f32(sum, v1, r1);
            sum = vmlaq_f32(sum, v2, r2);
            sum = vmlaq_f32(sum, v3, r3);
    #endif
        }
        if(widthRem){
            for(int j = 0; j < widthRem; ++j)
            {
                ext[0] = p0[j];
                ext[1] = p1[j];
                ext[2] = p2[j];
                ext[3] = p3[j];
                ext[4] = k0[j];
                ext[5] = k1[j];
                ext[6] = k2[j];
                ext[7] = k3[j];
    #ifdef __aarch64__
                sum = vfmaq_f32(sum, vld1q_f32(ext + 4), vld1q_f32(ext));
    #else
                sum = vmlaq_f32(sum, vld1q_f32(ext + 4), vld1q_f32(ext));
    #endif
            }
        }
        vst1q_f32(output + i, sum);
    }
    for(int i = heightAligned; i < height; ++i)
    {
        float* p = const_cast<float *>(input) + i * width;
        float* k = kernel + i * width;
        float sum = 0.f;
        for(int j = 0; j < width; ++j)
        {
            sum += p[j] * k[j];
        }
        output[i] = sum; // output[heightAligned + i] = sum;
    }
    */
}

//optimized by xningwang on 22/12/2017
//Add layer fusion by haidonglan on 24/07/2018
template <bool fuseBias, bool fuseRelu>
void dwConvs1(float *output, float *input, int inw, int inh, int stridew, int strideh, float *kernel, int kw, int kh, int group, int nThreads, float *bias_arr)
{
    int outw = (inw - kw + 1) / stridew; //for strided case in odd dimensions, should take the floor value as output dim.
    int outh = (inh - kh + 1) / strideh;
    float32x4_t vZero = vdupq_n_f32(0.f);
    float32x4_t vBias;
    #pragma omp parallel for num_threads(nThreads) schedule(static)
    for (int g = 0; g < group; ++g)
    {

        float *kp = kernel + 9 * g;
        if (fuseBias)
            vBias = vld1q_dup_f32(bias_arr + g);
        float32x4_t k0 = vld1q_dup_f32(kp);
        float32x4_t k1 = vld1q_dup_f32(kp + 1);
        float32x4_t k2 = vld1q_dup_f32(kp + 2);
        float32x4_t k3 = vld1q_dup_f32(kp + 3);
        float32x4_t k4 = vld1q_dup_f32(kp + 4);
        float32x4_t k5 = vld1q_dup_f32(kp + 5);
        float32x4_t k6 = vld1q_dup_f32(kp + 6);
        float32x4_t k7 = vld1q_dup_f32(kp + 7);
        float32x4_t k8 = vld1q_dup_f32(kp + 8);

        float32x4_t k0123, k3456, k6789;
        float *outg = output + g * outw * outh;
        float *ing = input + g * inw * inh;

        float32x4_t sum1, sum2, sum3, sum4;
        int i = 0;
        for (; i + 1 < outh; i += 2)
        {
#ifdef __aarch64__
            int nout = outw >> 3;
            int remain = outw & 7;
            float *r0 = ing + inw * i;
            float *r1 = ing + inw * (i + 1);
            float *r2 = ing + inw * (i + 2);
            float *r3 = ing + inw * (i + 3);

            float *og = outg + outw * i;
            float *og3 = og + outw;

            for (; nout > 0; nout--)
            {
                float32x4_t r00 = vld1q_f32(r0);
                float32x4_t r00n = vld1q_f32(r0 + 4);
                float32x4_t r00nn = vld1q_f32(r0 + 8);
                float32x4_t r01 = vextq_f32(r00, r00n, 1);
                float32x4_t r02 = vextq_f32(r00, r00n, 2);

                float32x4_t r10 = vld1q_f32(r1);
                float32x4_t r10n = vld1q_f32(r1 + 4);
                float32x4_t r10nn = vld1q_f32(r1 + 8);
                float32x4_t r11 = vextq_f32(r10, r10n, 1);
                float32x4_t r12 = vextq_f32(r10, r10n, 2);

                float32x4_t r20 = vld1q_f32(r2);
                float32x4_t r20n = vld1q_f32(r2 + 4);
                float32x4_t r20nn = vld1q_f32(r2 + 8);
                float32x4_t r21 = vextq_f32(r20, r20n, 1);
                float32x4_t r22 = vextq_f32(r20, r20n, 2);

                float32x4_t r30 = vld1q_f32(r3);
                float32x4_t r30n = vld1q_f32(r3 + 4);
                float32x4_t r30nn = vld1q_f32(r3 + 8);
                float32x4_t r31 = vextq_f32(r30, r30n, 1);
                float32x4_t r32 = vextq_f32(r30, r30n, 2);

                sum1 = vmulq_f32(r00, k0);
                sum1 = vfmaq_f32(sum1, r01, k1);
                sum1 = vfmaq_f32(sum1, r02, k2);
                sum1 = vfmaq_f32(sum1, r10, k3);
                sum1 = vfmaq_f32(sum1, r11, k4);
                sum1 = vfmaq_f32(sum1, r12, k5);
                sum1 = vfmaq_f32(sum1, r20, k6);
                sum1 = vfmaq_f32(sum1, r21, k7);
                sum1 = vfmaq_f32(sum1, r22, k8);

                sum2 = vmulq_f32(r10, k0);
                sum2 = vfmaq_f32(sum2, r11, k1);
                sum2 = vfmaq_f32(sum2, r12, k2);
                sum2 = vfmaq_f32(sum2, r20, k3);
                sum2 = vfmaq_f32(sum2, r21, k4);
                sum2 = vfmaq_f32(sum2, r22, k5);
                sum2 = vfmaq_f32(sum2, r30, k6);
                sum2 = vfmaq_f32(sum2, r31, k7);
                sum2 = vfmaq_f32(sum2, r32, k8);

                r01 = vextq_f32(r00n, r00nn, 1);
                r02 = vextq_f32(r00n, r00nn, 2);
                r11 = vextq_f32(r10n, r10nn, 1);
                r12 = vextq_f32(r10n, r10nn, 2);
                r21 = vextq_f32(r20n, r20nn, 1);
                r22 = vextq_f32(r20n, r20nn, 2);
                r31 = vextq_f32(r30n, r30nn, 1);
                r32 = vextq_f32(r30n, r30nn, 2);

                sum3 = vmulq_f32(r00n, k0);
                sum3 = vfmaq_f32(sum3, r01, k1);
                sum3 = vfmaq_f32(sum3, r02, k2);
                sum3 = vfmaq_f32(sum3, r10n, k3);
                sum3 = vfmaq_f32(sum3, r11, k4);
                sum3 = vfmaq_f32(sum3, r12, k5);
                sum3 = vfmaq_f32(sum3, r20n, k6);
                sum3 = vfmaq_f32(sum3, r21, k7);
                sum3 = vfmaq_f32(sum3, r22, k8);

                sum4 = vmulq_f32(r10n, k0);
                sum4 = vfmaq_f32(sum4, r11, k1);
                sum4 = vfmaq_f32(sum4, r12, k2);
                sum4 = vfmaq_f32(sum4, r20n, k3);
                sum4 = vfmaq_f32(sum4, r21, k4);
                sum4 = vfmaq_f32(sum4, r22, k5);
                sum4 = vfmaq_f32(sum4, r30n, k6);
                sum4 = vfmaq_f32(sum4, r31, k7);
                sum4 = vfmaq_f32(sum4, r32, k8);
                if (fuseBias)
                {
                    sum1 = vaddq_f32(sum1, vBias);
                    sum2 = vaddq_f32(sum2, vBias);
                    sum3 = vaddq_f32(sum3, vBias);
                    sum4 = vaddq_f32(sum4, vBias);
                }
                if (fuseRelu)
                {
                    sum1 = vmaxq_f32(sum1, vZero);
                    sum2 = vmaxq_f32(sum2, vZero);
                    sum3 = vmaxq_f32(sum3, vZero);
                    sum4 = vmaxq_f32(sum4, vZero);
                }
                vst1q_f32(og, sum1);
                vst1q_f32(og + 4, sum3);
                vst1q_f32(og3, sum2);
                vst1q_f32(og3 + 4, sum4);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                og += 8;
                og3 += 8;
            }
            //compute 2 * 4 in case of remain > = 4, eg: 4 5 6 7
            for (; remain - 3 > 0; remain -= 4)
            {
                float32x4_t r00 = vld1q_f32(r0);
                float32x4_t r00n = vld1q_f32(r0 + 4);
                float32x4_t r01 = vextq_f32(r00, r00n, 1);
                float32x4_t r02 = vextq_f32(r00, r00n, 2);

                float32x4_t r10 = vld1q_f32(r1);
                float32x4_t r10n = vld1q_f32(r1 + 4);
                float32x4_t r11 = vextq_f32(r10, r10n, 1);
                float32x4_t r12 = vextq_f32(r10, r10n, 2);

                float32x4_t r20 = vld1q_f32(r2);
                float32x4_t r20n = vld1q_f32(r2 + 4);
                float32x4_t r21 = vextq_f32(r20, r20n, 1);
                float32x4_t r22 = vextq_f32(r20, r20n, 2);

                float32x4_t r30 = vld1q_f32(r3);
                float32x4_t r30n = vld1q_f32(r3 + 4);
                float32x4_t r31 = vextq_f32(r30, r30n, 1);
                float32x4_t r32 = vextq_f32(r30, r30n, 2);

                sum1 = vmulq_f32(r00, k0);
                sum1 = vfmaq_f32(sum1, r01, k1);
                sum1 = vfmaq_f32(sum1, r02, k2);
                sum1 = vfmaq_f32(sum1, r10, k3);
                sum1 = vfmaq_f32(sum1, r11, k4);
                sum1 = vfmaq_f32(sum1, r12, k5);
                sum1 = vfmaq_f32(sum1, r20, k6);
                sum1 = vfmaq_f32(sum1, r21, k7);
                sum1 = vfmaq_f32(sum1, r22, k8);

                sum2 = vmulq_f32(r10, k0);
                sum2 = vfmaq_f32(sum2, r11, k1);
                sum2 = vfmaq_f32(sum2, r12, k2);
                sum2 = vfmaq_f32(sum2, r20, k3);
                sum2 = vfmaq_f32(sum2, r21, k4);
                sum2 = vfmaq_f32(sum2, r22, k5);
                sum2 = vfmaq_f32(sum2, r30, k6);
                sum2 = vfmaq_f32(sum2, r31, k7);
                sum2 = vfmaq_f32(sum2, r32, k8);

                if (fuseBias)
                {
                    sum1 = vaddq_f32(sum1, vBias);
                    sum2 = vaddq_f32(sum2, vBias);
                }
                if (fuseRelu)
                {
                    sum1 = vmaxq_f32(sum1, vZero);
                    sum2 = vmaxq_f32(sum2, vZero);
                }
                vst1q_f32(og, sum1);
                vst1q_f32(og3, sum2);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                og += 4;
                og3 += 4;
            }

            //the columns remained every 2 rows
            k0123 = vld1q_f32(kp);
            k3456 = vld1q_f32(kp + 3);
            k6789 = vld1q_f32(kp + 6);
            for (; remain > 0; remain--)
            {
                float32x4_t r00 = vld1q_f32(r0);
                float32x4_t r10 = vld1q_f32(r1);
                float32x4_t r20 = vld1q_f32(r2);
                float32x4_t r30 = vld1q_f32(r3);

                float32x4_t sum1 = vmulq_f32(r00, k0123);
                sum1 = vfmaq_f32(sum1, r10, k3456);
                sum1 = vfmaq_f32(sum1, r20, k6789);

                float32x4_t sum2 = vmulq_f32(r10, k0123);
                sum2 = vfmaq_f32(sum2, r20, k3456);
                sum2 = vfmaq_f32(sum2, r30, k6789);

                sum1 = vsetq_lane_f32(0.0f, sum1, 3); //set third value of og to 0
                float vsum = vaddvq_f32(sum1);        //accumulate the first three value of og
                sum2 = vsetq_lane_f32(0.0f, sum2, 3); //set third value of og to 0
                float vsum2 = vaddvq_f32(sum2);       //accumulate the first three value of og
                if (fuseBias)
                {
                    vsum += bias_arr[g];
                    vsum2 += bias_arr[g];
                }
                if (fuseRelu)
                {
                    vsum = (vsum > 0.f) ? vsum : 0.f;
                    vsum2 = (vsum2 > 0.f) ? vsum2 : 0.f;
                }
                *og = vsum;
                *og3 = vsum2;

                r0++;
                r1++;
                r2++;
                r3++;
                og++;
                og3++;
            }

#else //ARMv7, 2 * 4
            int nout = outw >> 2; //outw / 4, compute 4 cols per time
            int remain = outw & 3;
            float *r0 = ing + inw * i;
            float *r1 = ing + inw * (i + 1);
            float *r2 = ing + inw * (i + 2);
            float *r3 = ing + inw * (i + 3);

            float *og = outg + outw * i;
            float *og3 = og + outw;

            for (; nout > 0; nout--)
            {
                float32x4_t r00 = vld1q_f32(r0);
                float32x4_t r00n = vld1q_f32(r0 + 4);
                float32x4_t r01 = vextq_f32(r00, r00n, 1);
                float32x4_t r02 = vextq_f32(r00, r00n, 2);

                float32x4_t r10 = vld1q_f32(r1);
                float32x4_t r10n = vld1q_f32(r1 + 4);
                float32x4_t r11 = vextq_f32(r10, r10n, 1);
                float32x4_t r12 = vextq_f32(r10, r10n, 2);

                float32x4_t r20 = vld1q_f32(r2);
                float32x4_t r20n = vld1q_f32(r2 + 4);
                float32x4_t r21 = vextq_f32(r20, r20n, 1);
                float32x4_t r22 = vextq_f32(r20, r20n, 2);

                float32x4_t r30 = vld1q_f32(r3);
                float32x4_t r30n = vld1q_f32(r3 + 4);
                float32x4_t r31 = vextq_f32(r30, r30n, 1);
                float32x4_t r32 = vextq_f32(r30, r30n, 2);

                sum1 = vmulq_f32(r00, k0);
                sum1 = vmlaq_f32(sum1, r01, k1);
                sum1 = vmlaq_f32(sum1, r02, k2);
                sum1 = vmlaq_f32(sum1, r10, k3);
                sum1 = vmlaq_f32(sum1, r11, k4);
                sum1 = vmlaq_f32(sum1, r12, k5);
                sum1 = vmlaq_f32(sum1, r20, k6);
                sum1 = vmlaq_f32(sum1, r21, k7);
                sum1 = vmlaq_f32(sum1, r22, k8);

                sum2 = vmulq_f32(r10, k0);
                sum2 = vmlaq_f32(sum2, r11, k1);
                sum2 = vmlaq_f32(sum2, r12, k2);
                sum2 = vmlaq_f32(sum2, r20, k3);
                sum2 = vmlaq_f32(sum2, r21, k4);
                sum2 = vmlaq_f32(sum2, r22, k5);
                sum2 = vmlaq_f32(sum2, r30, k6);
                sum2 = vmlaq_f32(sum2, r31, k7);
                sum2 = vmlaq_f32(sum2, r32, k8);

                if (fuseBias)
                {
                    sum1 = vaddq_f32(sum1, vBias);
                    sum2 = vaddq_f32(sum2, vBias);
                }
                if (fuseRelu)
                {
                    sum1 = vmaxq_f32(sum1, vZero);
                    sum2 = vmaxq_f32(sum2, vZero);
                }
                vst1q_f32(og, sum1);
                vst1q_f32(og3, sum2);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                og += 4;
                og3 += 4;
            }
            //the columns remained every 2 rows

            k0123 = vld1q_f32(kp);
            k3456 = vld1q_f32(kp + 3);
            k6789 = vld1q_f32(kp + 6);
            for (; remain > 0; remain--)
            {
                float32x4_t r00 = vld1q_f32(r0);
                float32x4_t r10 = vld1q_f32(r1);
                float32x4_t r20 = vld1q_f32(r2);
                float32x4_t r30 = vld1q_f32(r3);

                float32x4_t sum1 = vmulq_f32(r00, k0123);
                sum1 = vmlaq_f32(sum1, r10, k3456);
                sum1 = vmlaq_f32(sum1, r20, k6789);

                float32x4_t sum2 = vmulq_f32(r10, k0123);
                sum2 = vmlaq_f32(sum2, r20, k3456);
                sum2 = vmlaq_f32(sum2, r30, k6789);

                sum1 = vsetq_lane_f32(0.0f, sum1, 3); //set third value of og to 0
                //*og = vaddvq_f32(sum1);  //accumulate the first three value of og

                float32x2_t ss = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                float32x2_t ss2 = vpadd_f32(ss, ss);
                float vsum = vget_lane_f32(ss2, 0); //accumulate the first three value of og

                sum2 = vsetq_lane_f32(0.0f, sum2, 3); //set third value of og to 0
                //*og3 = vaddvq_f32(sum2);  //accumulate the first three value of og
                ss = vadd_f32(vget_low_f32(sum2), vget_high_f32(sum2));
                ss2 = vpadd_f32(ss, ss);
                float vsum2 = vget_lane_f32(ss2, 0); //accumulate the first three value of og
                if (fuseBias)
                {
                    vsum += bias_arr[g];
                    vsum2 += bias_arr[g];
                }
                if (fuseRelu)
                {
                    vsum = (vsum > 0.f) ? vsum : 0.f;
                    vsum2 = (vsum2 > 0.f) ? vsum2 : 0.f;
                }
                *og = vsum;
                *og3 = vsum2;
                r0++;
                r1++;
                r2++;
                r3++;
                og++;
                og3++;
            }
#endif
        }
        //the remain rows
        for (; i < outh; ++i)
        {
#ifdef __aarch64__                //1 * 16
            int nout = outw >> 4; //outw / 16, compute 16 cols per time
            int remain = outw & 15;
            float *r0 = ing + inw * i;
            float *r1 = ing + inw * (i + 1);
            float *r2 = ing + inw * (i + 2);

            float *og = outg + outw * i;
            float32x4_t sum1, sum2;
            for (; nout > 0; nout--)
            {
                float32x4_t r00 = vld1q_f32(r0);
                float32x4_t r00n = vld1q_f32(r0 + 4);
                float32x4_t r00nn = vld1q_f32(r0 + 8);
                float32x4_t r00nnn = vld1q_f32(r0 + 12);
                float32x4_t r00nnnn = vld1q_f32(r0 + 16);
                float32x4_t r01 = vextq_f32(r00, r00n, 1);
                float32x4_t r02 = vextq_f32(r00, r00n, 2);

                float32x4_t r10 = vld1q_f32(r1);
                float32x4_t r10n = vld1q_f32(r1 + 4);
                float32x4_t r10nn = vld1q_f32(r1 + 8);
                float32x4_t r10nnn = vld1q_f32(r1 + 12);
                float32x4_t r10nnnn = vld1q_f32(r1 + 16);
                float32x4_t r11 = vextq_f32(r10, r10n, 1);
                float32x4_t r12 = vextq_f32(r10, r10n, 2);

                float32x4_t r20 = vld1q_f32(r2);
                float32x4_t r20n = vld1q_f32(r2 + 4);
                float32x4_t r20nn = vld1q_f32(r2 + 8);
                float32x4_t r20nnn = vld1q_f32(r2 + 12);
                float32x4_t r20nnnn = vld1q_f32(r2 + 16);
                float32x4_t r21 = vextq_f32(r20, r20n, 1);
                float32x4_t r22 = vextq_f32(r20, r20n, 2);

                sum1 = vmulq_f32(r00, k0);
                sum1 = vfmaq_f32(sum1, r01, k1);
                sum1 = vfmaq_f32(sum1, r02, k2);
                sum1 = vfmaq_f32(sum1, r10, k3);
                sum1 = vfmaq_f32(sum1, r11, k4);
                sum1 = vfmaq_f32(sum1, r12, k5);
                sum1 = vfmaq_f32(sum1, r20, k6);
                sum1 = vfmaq_f32(sum1, r21, k7);
                sum1 = vfmaq_f32(sum1, r22, k8);

                r01 = vextq_f32(r00n, r00nn, 1);
                r02 = vextq_f32(r00n, r00nn, 2);
                r11 = vextq_f32(r10n, r10nn, 1);
                r12 = vextq_f32(r10n, r10nn, 2);
                r21 = vextq_f32(r20n, r20nn, 1);
                r22 = vextq_f32(r20n, r20nn, 2);

                sum2 = vmulq_f32(r00n, k0);
                sum2 = vfmaq_f32(sum2, r01, k1);
                sum2 = vfmaq_f32(sum2, r02, k2);
                sum2 = vfmaq_f32(sum2, r10n, k3);
                sum2 = vfmaq_f32(sum2, r11, k4);
                sum2 = vfmaq_f32(sum2, r12, k5);
                sum2 = vfmaq_f32(sum2, r20n, k6);
                sum2 = vfmaq_f32(sum2, r21, k7);
                sum2 = vfmaq_f32(sum2, r22, k8);

                r01 = vextq_f32(r00nn, r00nnn, 1);
                r02 = vextq_f32(r00nn, r00nnn, 2);
                r11 = vextq_f32(r10nn, r10nnn, 1);
                r12 = vextq_f32(r10nn, r10nnn, 2);
                r21 = vextq_f32(r20nn, r20nnn, 1);
                r22 = vextq_f32(r20nn, r20nnn, 2);

                sum3 = vmulq_f32(r00nn, k0);
                sum3 = vfmaq_f32(sum3, r01, k1);
                sum3 = vfmaq_f32(sum3, r02, k2);
                sum3 = vfmaq_f32(sum3, r10nn, k3);
                sum3 = vfmaq_f32(sum3, r11, k4);
                sum3 = vfmaq_f32(sum3, r12, k5);
                sum3 = vfmaq_f32(sum3, r20nn, k6);
                sum3 = vfmaq_f32(sum3, r21, k7);
                sum3 = vfmaq_f32(sum3, r22, k8);

                r01 = vextq_f32(r00nnn, r00nnnn, 1);
                r02 = vextq_f32(r00nnn, r00nnnn, 2);
                r11 = vextq_f32(r10nnn, r10nnnn, 1);
                r12 = vextq_f32(r10nnn, r10nnnn, 2);
                r21 = vextq_f32(r20nnn, r20nnnn, 1);
                r22 = vextq_f32(r20nnn, r20nnnn, 2);

                sum4 = vmulq_f32(r00nnn, k0);
                sum4 = vfmaq_f32(sum4, r01, k1);
                sum4 = vfmaq_f32(sum4, r02, k2);
                sum4 = vfmaq_f32(sum4, r10nnn, k3);
                sum4 = vfmaq_f32(sum4, r11, k4);
                sum4 = vfmaq_f32(sum4, r12, k5);
                sum4 = vfmaq_f32(sum4, r20nnn, k6);
                sum4 = vfmaq_f32(sum4, r21, k7);
                sum4 = vfmaq_f32(sum4, r22, k8);

                if (fuseBias)
                {
                    sum1 = vaddq_f32(sum1, vBias);
                    sum2 = vaddq_f32(sum2, vBias);
                    sum3 = vaddq_f32(sum3, vBias);
                    sum4 = vaddq_f32(sum4, vBias);
                }
                if (fuseRelu)
                {
                    sum1 = vmaxq_f32(sum1, vZero);
                    sum2 = vmaxq_f32(sum2, vZero);
                    sum3 = vmaxq_f32(sum3, vZero);
                    sum4 = vmaxq_f32(sum4, vZero);
                }
                vst1q_f32(og, sum1);
                vst1q_f32(og + 4, sum2);
                vst1q_f32(og + 8, sum3);
                vst1q_f32(og + 12, sum4);
                r0 += 16;
                r1 += 16;
                r2 += 16;
                og += 16;
            }

            //the columns remained every 4 rows
            for (; remain > 0; remain--)
            {
                float32x4_t r00 = vld1q_f32(r0);
                float32x4_t r10 = vld1q_f32(r1);
                float32x4_t r20 = vld1q_f32(r2);

                float32x4_t sum1 = vmulq_f32(r00, k0123);
                sum1 = vfmaq_f32(sum1, r10, k3456);
                sum1 = vfmaq_f32(sum1, r20, k6789);

                sum1 = vsetq_lane_f32(0.0f, sum1, 3); //set third value of og to 0
                float vsum = vaddvq_f32(sum1);        //accumulate the first three value of og
                if (fuseBias)
                {
                    vsum += bias_arr[g];
                }
                if (fuseRelu)
                {
                    vsum = (vsum > 0.f) ? vsum : 0.f;
                }
                *og = vsum;
                r0++;
                r1++;
                r2++;
                og++;
            }
#else  //ARMv7, 1 * 8
            int nout = outw >> 3; //outw / 8, compute 8 cols per time
            int remain = outw & 7;
            float *r0 = ing + inw * i;
            float *r1 = ing + inw * (i + 1);
            float *r2 = ing + inw * (i + 2);

            float *og = outg + outw * i;
            float32x4_t sum1, sum2;
            for (; nout > 0; nout--)
            {
                float32x4_t r00 = vld1q_f32(r0);
                float32x4_t r00n = vld1q_f32(r0 + 4);
                float32x4_t r00nn = vld1q_f32(r0 + 8);
                float32x4_t r01 = vextq_f32(r00, r00n, 1);
                float32x4_t r02 = vextq_f32(r00, r00n, 2);

                float32x4_t r10 = vld1q_f32(r1);
                float32x4_t r10n = vld1q_f32(r1 + 4);
                float32x4_t r10nn = vld1q_f32(r1 + 8);
                float32x4_t r11 = vextq_f32(r10, r10n, 1);
                float32x4_t r12 = vextq_f32(r10, r10n, 2);

                float32x4_t r20 = vld1q_f32(r2);
                float32x4_t r20n = vld1q_f32(r2 + 4);
                float32x4_t r20nn = vld1q_f32(r2 + 8);
                float32x4_t r21 = vextq_f32(r20, r20n, 1);
                float32x4_t r22 = vextq_f32(r20, r20n, 2);

                sum1 = vmulq_f32(r00, k0);
                sum1 = vmlaq_f32(sum1, r01, k1);
                sum1 = vmlaq_f32(sum1, r02, k2);
                sum1 = vmlaq_f32(sum1, r10, k3);
                sum1 = vmlaq_f32(sum1, r11, k4);
                sum1 = vmlaq_f32(sum1, r12, k5);
                sum1 = vmlaq_f32(sum1, r20, k6);
                sum1 = vmlaq_f32(sum1, r21, k7);
                sum1 = vmlaq_f32(sum1, r22, k8);

                r01 = vextq_f32(r00n, r00nn, 1);
                r02 = vextq_f32(r00n, r00nn, 2);
                r11 = vextq_f32(r10n, r10nn, 1);
                r12 = vextq_f32(r10n, r10nn, 2);
                r21 = vextq_f32(r20n, r20nn, 1);
                r22 = vextq_f32(r20n, r20nn, 2);

                sum2 = vmulq_f32(r00n, k0);
                sum2 = vmlaq_f32(sum2, r01, k1);
                sum2 = vmlaq_f32(sum2, r02, k2);
                sum2 = vmlaq_f32(sum2, r10n, k3);
                sum2 = vmlaq_f32(sum2, r11, k4);
                sum2 = vmlaq_f32(sum2, r12, k5);
                sum2 = vmlaq_f32(sum2, r20n, k6);
                sum2 = vmlaq_f32(sum2, r21, k7);
                sum2 = vmlaq_f32(sum2, r22, k8);

                if (fuseBias)
                {
                    sum1 = vaddq_f32(sum1, vBias);
                    sum2 = vaddq_f32(sum2, vBias);
                }
                if (fuseRelu)
                {
                    sum1 = vmaxq_f32(sum1, vZero);
                    sum2 = vmaxq_f32(sum2, vZero);
                }

                vst1q_f32(og, sum1);
                vst1q_f32(og + 4, sum2);
                r0 += 8;
                r1 += 8;
                r2 += 8;
                og += 8;
            }

            //the columns remained every 4 rows
            for (; remain > 0; remain--)
            {
                float32x4_t r00 = vld1q_f32(r0);
                float32x4_t r10 = vld1q_f32(r1);
                float32x4_t r20 = vld1q_f32(r2);

                float32x4_t sum1 = vmulq_f32(r00, k0123);
                sum1 = vmlaq_f32(sum1, r10, k3456);
                sum1 = vmlaq_f32(sum1, r20, k6789);

                sum1 = vsetq_lane_f32(0.0f, sum1, 3); //set third value of og to 0

                float32x2_t ss = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                float32x2_t ss2 = vpadd_f32(ss, ss);
                float vsum = vget_lane_f32(ss2, 0); //accumulate the first three value of og
                if (fuseBias)
                {
                    vsum += bias_arr[g];
                }
                if (fuseRelu)
                {
                    vsum = (vsum > 0.f) ? vsum : 0.f;
                }
                *og = vsum;
                r0++;
                r1++;
                r2++;
                og++;
            }
#endif //__aarch64__
        }
    }
}

template <bool fuseBias, bool fuseRelu>
void dwConvs2(float *output, float *input, int inw, int inh, int stridew, int strideh, float *kernel, int kw, int kh, int group, int nThreads, float *bias_arr)
{
    int outw = (inw - kw + 1) / stridew; //for strided case in odd dimensions, should take the floor value as output dim.
    int outh = (inh - kh + 1) / strideh;
    float32x4_t vZero = vdupq_n_f32(0.f);
    float32x4_t vBias;
    #pragma omp parallel for num_threads(nThreads) schedule(static)
    for (int g = 0; g < group; ++g)
    {
        if (fuseBias)
            vBias = vld1q_dup_f32(bias_arr + g);
        float *kp = kernel + 9 * g;
        float32x4_t k0 = vld1q_dup_f32(kp);
        float32x4_t k1 = vld1q_dup_f32(kp + 1);
        float32x4_t k2 = vld1q_dup_f32(kp + 2);
        float32x4_t k3 = vld1q_dup_f32(kp + 3);
        float32x4_t k4 = vld1q_dup_f32(kp + 4);
        float32x4_t k5 = vld1q_dup_f32(kp + 5);
        float32x4_t k6 = vld1q_dup_f32(kp + 6);
        float32x4_t k7 = vld1q_dup_f32(kp + 7);
        float32x4_t k8 = vld1q_dup_f32(kp + 8);

        float32x4_t k0123, k3456, k6789;
        float *outg = output + g * outw * outh;
        float *ing = input + g * inw * inh;

        float32x4_t sum1, sum2, sum3, sum4;
        int i = 0;
        for (; i < outh; i++) // 1 rows per loop
        {
#ifdef __aarch64__
            int nout = outw >> 4; //outw / 16, compute 16 cols per time
            int remain = outw & 15;
            float *_r0 = ing + inw * i * 2;
            float *_r1 = _r0 + inw;
            float *_r2 = _r1 + inw;

            float *og = outg + outw * i;

            for (; nout > 0; nout--)
            {
                float32x4x2_t r0 = vld2q_f32(_r0);
                float32x4x2_t r0n1 = vld2q_f32(_r0 + 8);
                float32x4x2_t r0n2 = vld2q_f32(_r0 + 16);
                float32x4x2_t r0n3 = vld2q_f32(_r0 + 24);
                float32x4x2_t r0n4 = vld2q_f32(_r0 + 32);
                float32x4_t r00 = r0.val[0];                      //0 2 4 6
                float32x4_t r01 = r0.val[1];                      //1 3 5 7
                float32x4_t r02 = vextq_f32(r00, r0n1.val[0], 1); //2 4 6 8

                float32x4x2_t r1 = vld2q_f32(_r1);
                float32x4x2_t r1n1 = vld2q_f32(_r1 + 8);
                float32x4x2_t r1n2 = vld2q_f32(_r1 + 16);
                float32x4x2_t r1n3 = vld2q_f32(_r1 + 24);
                float32x4x2_t r1n4 = vld2q_f32(_r1 + 32);
                float32x4_t r10 = r1.val[0];                      //0 2 4 6
                float32x4_t r11 = r1.val[1];                      //1 3 5 7
                float32x4_t r12 = vextq_f32(r10, r1n1.val[0], 1); //2 4 6 8

                float32x4x2_t r2 = vld2q_f32(_r2);
                float32x4x2_t r2n1 = vld2q_f32(_r2 + 8);
                float32x4x2_t r2n2 = vld2q_f32(_r2 + 16);
                float32x4x2_t r2n3 = vld2q_f32(_r2 + 24);
                float32x4x2_t r2n4 = vld2q_f32(_r2 + 32);
                float32x4_t r20 = r2.val[0];                      //0 2 4 6
                float32x4_t r21 = r2.val[1];                      //1 3 5 7
                float32x4_t r22 = vextq_f32(r20, r2n1.val[0], 1); //2 4 6 8

                sum1 = vmulq_f32(r00, k0);
                sum2 = vmulq_f32(r01, k1);
                sum1 = vfmaq_f32(sum1, r02, k2);
                sum2 = vfmaq_f32(sum2, r10, k3);
                sum1 = vfmaq_f32(sum1, r11, k4);
                sum2 = vfmaq_f32(sum2, r12, k5);
                sum1 = vfmaq_f32(sum1, r20, k6);
                sum2 = vfmaq_f32(sum2, r21, k7);
                sum1 = vfmaq_f32(sum1, r22, k8);

                sum1 = vaddq_f32(sum1, sum2);
                if (fuseBias)
                    sum1 = vaddq_f32(sum1, vBias);
                if (fuseRelu)
                    sum1 = vmaxq_f32(sum1, vZero);
                vst1q_f32(og, sum1);

                r00 = r0n1.val[0];                    //0 2 4 6
                r01 = r0n1.val[1];                    //1 3 5 7
                r02 = vextq_f32(r00, r0n2.val[0], 1); //2 4 6 8

                r10 = r1n1.val[0];                    //0 2 4 6
                r11 = r1n1.val[1];                    //1 3 5 7
                r12 = vextq_f32(r10, r1n2.val[0], 1); //2 4 6 8

                r20 = r2n1.val[0];                    //0 2 4 6
                r21 = r2n1.val[1];                    //1 3 5 7
                r22 = vextq_f32(r20, r2n2.val[0], 1); //2 4 6 8

                sum1 = vmulq_f32(r00, k0);
                sum2 = vmulq_f32(r01, k1);
                sum1 = vfmaq_f32(sum1, r02, k2);
                sum2 = vfmaq_f32(sum2, r10, k3);
                sum1 = vfmaq_f32(sum1, r11, k4);
                sum2 = vfmaq_f32(sum2, r12, k5);
                sum1 = vfmaq_f32(sum1, r20, k6);
                sum2 = vfmaq_f32(sum2, r21, k7);
                sum1 = vfmaq_f32(sum1, r22, k8);

                sum1 = vaddq_f32(sum1, sum2);
                if (fuseBias)
                    sum1 = vaddq_f32(sum1, vBias);
                if (fuseRelu)
                    sum1 = vmaxq_f32(sum1, vZero);
                vst1q_f32(og + 4, sum1);

                r00 = r0n2.val[0];                    //0 2 4 6
                r01 = r0n2.val[1];                    //1 3 5 7
                r02 = vextq_f32(r00, r0n3.val[0], 1); //2 4 6 8

                r10 = r1n2.val[0];                    //0 2 4 6
                r11 = r1n2.val[1];                    //1 3 5 7
                r12 = vextq_f32(r10, r1n3.val[0], 1); //2 4 6 8

                r20 = r2n2.val[0];                    //0 2 4 6
                r21 = r2n2.val[1];                    //1 3 5 7
                r22 = vextq_f32(r20, r2n3.val[0], 1); //2 4 6 8

                sum1 = vmulq_f32(r00, k0);
                sum2 = vmulq_f32(r01, k1);
                sum1 = vfmaq_f32(sum1, r02, k2);
                sum2 = vfmaq_f32(sum2, r10, k3);
                sum1 = vfmaq_f32(sum1, r11, k4);
                sum2 = vfmaq_f32(sum2, r12, k5);
                sum1 = vfmaq_f32(sum1, r20, k6);
                sum2 = vfmaq_f32(sum2, r21, k7);
                sum1 = vfmaq_f32(sum1, r22, k8);

                sum1 = vaddq_f32(sum1, sum2);
                if (fuseBias)
                    sum1 = vaddq_f32(sum1, vBias);
                if (fuseRelu)
                    sum1 = vmaxq_f32(sum1, vZero);
                vst1q_f32(og + 8, sum1);

                r00 = r0n3.val[0];                    //0 2 4 6
                r01 = r0n3.val[1];                    //1 3 5 7
                r02 = vextq_f32(r00, r0n4.val[0], 1); //2 4 6 8

                r10 = r1n3.val[0];                    //0 2 4 6
                r11 = r1n3.val[1];                    //1 3 5 7
                r12 = vextq_f32(r10, r1n4.val[0], 1); //2 4 6 8

                r20 = r2n3.val[0];                    //0 2 4 6
                r21 = r2n3.val[1];                    //1 3 5 7
                r22 = vextq_f32(r20, r2n4.val[0], 1); //2 4 6 8

                sum1 = vmulq_f32(r00, k0);
                sum2 = vmulq_f32(r01, k1);
                sum1 = vfmaq_f32(sum1, r02, k2);
                sum2 = vfmaq_f32(sum2, r10, k3);
                sum1 = vfmaq_f32(sum1, r11, k4);
                sum2 = vfmaq_f32(sum2, r12, k5);
                sum1 = vfmaq_f32(sum1, r20, k6);
                sum2 = vfmaq_f32(sum2, r21, k7);
                sum1 = vfmaq_f32(sum1, r22, k8);

                sum1 = vaddq_f32(sum1, sum2);

                if (fuseBias)
                    sum1 = vaddq_f32(sum1, vBias);
                if (fuseRelu)
                    sum1 = vmaxq_f32(sum1, vZero);
                vst1q_f32(og + 12, sum1);

                _r0 += 32;
                _r1 += 32;
                _r2 += 32;
                og += 16;
            }
            //the columns remained every 4 rows
#if 1 //compute 1 * 8 outputs
            for (; remain - 7 > 0; remain -= 8)
            {
                float32x4x2_t r0 = vld2q_f32(_r0);
                float32x4x2_t r0n1 = vld2q_f32(_r0 + 8);
                float32x4x2_t r0n2 = vld2q_f32(_r0 + 16);
                float32x4_t r00 = r0.val[0];                      //0 2 4 6
                float32x4_t r01 = r0.val[1];                      //1 3 5 7
                float32x4_t r02 = vextq_f32(r00, r0n1.val[0], 1); //2 4 6 8

                float32x4x2_t r1 = vld2q_f32(_r1);
                float32x4x2_t r1n1 = vld2q_f32(_r1 + 8);
                float32x4x2_t r1n2 = vld2q_f32(_r1 + 16);
                float32x4_t r10 = r1.val[0];                      //0 2 4 6
                float32x4_t r11 = r1.val[1];                      //1 3 5 7
                float32x4_t r12 = vextq_f32(r10, r1n1.val[0], 1); //2 4 6 8

                float32x4x2_t r2 = vld2q_f32(_r2);
                float32x4x2_t r2n1 = vld2q_f32(_r2 + 8);
                float32x4x2_t r2n2 = vld2q_f32(_r2 + 16);
                float32x4_t r20 = r2.val[0];                      //0 2 4 6
                float32x4_t r21 = r2.val[1];                      //1 3 5 7
                float32x4_t r22 = vextq_f32(r20, r2n1.val[0], 1); //2 4 6 8

                sum1 = vmulq_f32(r00, k0);
                sum2 = vmulq_f32(r01, k1);
                sum1 = vfmaq_f32(sum1, r02, k2);
                sum2 = vfmaq_f32(sum2, r10, k3);
                sum1 = vfmaq_f32(sum1, r11, k4);
                sum2 = vfmaq_f32(sum2, r12, k5);
                sum1 = vfmaq_f32(sum1, r20, k6);
                sum2 = vfmaq_f32(sum2, r21, k7);
                sum1 = vfmaq_f32(sum1, r22, k8);

                sum1 = vaddq_f32(sum1, sum2);
                if (fuseBias)
                    sum1 = vaddq_f32(sum1, vBias);
                if (fuseRelu)
                    sum1 = vmaxq_f32(sum1, vZero);
                vst1q_f32(og, sum1);

                r00 = r0n1.val[0];                    //0 2 4 6
                r01 = r0n1.val[1];                    //1 3 5 7
                r02 = vextq_f32(r00, r0n2.val[0], 1); //2 4 6 8

                r10 = r1n1.val[0];                    //0 2 4 6
                r11 = r1n1.val[1];                    //1 3 5 7
                r12 = vextq_f32(r10, r1n2.val[0], 1); //2 4 6 8

                r20 = r2n1.val[0];                    //0 2 4 6
                r21 = r2n1.val[1];                    //1 3 5 7
                r22 = vextq_f32(r20, r2n2.val[0], 1); //2 4 6 8

                sum1 = vmulq_f32(r00, k0);
                sum2 = vmulq_f32(r01, k1);
                sum1 = vfmaq_f32(sum1, r02, k2);
                sum2 = vfmaq_f32(sum2, r10, k3);
                sum1 = vfmaq_f32(sum1, r11, k4);
                sum2 = vfmaq_f32(sum2, r12, k5);
                sum1 = vfmaq_f32(sum1, r20, k6);
                sum2 = vfmaq_f32(sum2, r21, k7);
                sum1 = vfmaq_f32(sum1, r22, k8);

                sum1 = vaddq_f32(sum1, sum2);
                if (fuseBias)
                    sum1 = vaddq_f32(sum1, vBias);
                if (fuseRelu)
                    sum1 = vmaxq_f32(sum1, vZero);
                vst1q_f32(og + 4, sum1);

                _r0 += 16;
                _r1 += 16;
                _r2 += 16;
                og += 8;
            }

            //compute 1 * 4 outputs
            for (; remain - 3 > 0; remain -= 4)
            {
                float32x4x2_t r0 = vld2q_f32(_r0);
                float32x4x2_t r0n1 = vld2q_f32(_r0 + 8);
                float32x4_t r00 = r0.val[0];                      //0 2 4 6
                float32x4_t r01 = r0.val[1];                      //1 3 5 7
                float32x4_t r02 = vextq_f32(r00, r0n1.val[0], 1); //2 4 6 8

                float32x4x2_t r1 = vld2q_f32(_r1);
                float32x4x2_t r1n1 = vld2q_f32(_r1 + 8);
                float32x4_t r10 = r1.val[0];                      //0 2 4 6
                float32x4_t r11 = r1.val[1];                      //1 3 5 7
                float32x4_t r12 = vextq_f32(r10, r1n1.val[0], 1); //2 4 6 8

                float32x4x2_t r2 = vld2q_f32(_r2);
                float32x4x2_t r2n1 = vld2q_f32(_r2 + 8);
                float32x4_t r20 = r2.val[0];                      //0 2 4 6
                float32x4_t r21 = r2.val[1];                      //1 3 5 7
                float32x4_t r22 = vextq_f32(r20, r2n1.val[0], 1); //2 4 6 8

                sum1 = vmulq_f32(r00, k0);
                sum2 = vmulq_f32(r01, k1);
                sum1 = vfmaq_f32(sum1, r02, k2);
                sum2 = vfmaq_f32(sum2, r10, k3);
                sum1 = vfmaq_f32(sum1, r11, k4);
                sum2 = vfmaq_f32(sum2, r12, k5);
                sum1 = vfmaq_f32(sum1, r20, k6);
                sum2 = vfmaq_f32(sum2, r21, k7);
                sum1 = vfmaq_f32(sum1, r22, k8);

                sum1 = vaddq_f32(sum1, sum2);
                if (fuseBias)
                    sum1 = vaddq_f32(sum1, vBias);
                if (fuseRelu)
                    sum1 = vmaxq_f32(sum1, vZero);
                vst1q_f32(og, sum1);

                _r0 += 8;
                _r1 += 8;
                _r2 += 8;
                og += 4;
            }
#endif
            k0123 = vld1q_f32(kp);
            k3456 = vld1q_f32(kp + 3);
            k6789 = vld1q_f32(kp + 6);

            //compute the remain outputs which less than 4
            for (; remain > 0; remain--)
            {
                float32x4_t r00 = vld1q_f32(_r0);
                float32x4_t r10 = vld1q_f32(_r1);
                float32x4_t r20 = vld1q_f32(_r2);

                float32x4_t sum1 = vmulq_f32(r00, k0123);
                sum1 = vfmaq_f32(sum1, r10, k3456);
                sum1 = vfmaq_f32(sum1, r20, k6789);

                sum1 = vsetq_lane_f32(0.0f, sum1, 3); //set third value of og to 0

                float vsum = vaddvq_f32(sum1);        //accumulate the first three value of og
                if (fuseBias)
                    vsum += bias_arr[g];
                if (fuseRelu)
                    vsum = (vsum > 0.f) ? vsum : 0.f;

                *og = vsum;
                _r0 += 2;
                _r1 += 2;
                _r2 += 2;
                og++;
            }
#else  //ARMv7
            int nout = outw >> 3; //outw / 8, compute 8 cols per time
            int remain = outw & 7;
            float *_r0 = ing + inw * i * 2;
            float *_r1 = _r0 + inw;
            float *_r2 = _r1 + inw;

            float *og = outg + outw * i;

            for (; nout > 0; nout--)
            {
                float32x4x2_t r0 = vld2q_f32(_r0);
                float32x4x2_t r0n1 = vld2q_f32(_r0 + 8);
                float32x4x2_t r0n2 = vld2q_f32(_r0 + 16);
                float32x4_t r00 = r0.val[0];                      //0 2 4 6
                float32x4_t r01 = r0.val[1];                      //1 3 5 7
                float32x4_t r02 = vextq_f32(r00, r0n1.val[0], 1); //2 4 6 8

                float32x4x2_t r1 = vld2q_f32(_r1);
                float32x4x2_t r1n1 = vld2q_f32(_r1 + 8);
                float32x4x2_t r1n2 = vld2q_f32(_r1 + 16);
                float32x4_t r10 = r1.val[0];                      //0 2 4 6
                float32x4_t r11 = r1.val[1];                      //1 3 5 7
                float32x4_t r12 = vextq_f32(r10, r1n1.val[0], 1); //2 4 6 8

                float32x4x2_t r2 = vld2q_f32(_r2);
                float32x4x2_t r2n1 = vld2q_f32(_r2 + 8);
                float32x4x2_t r2n2 = vld2q_f32(_r2 + 16);
                float32x4_t r20 = r2.val[0];                      //0 2 4 6
                float32x4_t r21 = r2.val[1];                      //1 3 5 7
                float32x4_t r22 = vextq_f32(r20, r2n1.val[0], 1); //2 4 6 8

                sum1 = vmulq_f32(r00, k0);
                sum2 = vmulq_f32(r01, k1);
                sum1 = vmlaq_f32(sum1, r02, k2);
                sum2 = vmlaq_f32(sum2, r10, k3);
                sum1 = vmlaq_f32(sum1, r11, k4);
                sum2 = vmlaq_f32(sum2, r12, k5);
                sum1 = vmlaq_f32(sum1, r20, k6);
                sum2 = vmlaq_f32(sum2, r21, k7);
                sum1 = vmlaq_f32(sum1, r22, k8);

                sum1 = vaddq_f32(sum1, sum2);
                if (fuseBias)
                    sum1 = vaddq_f32(sum1, vBias);
                if (fuseRelu)
                    sum1 = vmaxq_f32(sum1, vZero);
                vst1q_f32(og, sum1);

                r00 = r0n1.val[0];                    //0 2 4 6
                r01 = r0n1.val[1];                    //1 3 5 7
                r02 = vextq_f32(r00, r0n2.val[0], 1); //2 4 6 8

                r10 = r1n1.val[0];                    //0 2 4 6
                r11 = r1n1.val[1];                    //1 3 5 7
                r12 = vextq_f32(r10, r1n2.val[0], 1); //2 4 6 8

                r20 = r2n1.val[0];                    //0 2 4 6
                r21 = r2n1.val[1];                    //1 3 5 7
                r22 = vextq_f32(r20, r2n2.val[0], 1); //2 4 6 8

                sum1 = vmulq_f32(r00, k0);
                sum2 = vmulq_f32(r01, k1);
                sum1 = vmlaq_f32(sum1, r02, k2);
                sum2 = vmlaq_f32(sum2, r10, k3);
                sum1 = vmlaq_f32(sum1, r11, k4);
                sum2 = vmlaq_f32(sum2, r12, k5);
                sum1 = vmlaq_f32(sum1, r20, k6);
                sum2 = vmlaq_f32(sum2, r21, k7);
                sum1 = vmlaq_f32(sum1, r22, k8);

                sum1 = vaddq_f32(sum1, sum2);
                if (fuseBias)
                    sum1 = vaddq_f32(sum1, vBias);
                if (fuseRelu)
                    sum1 = vmaxq_f32(sum1, vZero);
                vst1q_f32(og + 4, sum1);

                _r0 += 16;
                _r1 += 16;
                _r2 += 16;
                og += 8;
            }

            //compute 1 * 4 outputs
            for (; remain - 3 > 0; remain -= 4)
            {
                float32x4x2_t r0 = vld2q_f32(_r0);
                float32x4x2_t r0n1 = vld2q_f32(_r0 + 8);
                float32x4_t r00 = r0.val[0];                      //0 2 4 6
                float32x4_t r01 = r0.val[1];                      //1 3 5 7
                float32x4_t r02 = vextq_f32(r00, r0n1.val[0], 1); //2 4 6 8

                float32x4x2_t r1 = vld2q_f32(_r1);
                float32x4x2_t r1n1 = vld2q_f32(_r1 + 8);
                float32x4_t r10 = r1.val[0];                      //0 2 4 6
                float32x4_t r11 = r1.val[1];                      //1 3 5 7
                float32x4_t r12 = vextq_f32(r10, r1n1.val[0], 1); //2 4 6 8

                float32x4x2_t r2 = vld2q_f32(_r2);
                float32x4x2_t r2n1 = vld2q_f32(_r2 + 8);
                float32x4_t r20 = r2.val[0];                      //0 2 4 6
                float32x4_t r21 = r2.val[1];                      //1 3 5 7
                float32x4_t r22 = vextq_f32(r20, r2n1.val[0], 1); //2 4 6 8

                sum1 = vmulq_f32(r00, k0);
                sum2 = vmulq_f32(r01, k1);
                sum1 = vmlaq_f32(sum1, r02, k2);
                sum2 = vmlaq_f32(sum2, r10, k3);
                sum1 = vmlaq_f32(sum1, r11, k4);
                sum2 = vmlaq_f32(sum2, r12, k5);
                sum1 = vmlaq_f32(sum1, r20, k6);
                sum2 = vmlaq_f32(sum2, r21, k7);
                sum1 = vmlaq_f32(sum1, r22, k8);

                sum1 = vaddq_f32(sum1, sum2);
                if (fuseBias)
                    sum1 = vaddq_f32(sum1, vBias);
                if (fuseRelu)
                    sum1 = vmaxq_f32(sum1, vZero);
                vst1q_f32(og, sum1);

                _r0 += 8;
                _r1 += 8;
                _r2 += 8;
                og += 4;
            }

            k0123 = vld1q_f32(kp);
            k3456 = vld1q_f32(kp + 3);
            k6789 = vld1q_f32(kp + 6);

            //Compute the remain outputs which less than 4
            for (; remain > 0; remain--)
            {
                float32x4_t r00 = vld1q_f32(_r0);
                float32x4_t r10 = vld1q_f32(_r1);
                float32x4_t r20 = vld1q_f32(_r2);

                float32x4_t sum1 = vmulq_f32(r00, k0123);
                sum1 = vmlaq_f32(sum1, r10, k3456);
                sum1 = vmlaq_f32(sum1, r20, k6789);
                sum1 = vsetq_lane_f32(0.0f, sum1, 3); //set third value of og to 0
                //*og = vaddvq_f32(sum1);  //accumulate the first three value of og
                float32x2_t ss = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                float32x2_t ss2 = vpadd_f32(ss, ss);
                float vsum = vget_lane_f32(ss2, 0); //accumulate the first three value of og
                if (fuseBias)
                    vsum += bias_arr[g];
                if (fuseRelu)
                    vsum = (vsum > 0.f) ? vsum : 0.f;
                *og = vsum;
                _r0 += 2;
                _r1 += 2;
                _r2 += 2;
                og++;
            }
#endif //__aarch64__
        }
    }
}

template <bool fuseBias, bool fuseRelu>
void dwConv_template(float *output, float *input, int input_channels, int inw, int inh, int stridew, int strideh, float *kernel, int kw, int kh, int group, int nThreads, float *bias_arr)
{
    if ((kw == inw) && (kh == inh))
    {
        globalDwConv<fuseBias, fuseRelu>(output, input, input_channels, inw, inh, kernel, group, nThreads, bias_arr);
    }
    else if (kw == 3 && kh == 3 && stridew == 1 && strideh == 1)
        dwConvs1<fuseBias, fuseRelu>(output, input, inw, inh, stridew, strideh, kernel, kw, kh, group, nThreads, bias_arr);
    else if (kw == 3 && kh == 3 && stridew == 2 && strideh == 2)
        dwConvs2<fuseBias, fuseRelu>(output, input, inw, inh, stridew, strideh, kernel, kw, kh, group, nThreads, bias_arr);
    else
    {
        int outw = (inw - kw) / stridew + 1; //for strided case in odd dimensions, should take the floor value as output dim.
        int outh = (inh - kh) / strideh + 1;

// #pragma omp parallel for num_threads(nThreads) schedule(static)
        printf("dw param %d kernel %d %d stride %d %d input %d %d %d output %d %d\n", group, kh, kw, strideh, stridew, input_channels, inh, inw, outh, outw);
        for (int g = 0; g < group; ++g)
        {
            float *kp = kernel + kw * kh * g;
            float *outg = output + g * outw * outh;
            float *ing = input + g * inw * inh;
            for (int i = 0; i < outh; ++i)
            {
                for (int j = 0; j < outw; ++j)
                {
                    float *inp = ing + inw * (i * stridew) + (j * strideh);
                    float convSum = 0.f;
                    for (int m = 0; m < kh; m++)
                    {
                        for (int n = 0; n < kw; n++)
                        {
                            convSum += inp[m * inw + n] * kp[m * kw + n];
                        }
                    }
                    if (fuseBias)
                    {
                        convSum += bias_arr[g];
                    }
                    if (fuseRelu)
                    {
                        convSum = (convSum > 0.f) ? convSum : 0.f;
                    }
                    outg[j] = convSum;
                }
                outg += outw;
            }
        }
    }
}

template void dwConv_template<false, false>(float *, float *, int, int, int, int, int, float *, int, int, int, int, float *);
template void dwConv_template<false,  true>(float *, float *, int, int, int, int, int, float *, int, int, int, int, float *);
template void dwConv_template<true,  false>(float *, float *, int, int, int, int, int, float *, int, int, int, int, float *);
template void dwConv_template<true,   true>(float *, float *, int, int, int, int, int, float *, int, int, int, int, float *);
