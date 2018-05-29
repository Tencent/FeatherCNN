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


#include "depthwise.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <arm_neon.h>

#ifdef __APPLE__
#else
#include <omp.h>
#endif

//optimized by xningwang on 22/12/2017
void dwConvs1(float* output, float* input, int inw, int inh, int stridew, int strideh, float* kernel, int kw, int kh, int group, int nThreads)
{
    int outw = (inw - kw + 1) / stridew;//for strided case in odd dimensions, should take the floor value as output dim.
    int outh = (inh - kh + 1) / strideh;

    #pragma omp parallel for num_threads(nThreads) schedule(static)
    for (int g = 0; g < group; ++g)
    {

        float* kp = kernel + 9 * g;

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
        float* outg = output + g * outw * outh;
        float* ing = input + g * inw * inh;

        float32x4_t sum1, sum2, sum3, sum4;
        int i = 0;
        for (; i + 1 <  outh; i += 2)
        {
#ifdef __aarch64__
            int nout = outw >> 3;
            int remain = outw & 7;
            float* r0 = ing + inw * i;
            float* r1 = ing + inw * (i + 1);
            float* r2 = ing + inw * (i + 2);
            float* r3 = ing + inw * (i + 3);

            float* og = outg + outw * i;
            float* og3 = og + outw;

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


                sum1 = vsetq_lane_f32(0.0f, sum1, 3);  //set third value of og to 0
                *og = vaddvq_f32(sum1);  //accumulate the first three value of og
                sum2 = vsetq_lane_f32(0.0f, sum2, 3);  //set third value of og to 0
                *og3 = vaddvq_f32(sum2);  //accumulate the first three value of og
                r0++;
                r1++;
                r2++;
                r3++;
                og++;
                og3++;
            }

#else  //ARMv7, 2 * 4 
            int nout = outw >> 2;  //outw / 4, compute 4 cols per time
            int remain = outw & 3;
            float* r0 = ing + inw * i;
            float* r1 = ing + inw * (i + 1);
            float* r2 = ing + inw * (i + 2);
            float* r3 = ing + inw * (i + 3);

            float* og = outg + outw * i;
            float* og3 = og + outw;

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


                sum1 = vsetq_lane_f32(0.0f, sum1, 3);  //set third value of og to 0
                //*og = vaddvq_f32(sum1);  //accumulate the first three value of og
                float32x2_t ss = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                float32x2_t ss2 = vpadd_f32(ss, ss);
                *og = vget_lane_f32(ss2, 0);  //accumulate the first three value of og

                sum2 = vsetq_lane_f32(0.0f, sum2, 3);  //set third value of og to 0
                //*og3 = vaddvq_f32(sum2);  //accumulate the first three value of og
                ss = vadd_f32(vget_low_f32(sum2), vget_high_f32(sum2));
                ss2 = vpadd_f32(ss, ss);
                *og3 = vget_lane_f32(ss2, 0);  //accumulate the first three value of og
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
#ifdef __aarch64__     //1 * 16
            int nout = outw >> 4;  //outw / 16, compute 16 cols per time
            int remain = outw & 15;
            float* r0 = ing + inw * i;
            float* r1 = ing + inw * (i + 1);
            float* r2 = ing + inw * (i + 2);

            float* og = outg + outw * i;
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

                sum1 = vsetq_lane_f32(0.0f, sum1, 3);  //set third value of og to 0
                *og = vaddvq_f32(sum1);  //accumulate the first three value of og

                r0++;
                r1++;
                r2++;
                og++;
            }
#else    //ARMv7, 1 * 8
            int nout = outw >> 3;  //outw / 8, compute 8 cols per time
            int remain = outw & 7;
            float* r0 = ing + inw * i;
            float* r1 = ing + inw * (i + 1);
            float* r2 = ing + inw * (i + 2);

            float* og = outg + outw * i;
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

                sum1 = vsetq_lane_f32(0.0f, sum1, 3);  //set third value of og to 0
                //*og = vaddvq_f32(sum1);  //accumulate the first three value of og
                float32x2_t ss = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                float32x2_t ss2 = vpadd_f32(ss, ss);
                *og = vget_lane_f32(ss2, 0);  //accumulate the first three value of og
                r0++;
                r1++;
                r2++;
                og++;
            }
#endif  //__aarch64__
        }
    }
}

void dwConvs2(float* output, float* input, int inw, int inh, int stridew, int strideh, float* kernel, int kw, int kh, int group, int nThreads)
{
    int outw = (inw - kw + 1) / stridew;//for strided case in odd dimensions, should take the floor value as output dim.
    int outh = (inh - kh + 1) / strideh;

    #pragma omp parallel for num_threads(nThreads) schedule(static)
    for (int g = 0; g < group; ++g)
    {

        float* kp = kernel + 9 * g;
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
        float* outg = output + g * outw * outh;
        float* ing = input + g * inw * inh;

        float32x4_t sum1, sum2, sum3, sum4;
        int i = 0;
        for (; i <  outh; i++)  // 1 rows per loop
        {
#ifdef __aarch64__
            int nout = outw >> 4;  //outw / 16, compute 16 cols per time
            int remain = outw & 15;
            float* _r0 = ing + inw * i * 2;
            float* _r1 = _r0 + inw;
            float* _r2 = _r1 + inw;

            float* og = outg + outw * i;

            for (; nout > 0; nout--)
            {
                float32x4x2_t r0 = vld2q_f32(_r0);
                float32x4x2_t r0n1 = vld2q_f32(_r0 + 8);
                float32x4x2_t r0n2 = vld2q_f32(_r0 + 16);
                float32x4x2_t r0n3 = vld2q_f32(_r0 + 24);
                float32x4x2_t r0n4 = vld2q_f32(_r0 + 32);
                float32x4_t r00 = r0.val[0];  //0 2 4 6
                float32x4_t r01 = r0.val[1];  //1 3 5 7
                float32x4_t r02 = vextq_f32(r00, r0n1.val[0], 1);  //2 4 6 8

                float32x4x2_t r1 = vld2q_f32(_r1);
                float32x4x2_t r1n1 = vld2q_f32(_r1 + 8);
                float32x4x2_t r1n2 = vld2q_f32(_r1 + 16);
                float32x4x2_t r1n3 = vld2q_f32(_r1 + 24);
                float32x4x2_t r1n4 = vld2q_f32(_r1 + 32);
                float32x4_t r10 = r1.val[0];  //0 2 4 6
                float32x4_t r11 = r1.val[1];  //1 3 5 7
                float32x4_t r12 = vextq_f32(r10, r1n1.val[0], 1);  //2 4 6 8

                float32x4x2_t r2 = vld2q_f32(_r2);
                float32x4x2_t r2n1 = vld2q_f32(_r2 + 8);
                float32x4x2_t r2n2 = vld2q_f32(_r2 + 16);
                float32x4x2_t r2n3 = vld2q_f32(_r2 + 24);
                float32x4x2_t r2n4 = vld2q_f32(_r2 + 32);
                float32x4_t r20 = r2.val[0];  //0 2 4 6
                float32x4_t r21 = r2.val[1];  //1 3 5 7
                float32x4_t r22 = vextq_f32(r20, r2n1.val[0], 1);  //2 4 6 8

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
                vst1q_f32(og, sum1);

                r00 = r0n1.val[0];  //0 2 4 6
                r01 = r0n1.val[1];  //1 3 5 7
                r02 = vextq_f32(r00, r0n2.val[0], 1);  //2 4 6 8

                r10 = r1n1.val[0];  //0 2 4 6
                r11 = r1n1.val[1];  //1 3 5 7
                r12 = vextq_f32(r10, r1n2.val[0], 1);  //2 4 6 8

                r20 = r2n1.val[0];  //0 2 4 6
                r21 = r2n1.val[1];  //1 3 5 7
                r22 = vextq_f32(r20, r2n2.val[0], 1);  //2 4 6 8

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
                vst1q_f32(og + 4, sum1);

                r00 = r0n2.val[0];  //0 2 4 6
                r01 = r0n2.val[1];  //1 3 5 7
                r02 = vextq_f32(r00, r0n3.val[0], 1);  //2 4 6 8

                r10 = r1n2.val[0];  //0 2 4 6
                r11 = r1n2.val[1];  //1 3 5 7
                r12 = vextq_f32(r10, r1n3.val[0], 1);  //2 4 6 8

                r20 = r2n2.val[0];  //0 2 4 6
                r21 = r2n2.val[1];  //1 3 5 7
                r22 = vextq_f32(r20, r2n3.val[0], 1);  //2 4 6 8

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
                vst1q_f32(og + 8, sum1);

                r00 = r0n3.val[0];  //0 2 4 6
                r01 = r0n3.val[1];  //1 3 5 7
                r02 = vextq_f32(r00, r0n4.val[0], 1);  //2 4 6 8

                r10 = r1n3.val[0];  //0 2 4 6
                r11 = r1n3.val[1];  //1 3 5 7
                r12 = vextq_f32(r10, r1n4.val[0], 1);  //2 4 6 8

                r20 = r2n3.val[0];  //0 2 4 6
                r21 = r2n3.val[1];  //1 3 5 7
                r22 = vextq_f32(r20, r2n4.val[0], 1);  //2 4 6 8

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
                vst1q_f32(og + 12, sum1);

                _r0 += 32;
                _r1 += 32;
                _r2 += 32;
                og += 16;
            }
            //the columns remained every 4 rows
#if 1                        //compute 1 * 8 outputs
            for (; remain - 7 > 0; remain -= 8)
            {
                float32x4x2_t r0 = vld2q_f32(_r0);
                float32x4x2_t r0n1 = vld2q_f32(_r0 + 8);
                float32x4x2_t r0n2 = vld2q_f32(_r0 + 16);
                float32x4_t r00 = r0.val[0];  //0 2 4 6
                float32x4_t r01 = r0.val[1];  //1 3 5 7
                float32x4_t r02 = vextq_f32(r00, r0n1.val[0], 1);  //2 4 6 8

                float32x4x2_t r1 = vld2q_f32(_r1);
                float32x4x2_t r1n1 = vld2q_f32(_r1 + 8);
                float32x4x2_t r1n2 = vld2q_f32(_r1 + 16);
                float32x4_t r10 = r1.val[0];  //0 2 4 6
                float32x4_t r11 = r1.val[1];  //1 3 5 7
                float32x4_t r12 = vextq_f32(r10, r1n1.val[0], 1);  //2 4 6 8

                float32x4x2_t r2 = vld2q_f32(_r2);
                float32x4x2_t r2n1 = vld2q_f32(_r2 + 8);
                float32x4x2_t r2n2 = vld2q_f32(_r2 + 16);
                float32x4_t r20 = r2.val[0];  //0 2 4 6
                float32x4_t r21 = r2.val[1];  //1 3 5 7
                float32x4_t r22 = vextq_f32(r20, r2n1.val[0], 1);  //2 4 6 8

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
                vst1q_f32(og, sum1);

                r00 = r0n1.val[0];  //0 2 4 6
                r01 = r0n1.val[1];  //1 3 5 7
                r02 = vextq_f32(r00, r0n2.val[0], 1);  //2 4 6 8

                r10 = r1n1.val[0];  //0 2 4 6
                r11 = r1n1.val[1];  //1 3 5 7
                r12 = vextq_f32(r10, r1n2.val[0], 1);  //2 4 6 8

                r20 = r2n1.val[0];  //0 2 4 6
                r21 = r2n1.val[1];  //1 3 5 7
                r22 = vextq_f32(r20, r2n2.val[0], 1);  //2 4 6 8

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
                float32x4_t r00 = r0.val[0];  //0 2 4 6
                float32x4_t r01 = r0.val[1];  //1 3 5 7
                float32x4_t r02 = vextq_f32(r00, r0n1.val[0], 1);  //2 4 6 8

                float32x4x2_t r1 = vld2q_f32(_r1);
                float32x4x2_t r1n1 = vld2q_f32(_r1 + 8);
                float32x4_t r10 = r1.val[0];  //0 2 4 6
                float32x4_t r11 = r1.val[1];  //1 3 5 7
                float32x4_t r12 = vextq_f32(r10, r1n1.val[0], 1);  //2 4 6 8

                float32x4x2_t r2 = vld2q_f32(_r2);
                float32x4x2_t r2n1 = vld2q_f32(_r2 + 8);
                float32x4_t r20 = r2.val[0];  //0 2 4 6
                float32x4_t r21 = r2.val[1];  //1 3 5 7
                float32x4_t r22 = vextq_f32(r20, r2n1.val[0], 1);  //2 4 6 8

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


                sum1 = vsetq_lane_f32(0.0f, sum1, 3);  //set third value of og to 0
                *og = vaddvq_f32(sum1);  //accumulate the first three value of og
                _r0 += 2;
                _r1 += 2;
                _r2 += 2;
                og++;
            }
#else     //ARMv7
            int nout = outw >> 3;  //outw / 8, compute 8 cols per time
            int remain = outw & 7;
            float* _r0 = ing + inw * i * 2;
            float* _r1 = _r0 + inw;
            float* _r2 = _r1 + inw;

            float* og = outg + outw * i;

            for (; nout > 0; nout--)
            {
                float32x4x2_t r0 = vld2q_f32(_r0);
                float32x4x2_t r0n1 = vld2q_f32(_r0 + 8);
                float32x4x2_t r0n2 = vld2q_f32(_r0 + 16);
                float32x4_t r00 = r0.val[0];  //0 2 4 6
                float32x4_t r01 = r0.val[1];  //1 3 5 7
                float32x4_t r02 = vextq_f32(r00, r0n1.val[0], 1);  //2 4 6 8

                float32x4x2_t r1 = vld2q_f32(_r1);
                float32x4x2_t r1n1 = vld2q_f32(_r1 + 8);
                float32x4x2_t r1n2 = vld2q_f32(_r1 + 16);
                float32x4_t r10 = r1.val[0];  //0 2 4 6
                float32x4_t r11 = r1.val[1];  //1 3 5 7
                float32x4_t r12 = vextq_f32(r10, r1n1.val[0], 1);  //2 4 6 8

                float32x4x2_t r2 = vld2q_f32(_r2);
                float32x4x2_t r2n1 = vld2q_f32(_r2 + 8);
                float32x4x2_t r2n2 = vld2q_f32(_r2 + 16);
                float32x4_t r20 = r2.val[0];  //0 2 4 6
                float32x4_t r21 = r2.val[1];  //1 3 5 7
                float32x4_t r22 = vextq_f32(r20, r2n1.val[0], 1);  //2 4 6 8

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
                vst1q_f32(og, sum1);

                r00 = r0n1.val[0];  //0 2 4 6
                r01 = r0n1.val[1];  //1 3 5 7
                r02 = vextq_f32(r00, r0n2.val[0], 1);  //2 4 6 8

                r10 = r1n1.val[0];  //0 2 4 6
                r11 = r1n1.val[1];  //1 3 5 7
                r12 = vextq_f32(r10, r1n2.val[0], 1);  //2 4 6 8

                r20 = r2n1.val[0];  //0 2 4 6
                r21 = r2n1.val[1];  //1 3 5 7
                r22 = vextq_f32(r20, r2n2.val[0], 1);  //2 4 6 8

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
                float32x4_t r00 = r0.val[0];  //0 2 4 6
                float32x4_t r01 = r0.val[1];  //1 3 5 7
                float32x4_t r02 = vextq_f32(r00, r0n1.val[0], 1);  //2 4 6 8

                float32x4x2_t r1 = vld2q_f32(_r1);
                float32x4x2_t r1n1 = vld2q_f32(_r1 + 8);
                float32x4_t r10 = r1.val[0];  //0 2 4 6
                float32x4_t r11 = r1.val[1];  //1 3 5 7
                float32x4_t r12 = vextq_f32(r10, r1n1.val[0], 1);  //2 4 6 8

                float32x4x2_t r2 = vld2q_f32(_r2);
                float32x4x2_t r2n1 = vld2q_f32(_r2 + 8);
                float32x4_t r20 = r2.val[0];  //0 2 4 6
                float32x4_t r21 = r2.val[1];  //1 3 5 7
                float32x4_t r22 = vextq_f32(r20, r2n1.val[0], 1);  //2 4 6 8

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
                vst1q_f32(og, sum1);

                _r0 += 8;
                _r1 += 8;
                _r2 += 8;
                og += 4;
            }

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
                sum1 = vmlaq_f32(sum1, r10, k3456);
                sum1 = vmlaq_f32(sum1, r20, k6789);


                sum1 = vsetq_lane_f32(0.0f, sum1, 3);  //set third value of og to 0
                //*og = vaddvq_f32(sum1);  //accumulate the first three value of og
                float32x2_t ss = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                float32x2_t ss2 = vpadd_f32(ss, ss);
                *og = vget_lane_f32(ss2, 0);  //accumulate the first three value of og
                _r0 += 2;
                _r1 += 2;
                _r2 += 2;
                og++;
            }
#endif    //__aarch64__
        }
    }
}


void dwConv(float* output, float* input, int inw, int inh, int stridew, int strideh, float* kernel, int kw, int kh, int group, int nThreads)
{
    if (stridew == 1 && strideh == 1)
        dwConvs1(output, input, inw, inh, stridew, strideh, kernel, kw, kh, group, nThreads);
    else if (stridew == 2 && strideh == 2)
        dwConvs2(output, input, inw, inh, stridew, strideh, kernel, kw, kh, group, nThreads);
    else
    {
        int outw = (inw - kw) / stridew + 1;//for strided case in odd dimensions, should take the floor value as output dim.
        int outh = (inh - kh) / strideh + 1;

        #pragma omp parallel for num_threads(nThreads) schedule(static)
        for (int g = 0; g < group; ++g)
        {
            float* kp = kernel + kw * kh * g;
            float* outg = output + g * outw * outh;
            float* ing = input + g * inw * inh;
            for (int i = 0; i < outh; ++i)
            {
                for (int j = 0; j < outw; ++j)
                {
                    float* inp = ing + inw * (i * stridew) + (j * strideh);
                    float convSum = 0.f;
                    for (int m = 0; m < kh; m++)
                    {
                        for (int n = 0; n < kw; n++)
                        {
                            convSum += inp[m * inw + n] * kp[m * kw + n];
                        }
                    }
                    outg[j] = convSum;
                }
                outg += outw;
            }
        }
    }
}
