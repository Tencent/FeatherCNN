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
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <arm_neon.h>
#include <omp.h>
#include "common.h"
#include "sgemm.h"

const int mc = 1024;
const int kc = 256;
const int nc = 256;

void (*sgemm_tiny_scale)(int L, float *a, int lda, float *b, int ldb, float *c, int ldc) = NULL;
void (*internalPackA)(int L, float* packA, float* a, int lda) = NULL;
void block_sgemm_internal_pack(int M, int N, int L, float *a, int lda, float *b, int ldb, float *c, int ldc);
void block_sgemm_pack(int M, int N, int L, float *a, int lda, float *b, int ldb, float *c, int ldc);

static void internalPackA8(int L, float* packA, float* a, int lda)
{
    float *packAptr = packA;
    float *a_p0_ptr, *a_p1_ptr, *a_p2_ptr, *a_p3_ptr;
    float *a_p4_ptr, *a_p5_ptr, *a_p6_ptr, *a_p7_ptr;
    a_p0_ptr = a;
    a_p1_ptr = a + lda;
    a_p2_ptr = a + lda * 2;
    a_p3_ptr = a + lda * 3;
    a_p4_ptr = a + lda * 4;
    a_p5_ptr = a + lda * 5;
    a_p6_ptr = a + lda * 6;
    a_p7_ptr = a + lda * 7;
    for (int i = 0; i < L; ++i)
    {
        *packAptr++ = *a_p0_ptr++;
        *packAptr++ = *a_p1_ptr++;
        *packAptr++ = *a_p2_ptr++;
        *packAptr++ = *a_p3_ptr++;

        *packAptr++ = *a_p4_ptr++;
        *packAptr++ = *a_p5_ptr++;
        *packAptr++ = *a_p6_ptr++;
        *packAptr++ = *a_p7_ptr++;
    }
}

static void internalPackA4(int L, float* packA, float* a, int lda)
{
    float *packAptr = packA;
    float *a_p0_ptr, *a_p1_ptr, *a_p2_ptr, *a_p3_ptr;
    a_p0_ptr = a;
    a_p1_ptr = a + lda;
    a_p2_ptr = a + lda * 2;
    a_p3_ptr = a + lda * 3;
    for (int i = 0; i < L; ++i)
    {
        *packAptr++ = *a_p0_ptr++;
        *packAptr++ = *a_p1_ptr++;
        *packAptr++ = *a_p2_ptr++;
        *packAptr++ = *a_p3_ptr++;
    }
}

static void internalPackA3(int L, float* packA, float* a, int lda)
{
    float *packAptr = packA;
    float *a_p0_ptr, *a_p1_ptr, *a_p2_ptr;
    a_p0_ptr = a;
    a_p1_ptr = a + lda;
    a_p2_ptr = a + lda * 2;
    for (int i = 0; i < L; ++i)
    {
        *packAptr++ = *a_p0_ptr++;
        *packAptr++ = *a_p1_ptr++;
        *packAptr++ = *a_p2_ptr++;
        *packAptr++ = 0.0f;
    }
}

static void internalPackA2(int L, float* packA, float* a, int lda)
{
    float *packAptr = packA;
    float *a_p0_ptr, *a_p1_ptr;
    a_p0_ptr = a;
    a_p1_ptr = a + lda;
    for (int i = 0; i < L; ++i)
    {
        *packAptr++ = *a_p0_ptr++;
        *packAptr++ = *a_p1_ptr++;
        *packAptr++ = 0.0f;
        *packAptr++ = 0.0f;
    }
}

static void internalPackA1(int L, float* packA, float* a, int lda)
{
    float *packAptr = packA;
    float *a_p0_ptr;
    a_p0_ptr = a;
    for (int i = 0; i < L; ++i)
    {
        *packAptr++ = *a_p0_ptr++;
        *packAptr++ = +0.0f;
        *packAptr++ = +0.0f;
        *packAptr++ = +0.0f;
    }
}

static void internalPackB4(int L, float* packB, float* B, int ldb)
{
    float *bp = B;
    float *packBptr = packB;
    for (int i = 0; i < L; ++i)
    {
        vst1q_f32(packBptr, vld1q_f32(bp));
        packBptr += 4;
        bp += ldb;
    }
}

static void internalPackB8(int L, float* packB, float* B, int ldb)
{
    float *bp = B;
    float *packBptr = packB;
    for (int i = 0; i < L; ++i)
    {
        vst1q_f32(packBptr, vld1q_f32(bp));
        vst1q_f32(packBptr + 4, vld1q_f32(bp + 4));
        packBptr += 8;
        bp += ldb;
    }
}

void sgemm_4x1(int L, float *a, int lda, float* b, int ldb, float *c, int ldc)
{
    float barr[1];
    float *cptr = c;
    float32x4_t va;
    float32x4_t vc[1];
    vc[0] = vld1q_lane_f32(cptr,     vc[0], 0);
    cptr += ldc;
    vc[0] = vld1q_lane_f32(cptr,     vc[0], 1);
    cptr += ldc;
    vc[0] = vld1q_lane_f32(cptr,     vc[0], 2);
    cptr += ldc;
    vc[0] = vld1q_lane_f32(cptr,     vc[0], 3);

    float *aptr = a;
    float *bptr = b;
    for (int p = 0; p < L; ++p)
    {
        va = vld1q_f32(aptr);
        barr[0] = *(bptr + 0);

#if __aarch64__
        vc[0] = vfmaq_n_f32(vc[0], va, barr[0]);
#else
        vc[0] = vmlaq_n_f32(vc[0], va, barr[0]);
#endif // __aarch64__

        aptr += 4;
        bptr += ldb;
    }

    cptr = c;
    vst1q_lane_f32(cptr,     vc[0], 0);
    cptr += ldc;
    vst1q_lane_f32(cptr,     vc[0], 1);
    cptr += ldc;
    vst1q_lane_f32(cptr,     vc[0], 2);
    cptr += ldc;
    vst1q_lane_f32(cptr,     vc[0], 3);
}

void sgemm_4x2(int L, float *a, int lda, float* b, int ldb, float *c, int ldc)
{
    float barr[2];
    float *cptr = c;
    float32x4_t va;
    float32x4_t vc[2];
    vc[0] = vld1q_lane_f32(cptr,     vc[0], 0);
    vc[1] = vld1q_lane_f32(cptr + 1, vc[1], 0);
    cptr += ldc;
    vc[0] = vld1q_lane_f32(cptr,     vc[0], 1);
    vc[1] = vld1q_lane_f32(cptr + 1, vc[1], 1);
    cptr += ldc;
    vc[0] = vld1q_lane_f32(cptr,     vc[0], 2);
    vc[1] = vld1q_lane_f32(cptr + 1, vc[1], 2);
    cptr += ldc;
    vc[0] = vld1q_lane_f32(cptr,     vc[0], 3);
    vc[1] = vld1q_lane_f32(cptr + 1, vc[1], 3);

    float *aptr = a;
    float *bptr = b;
    for (int p = 0; p < L; ++p)
    {
        va = vld1q_f32(aptr);

        barr[0] = *(bptr + 0);
        barr[1] = *(bptr + 1);

#if __aarch64__
        vc[0] = vfmaq_n_f32(vc[0], va, barr[0]);
        vc[1] = vfmaq_n_f32(vc[1], va, barr[1]);
#else
        vc[0] = vmlaq_n_f32(vc[0], va, barr[0]);
        vc[1] = vmlaq_n_f32(vc[1], va, barr[1]);
#endif // __aarch64__

        aptr += 4;
        bptr += ldb;
    }

    cptr = c;
    vst1q_lane_f32(cptr,     vc[0], 0);
    vst1q_lane_f32(cptr + 1, vc[1], 0);
    cptr += ldc;
    vst1q_lane_f32(cptr,     vc[0], 1);
    vst1q_lane_f32(cptr + 1, vc[1], 1);
    cptr += ldc;
    vst1q_lane_f32(cptr,     vc[0], 2);
    vst1q_lane_f32(cptr + 1, vc[1], 2);
    cptr += ldc;
    vst1q_lane_f32(cptr,     vc[0], 3);
    vst1q_lane_f32(cptr + 1, vc[1], 3);
}

void sgemm_4x3(int L, float *a, int lda, float* b, int ldb, float *c, int ldc)
{
    float barr[3];
    float *cptr = c;
    float32x4_t va;
    float32x4_t vc[3];
    vc[0] = vld1q_lane_f32(cptr,     vc[0], 0);
    vc[1] = vld1q_lane_f32(cptr + 1, vc[1], 0);
    vc[2] = vld1q_lane_f32(cptr + 2, vc[2], 0);
    cptr += ldc;
    vc[0] = vld1q_lane_f32(cptr,     vc[0], 1);
    vc[1] = vld1q_lane_f32(cptr + 1, vc[1], 1);
    vc[2] = vld1q_lane_f32(cptr + 2, vc[2], 1);
    cptr += ldc;
    vc[0] = vld1q_lane_f32(cptr,     vc[0], 2);
    vc[1] = vld1q_lane_f32(cptr + 1, vc[1], 2);
    vc[2] = vld1q_lane_f32(cptr + 2, vc[2], 2);
    cptr += ldc;
    vc[0] = vld1q_lane_f32(cptr,     vc[0], 3);
    vc[1] = vld1q_lane_f32(cptr + 1, vc[1], 3);
    vc[2] = vld1q_lane_f32(cptr + 2, vc[2], 3);

    float *aptr = a;
    float *bptr = b;
    for (int p = 0; p < L; ++p)
    {
        va = vld1q_f32(aptr);

        barr[0] = *(bptr + 0);
        barr[1] = *(bptr + 1);
        barr[2] = *(bptr + 2);

#if __aarch64__
        vc[0] = vfmaq_n_f32(vc[0], va, barr[0]);
        vc[1] = vfmaq_n_f32(vc[1], va, barr[1]);
        vc[2] = vfmaq_n_f32(vc[2], va, barr[2]);
#else
        vc[0] = vmlaq_n_f32(vc[0], va, barr[0]);
        vc[1] = vmlaq_n_f32(vc[1], va, barr[1]);
        vc[2] = vmlaq_n_f32(vc[2], va, barr[2]);
#endif // __aarch64__

        aptr += 4;
        bptr += ldb;
    }

    cptr = c;
    vst1q_lane_f32(cptr,     vc[0], 0);
    vst1q_lane_f32(cptr + 1, vc[1], 0);
    vst1q_lane_f32(cptr + 2, vc[2], 0);
    cptr += ldc;
    vst1q_lane_f32(cptr,     vc[0], 1);
    vst1q_lane_f32(cptr + 1, vc[1], 1);
    vst1q_lane_f32(cptr + 2, vc[2], 1);
    cptr += ldc;
    vst1q_lane_f32(cptr,     vc[0], 2);
    vst1q_lane_f32(cptr + 1, vc[1], 2);
    vst1q_lane_f32(cptr + 2, vc[2], 2);
    cptr += ldc;
    vst1q_lane_f32(cptr,     vc[0], 3);
    vst1q_lane_f32(cptr + 1, vc[1], 3);
    vst1q_lane_f32(cptr + 2, vc[2], 3);
}

inline void sgemm_4x4(int L, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;
    float32x4_t vb;
    float32x4_t va0, va1, va2, va3;

    float32x4_t vc0 = vld1q_f32(cptr);
    cptr += ldc;
    float32x4_t vc1 = vld1q_f32(cptr);
    cptr += ldc;
    float32x4_t vc2 = vld1q_f32(cptr);
    cptr += ldc;
    float32x4_t vc3 = vld1q_f32(cptr);

    for (int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        va0 = vld1q_dup_f32(aptr);
        va1 = vld1q_dup_f32(aptr + 1);
        va2 = vld1q_dup_f32(aptr + 2);
        va3 = vld1q_dup_f32(aptr + 3);

#if __aarch64__
        vc0 = vfmaq_f32(vc0, va0, vb);
        vc1 = vfmaq_f32(vc1, va1, vb);
        vc2 = vfmaq_f32(vc2, va2, vb);
        vc3 = vfmaq_f32(vc3, va3, vb);
#else
        vc0 = vmlaq_f32(vc0, va0, vb);
        vc1 = vmlaq_f32(vc1, va1, vb);
        vc2 = vmlaq_f32(vc2, va2, vb);
        vc3 = vmlaq_f32(vc3, va3, vb);
#endif // __aarch64__

        bptr += ldb;
        aptr += 4;
    }
    cptr = c;
    vst1q_f32(cptr, vc0);
    cptr += ldc;
    vst1q_f32(cptr, vc1);
    cptr += ldc;
    vst1q_f32(cptr, vc2);
    cptr += ldc;
    vst1q_f32(cptr, vc3);
}

inline void sgemm_4x5(int L, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float b4;

    float32x4_t vb;
    float32x4_t va0, va1, va2, va3, va;
    float32x4_t vc0, vc1, vc2, vc3, vc4, vzero;
    vzero = vdupq_n_f32(0.0f);
    vc4 = vzero;
    vc0 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 0);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 1);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 2);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 3);


    for (int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        b4  = *(bptr + 4);

        va0 = vld1q_dup_f32(aptr);
        va1 = vld1q_dup_f32(aptr + 1);
        va2 = vld1q_dup_f32(aptr + 2);
        va3 = vld1q_dup_f32(aptr + 3);

        va = vld1q_f32(aptr);

#if __aarch64__
        vc0 = vfmaq_f32(vc0, va0, vb);
        vc1 = vfmaq_f32(vc1, va1, vb);
        vc2 = vfmaq_f32(vc2, va2, vb);
        vc3 = vfmaq_f32(vc3, va3, vb);

        vc4 = vfmaq_n_f32(vc4, va, b4);
#else
        vc0 = vmlaq_f32(vc0, va0, vb);
        vc1 = vmlaq_f32(vc1, va1, vb);
        vc2 = vmlaq_f32(vc2, va2, vb);
        vc3 = vmlaq_f32(vc3, va3, vb);

        vc4 = vmlaq_n_f32(vc4, va, b4);
#endif // __aarch64__

        bptr += ldb;
        aptr += 4;
    }
    cptr = c;
    vst1q_f32(cptr, vc0);
    vst1q_lane_f32(cptr + 4, vc4, 0);
    cptr += ldc;
    vst1q_f32(cptr, vc1);
    vst1q_lane_f32(cptr + 4, vc4, 1);
    cptr += ldc;
    vst1q_f32(cptr, vc2);
    vst1q_lane_f32(cptr + 4, vc4, 2);
    cptr += ldc;
    vst1q_f32(cptr, vc3);
    vst1q_lane_f32(cptr + 4, vc4, 3);
}

inline void sgemm_4x6(int L, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float b4, b5;

    float32x4_t vb;
    float32x4_t va0, va1, va2, va3, va;
    float32x4_t vc0, vc1, vc2, vc3, vc4, vc5, vzero;
    vzero = vdupq_n_f32(0.0f);
    vc4 = vzero;
    vc5 = vzero;
    vc0 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 0);
    vc5 =  vld1q_lane_f32(cptr + 5, vc5, 0);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 1);
    vc5 =  vld1q_lane_f32(cptr + 5, vc5, 1);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 2);
    vc5 =  vld1q_lane_f32(cptr + 5, vc5, 2);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 3);
    vc5 =  vld1q_lane_f32(cptr + 5, vc5, 3);


    for (int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        b4  = *(bptr + 4);
        b5  = *(bptr + 5);

        va0 = vld1q_dup_f32(aptr);
        va1 = vld1q_dup_f32(aptr + 1);
        va2 = vld1q_dup_f32(aptr + 2);
        va3 = vld1q_dup_f32(aptr + 3);

        va = vld1q_f32(aptr);

#if __aarch64__
        vc0 = vfmaq_f32(vc0, va0, vb);
        vc1 = vfmaq_f32(vc1, va1, vb);
        vc2 = vfmaq_f32(vc2, va2, vb);
        vc3 = vfmaq_f32(vc3, va3, vb);

        vc4 = vfmaq_n_f32(vc4, va, b4);
        vc5 = vfmaq_n_f32(vc5, va, b5);
#else
        vc0 = vmlaq_f32(vc0, va0, vb);
        vc1 = vmlaq_f32(vc1, va1, vb);
        vc2 = vmlaq_f32(vc2, va2, vb);
        vc3 = vmlaq_f32(vc3, va3, vb);

        vc4 = vmlaq_n_f32(vc4, va, b4);
        vc5 = vmlaq_n_f32(vc5, va, b5);
#endif // __aarch64__

        bptr += ldb;
        aptr += 4;
    }
    cptr = c;
    vst1q_f32(cptr, vc0);
    vst1q_lane_f32(cptr + 4, vc4, 0);
    vst1q_lane_f32(cptr + 5, vc5, 0);
    cptr += ldc;
    vst1q_f32(cptr, vc1);
    vst1q_lane_f32(cptr + 4, vc4, 1);
    vst1q_lane_f32(cptr + 5, vc5, 1);
    cptr += ldc;
    vst1q_f32(cptr, vc2);
    vst1q_lane_f32(cptr + 4, vc4, 2);
    vst1q_lane_f32(cptr + 5, vc5, 2);
    cptr += ldc;
    vst1q_f32(cptr, vc3);
    vst1q_lane_f32(cptr + 4, vc4, 3);
    vst1q_lane_f32(cptr + 5, vc5, 3);
}

inline void sgemm_4x7(int L, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float b4, b5, b6;

    float32x4_t vb;
    float32x4_t va0, va1, va2, va3, va;
    float32x4_t vc0, vc1, vc2, vc3, vc4, vc5, vc6, vzero;
    vzero = vdupq_n_f32(0.0f);
    vc4 = vc5 = vc6 = vzero;
    vc0 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 0);
    vc5 =  vld1q_lane_f32(cptr + 5, vc5, 0);
    vc6 =  vld1q_lane_f32(cptr + 6, vc6, 0);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 1);
    vc5 =  vld1q_lane_f32(cptr + 5, vc5, 1);
    vc6 =  vld1q_lane_f32(cptr + 6, vc6, 1);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 2);
    vc5 =  vld1q_lane_f32(cptr + 5, vc5, 2);
    vc6 =  vld1q_lane_f32(cptr + 6, vc6, 2);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 3);
    vc5 =  vld1q_lane_f32(cptr + 5, vc5, 3);
    vc6 =  vld1q_lane_f32(cptr + 6, vc6, 3);


    for (int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        b4  = *(bptr + 4);
        b5  = *(bptr + 5);
        b6  = *(bptr + 6);

        va0 = vld1q_dup_f32(aptr);
        va1 = vld1q_dup_f32(aptr + 1);
        va2 = vld1q_dup_f32(aptr + 2);
        va3 = vld1q_dup_f32(aptr + 3);

        va = vld1q_f32(aptr);

#if __aarch64__
        vc0 = vfmaq_f32(vc0, va0, vb);
        vc1 = vfmaq_f32(vc1, va1, vb);
        vc2 = vfmaq_f32(vc2, va2, vb);
        vc3 = vfmaq_f32(vc3, va3, vb);

        vc4 = vfmaq_n_f32(vc4, va, b4);
        vc5 = vfmaq_n_f32(vc5, va, b5);
        vc6 = vfmaq_n_f32(vc6, va, b6);
#else
        vc0 = vmlaq_f32(vc0, va0, vb);
        vc1 = vmlaq_f32(vc1, va1, vb);
        vc2 = vmlaq_f32(vc2, va2, vb);
        vc3 = vmlaq_f32(vc3, va3, vb);

        vc4 = vmlaq_n_f32(vc4, va, b4);
        vc5 = vmlaq_n_f32(vc5, va, b5);
        vc6 = vmlaq_n_f32(vc6, va, b6);
#endif // __aarch64__

        bptr += ldb;
        aptr += 4;
    }
    cptr = c;
    vst1q_f32(cptr, vc0);
    vst1q_lane_f32(cptr + 4, vc4, 0);
    vst1q_lane_f32(cptr + 5, vc5, 0);
    vst1q_lane_f32(cptr + 6, vc6, 0);
    cptr += ldc;
    vst1q_f32(cptr, vc1);
    vst1q_lane_f32(cptr + 4, vc4, 1);
    vst1q_lane_f32(cptr + 5, vc5, 1);
    vst1q_lane_f32(cptr + 6, vc6, 1);
    cptr += ldc;
    vst1q_f32(cptr, vc2);
    vst1q_lane_f32(cptr + 4, vc4, 2);
    vst1q_lane_f32(cptr + 5, vc5, 2);
    vst1q_lane_f32(cptr + 6, vc6, 2);
    cptr += ldc;
    vst1q_f32(cptr, vc3);
    vst1q_lane_f32(cptr + 4, vc4, 3);
    vst1q_lane_f32(cptr + 5, vc5, 3);
    vst1q_lane_f32(cptr + 6, vc6, 3);
}

void sgemm_8x1(int L, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;
    float b4;
    float32x4_t vzero = vdupq_n_f32(0.0f);
    float32x4_t vb;
    float32x4_t va0, va1;
    float32x4_t vc4;
    //next 4 rows
    float32x4_t vcE;
    //vc 4 5 6 and E F G hold column values.
    vc4 = vcE = vzero;
    vc4 =  vld1q_lane_f32(cptr, vc4, 0);
    cptr += ldc;
    vc4 =  vld1q_lane_f32(cptr, vc4, 1);
    cptr += ldc;
    vc4 =  vld1q_lane_f32(cptr, vc4, 2);
    cptr += ldc;
    vc4 =  vld1q_lane_f32(cptr, vc4, 3);
    cptr += ldc;
    vcE =  vld1q_lane_f32(cptr, vcE, 0);
    cptr += ldc;
    vcE =  vld1q_lane_f32(cptr, vcE, 1);
    cptr += ldc;
    vcE =  vld1q_lane_f32(cptr, vcE, 2);
    cptr += ldc;
    vcE =  vld1q_lane_f32(cptr, vcE, 3);

    for (int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        b4  = *(bptr);
        va0 = vld1q_f32(aptr);
        va1 = vld1q_f32(aptr + 4);

        //A row in A multiplies a single value in B by column
#if __aarch64__
        vc4 = vfmaq_n_f32(vc4, va0, b4);
        vcE = vfmaq_n_f32(vcE, va1, b4);
#else
        vc4 = vmlaq_n_f32(vc4, va0, b4);
        vcE = vmlaq_n_f32(vcE, va1, b4);
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    vst1q_lane_f32(cptr, vc4, 0);
    cptr += ldc;
    vst1q_lane_f32(cptr, vc4, 1);
    cptr += ldc;
    vst1q_lane_f32(cptr, vc4, 2);
    cptr += ldc;
    vst1q_lane_f32(cptr, vc4, 3);
    cptr += ldc;
    vst1q_lane_f32(cptr, vcE, 0);
    cptr += ldc;
    vst1q_lane_f32(cptr, vcE, 1);
    cptr += ldc;
    vst1q_lane_f32(cptr, vcE, 2);
    cptr += ldc;
    vst1q_lane_f32(cptr, vcE, 3);

}
void sgemm_8x2(int L, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;
    float b4, b5;
    float32x4_t vzero = vdupq_n_f32(0.0f);
    float32x4_t vb;
    float32x4_t va0, va1;
    float32x4_t vc4, vc5;
    //next 4 rows
    float32x4_t vcE, vcF;
    vc4 = vc5 = vcE = vcF = vzero;
    //vc 4 5 6 and E F G hold column values.
    vc4 =  vld1q_lane_f32(cptr + 0, vc4, 0);
    vc5 =  vld1q_lane_f32(cptr + 1, vc5, 0);
    cptr += ldc;
    vc4 =  vld1q_lane_f32(cptr + 0, vc4, 1);
    vc5 =  vld1q_lane_f32(cptr + 1, vc5, 1);
    cptr += ldc;
    vc4 =  vld1q_lane_f32(cptr + 0, vc4, 2);
    vc5 =  vld1q_lane_f32(cptr + 1, vc5, 2);
    cptr += ldc;
    vc4 =  vld1q_lane_f32(cptr + 0, vc4, 3);
    vc5 =  vld1q_lane_f32(cptr + 1, vc5, 3);
    cptr += ldc;
    vcE =  vld1q_lane_f32(cptr + 0, vcE, 0);
    vcF =  vld1q_lane_f32(cptr + 1, vcF, 0);
    cptr += ldc;
    vcE =  vld1q_lane_f32(cptr + 0, vcE, 1);
    vcF =  vld1q_lane_f32(cptr + 1, vcF, 1);
    cptr += ldc;
    vcE =  vld1q_lane_f32(cptr + 0, vcE, 2);
    vcF =  vld1q_lane_f32(cptr + 1, vcF, 2);
    cptr += ldc;
    vcE =  vld1q_lane_f32(cptr + 0, vcE, 3);
    vcF =  vld1q_lane_f32(cptr + 1, vcF, 3);

    for (int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        b4  = *(bptr);
        b5  = *(bptr + 1);
        va0 = vld1q_f32(aptr);
        va1 = vld1q_f32(aptr + 4);

        //A row in A multiplies a single value in B by column
#if __aarch64__
        vc4 = vfmaq_n_f32(vc4, va0, b4);
        vc5 = vfmaq_n_f32(vc5, va0, b5);

        vcE = vfmaq_n_f32(vcE, va1, b4);
        vcF = vfmaq_n_f32(vcF, va1, b5);
#else
        vc4 = vmlaq_n_f32(vc4, va0, b4);
        vc5 = vmlaq_n_f32(vc5, va0, b5);

        vcE = vmlaq_n_f32(vcE, va1, b4);
        vcF = vmlaq_n_f32(vcF, va1, b5);
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    vst1q_lane_f32(cptr + 0, vc4, 0);
    vst1q_lane_f32(cptr + 1, vc5, 0);
    cptr += ldc;
    vst1q_lane_f32(cptr + 0, vc4, 1);
    vst1q_lane_f32(cptr + 1, vc5, 1);
    cptr += ldc;
    vst1q_lane_f32(cptr + 0, vc4, 2);
    vst1q_lane_f32(cptr + 1, vc5, 2);
    cptr += ldc;
    vst1q_lane_f32(cptr + 0, vc4, 3);
    vst1q_lane_f32(cptr + 1, vc5, 3);
    cptr += ldc;
    vst1q_lane_f32(cptr + 0, vcE, 0);
    vst1q_lane_f32(cptr + 1, vcF, 0);
    cptr += ldc;
    vst1q_lane_f32(cptr + 0, vcE, 1);
    vst1q_lane_f32(cptr + 1, vcF, 1);
    cptr += ldc;
    vst1q_lane_f32(cptr + 0, vcE, 2);
    vst1q_lane_f32(cptr + 1, vcF, 2);
    cptr += ldc;
    vst1q_lane_f32(cptr + 0, vcE, 3);
    vst1q_lane_f32(cptr + 1, vcF, 3);
}

void sgemm_8x3(int L, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;
    float b4, b5, b6;
    float32x4_t vb;
    float32x4_t va0, va1;
    float32x4_t vc4, vc5, vc6;
    //next 4 rows
    float32x4_t vcE, vcF, vcG;
    float32x4_t vzero = vdupq_n_f32(0.0f);
    vc4 = vc5 = vc6 = vcE = vcF = vcG = vzero;
    //vc 4 5 6 and E F G hold column values.
    vc4 =  vld1q_lane_f32(cptr + 0, vc4, 0);
    vc5 =  vld1q_lane_f32(cptr + 1, vc5, 0);
    vc6 =  vld1q_lane_f32(cptr + 2, vc6, 0);
    cptr += ldc;
    vc4 =  vld1q_lane_f32(cptr + 0, vc4, 1);
    vc5 =  vld1q_lane_f32(cptr + 1, vc5, 1);
    vc6 =  vld1q_lane_f32(cptr + 2, vc6, 1);
    cptr += ldc;
    vc4 =  vld1q_lane_f32(cptr + 0, vc4, 2);
    vc5 =  vld1q_lane_f32(cptr + 1, vc5, 2);
    vc6 =  vld1q_lane_f32(cptr + 2, vc6, 2);
    cptr += ldc;
    vc4 =  vld1q_lane_f32(cptr + 0, vc4, 3);
    vc5 =  vld1q_lane_f32(cptr + 1, vc5, 3);
    vc6 =  vld1q_lane_f32(cptr + 2, vc6, 3);
    cptr += ldc;
    vcE =  vld1q_lane_f32(cptr + 0, vcE, 0);
    vcF =  vld1q_lane_f32(cptr + 1, vcF, 0);
    vcG =  vld1q_lane_f32(cptr + 2, vcG, 0);
    cptr += ldc;
    vcE =  vld1q_lane_f32(cptr + 0, vcE, 1);
    vcF =  vld1q_lane_f32(cptr + 1, vcF, 1);
    vcG =  vld1q_lane_f32(cptr + 2, vcG, 1);
    cptr += ldc;
    vcE =  vld1q_lane_f32(cptr + 0, vcE, 2);
    vcF =  vld1q_lane_f32(cptr + 1, vcF, 2);
    vcG =  vld1q_lane_f32(cptr + 2, vcG, 2);
    cptr += ldc;
    vcE =  vld1q_lane_f32(cptr + 0, vcE, 3);
    vcF =  vld1q_lane_f32(cptr + 1, vcF, 3);
    vcG =  vld1q_lane_f32(cptr + 2, vcG, 3);

    for (int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        b4  = *(bptr);
        b5  = *(bptr + 1);
        b6  = *(bptr + 2);
        va0 = vld1q_f32(aptr);
        va1 = vld1q_f32(aptr + 4);

#if __aarch64__
        //A row in A multiplies a single value in B by column
        vc4 = vfmaq_n_f32(vc4, va0, b4);
        vc5 = vfmaq_n_f32(vc5, va0, b5);
        vc6 = vfmaq_n_f32(vc6, va0, b6);

        vcE = vfmaq_n_f32(vcE, va1, b4);
        vcF = vfmaq_n_f32(vcF, va1, b5);
        vcG = vfmaq_n_f32(vcG, va1, b6);
#else
        vc4 = vmlaq_n_f32(vc4, va0, b4);
        vc5 = vmlaq_n_f32(vc5, va0, b5);
        vc6 = vmlaq_n_f32(vc6, va0, b6);

        vcE = vmlaq_n_f32(vcE, va1, b4);
        vcF = vmlaq_n_f32(vcF, va1, b5);
        vcG = vmlaq_n_f32(vcG, va1, b6);
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    vst1q_lane_f32(cptr + 0, vc4, 0);
    vst1q_lane_f32(cptr + 1, vc5, 0);
    vst1q_lane_f32(cptr + 2, vc6, 0);
    cptr += ldc;
    vst1q_lane_f32(cptr + 0, vc4, 1);
    vst1q_lane_f32(cptr + 1, vc5, 1);
    vst1q_lane_f32(cptr + 2, vc6, 1);
    cptr += ldc;
    vst1q_lane_f32(cptr + 0, vc4, 2);
    vst1q_lane_f32(cptr + 1, vc5, 2);
    vst1q_lane_f32(cptr + 2, vc6, 2);
    cptr += ldc;
    vst1q_lane_f32(cptr + 0, vc4, 3);
    vst1q_lane_f32(cptr + 1, vc5, 3);
    vst1q_lane_f32(cptr + 2, vc6, 3);
    cptr += ldc;
    vst1q_lane_f32(cptr + 0, vcE, 0);
    vst1q_lane_f32(cptr + 1, vcF, 0);
    vst1q_lane_f32(cptr + 2, vcG, 0);
    cptr += ldc;
    vst1q_lane_f32(cptr + 0, vcE, 1);
    vst1q_lane_f32(cptr + 1, vcF, 1);
    vst1q_lane_f32(cptr + 2, vcG, 1);
    cptr += ldc;
    vst1q_lane_f32(cptr + 0, vcE, 2);
    vst1q_lane_f32(cptr + 1, vcF, 2);
    vst1q_lane_f32(cptr + 2, vcG, 2);
    cptr += ldc;
    vst1q_lane_f32(cptr + 0, vcE, 3);
    vst1q_lane_f32(cptr + 1, vcF, 3);
    vst1q_lane_f32(cptr + 2, vcG, 3);
}

void sgemm_8x4(int L, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;


    float32x4_t vb;
    float32x4_t va0, va1;
    float32x4_t vc0, vc1, vc2, vc3;
    //next 4 rows
    float32x4_t vcA, vcB, vcC, vcD;

    //vc0 1 2 3 and A B C D hold row values.
    vc0 = vld1q_f32(cptr);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    cptr += ldc;
    vcA = vld1q_f32(cptr);
    cptr += ldc;
    vcB = vld1q_f32(cptr);
    cptr += ldc;
    vcC = vld1q_f32(cptr);
    cptr += ldc;
    vcD = vld1q_f32(cptr);

    for (int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        va0 = vld1q_f32(aptr);
        va1 = vld1q_f32(aptr + 4);

#if __aarch64__
        vc0 = vfmaq_laneq_f32(vc0, vb, va0, 0);
        vc1 = vfmaq_laneq_f32(vc1, vb, va0, 1);
        vc2 = vfmaq_laneq_f32(vc2, vb, va0, 2);
        vc3 = vfmaq_laneq_f32(vc3, vb, va0, 3);

        vcA = vfmaq_laneq_f32(vcA, vb, va1, 0);
        vcB = vfmaq_laneq_f32(vcB, vb, va1, 1);
        vcC = vfmaq_laneq_f32(vcC, vb, va1, 2);
        vcD = vfmaq_laneq_f32(vcD, vb, va1, 3);
#else
        vc0 = vmlaq_f32(vc0, vb, vld1q_dup_f32(aptr + 0));
        vc1 = vmlaq_f32(vc1, vb, vld1q_dup_f32(aptr + 1));
        vc2 = vmlaq_f32(vc2, vb, vld1q_dup_f32(aptr + 2));
        vc3 = vmlaq_f32(vc3, vb, vld1q_dup_f32(aptr + 3));

        vcA = vmlaq_f32(vcA, vb, vld1q_dup_f32(aptr + 4));
        vcB = vmlaq_f32(vcB, vb, vld1q_dup_f32(aptr + 5));
        vcC = vmlaq_f32(vcC, vb, vld1q_dup_f32(aptr + 6));
        vcD = vmlaq_f32(vcD, vb, vld1q_dup_f32(aptr + 7));
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    vst1q_f32(cptr, vc0);
    cptr += ldc;
    vst1q_f32(cptr, vc1);
    cptr += ldc;
    vst1q_f32(cptr, vc2);
    cptr += ldc;
    vst1q_f32(cptr, vc3);
    cptr += ldc;
    vst1q_f32(cptr, vcA);
    cptr += ldc;
    vst1q_f32(cptr, vcB);
    cptr += ldc;
    vst1q_f32(cptr, vcC);
    cptr += ldc;
    vst1q_f32(cptr, vcD);
}

void sgemm_8x5(int L, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float b4;

    float32x4_t vb;
    float32x4_t va0, va1;
    float32x4_t vc0, vc1, vc2, vc3, vc4;
    //next 4 rows
    float32x4_t vcA, vcB, vcC, vcD, vcE;
    float32x4_t vzero = vdupq_n_f32(0.0f);
    vc4 = vcE = vzero;
    //vc0 1 2 3 and A B C D hold row values.
    vc0 = vld1q_f32(cptr);
    //vc 4 5 6 and E F G hold column values.
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 0);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 1);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 2);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 3);
    cptr += ldc;
    vcA = vld1q_f32(cptr);
    vcE =  vld1q_lane_f32(cptr + 4, vcE, 0);
    cptr += ldc;
    vcB = vld1q_f32(cptr);
    vcE =  vld1q_lane_f32(cptr + 4, vcE, 1);
    cptr += ldc;
    vcC = vld1q_f32(cptr);
    vcE =  vld1q_lane_f32(cptr + 4, vcE, 2);
    cptr += ldc;
    vcD = vld1q_f32(cptr);
    vcE =  vld1q_lane_f32(cptr + 4, vcE, 3);

    for (int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        b4  = *(bptr + 4);
        va0 = vld1q_f32(aptr);
        va1 = vld1q_f32(aptr + 4);

#if __aarch64__
        vc0 = vfmaq_laneq_f32(vc0, vb, va0, 0);
        vc1 = vfmaq_laneq_f32(vc1, vb, va0, 1);
        vc2 = vfmaq_laneq_f32(vc2, vb, va0, 2);
        vc3 = vfmaq_laneq_f32(vc3, vb, va0, 3);

        vcA = vfmaq_laneq_f32(vcA, vb, va1, 0);
        vcB = vfmaq_laneq_f32(vcB, vb, va1, 1);
        vcC = vfmaq_laneq_f32(vcC, vb, va1, 2);
        vcD = vfmaq_laneq_f32(vcD, vb, va1, 3);

        //A row in A multiplies a single value in B by column
        vc4 = vfmaq_n_f32(vc4, va0, b4);

        vcE = vfmaq_n_f32(vcE, va1, b4);
#else
        vc0 = vmlaq_f32(vc0, vb, vld1q_dup_f32(aptr + 0));
        vc1 = vmlaq_f32(vc1, vb, vld1q_dup_f32(aptr + 1));
        vc2 = vmlaq_f32(vc2, vb, vld1q_dup_f32(aptr + 2));
        vc3 = vmlaq_f32(vc3, vb, vld1q_dup_f32(aptr + 3));

        vcA = vmlaq_f32(vcA, vb, vld1q_dup_f32(aptr + 4));
        vcB = vmlaq_f32(vcB, vb, vld1q_dup_f32(aptr + 5));
        vcC = vmlaq_f32(vcC, vb, vld1q_dup_f32(aptr + 6));
        vcD = vmlaq_f32(vcD, vb, vld1q_dup_f32(aptr + 7));

        //A row in A multiplies a single value in B by column
        vc4 = vmlaq_n_f32(vc4, va0, b4);

        vcE = vmlaq_n_f32(vcE, va1, b4);
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    vst1q_f32(cptr, vc0);
    vst1q_lane_f32(cptr + 4, vc4, 0);
    cptr += ldc;
    vst1q_f32(cptr, vc1);
    vst1q_lane_f32(cptr + 4, vc4, 1);
    cptr += ldc;
    vst1q_f32(cptr, vc2);
    vst1q_lane_f32(cptr + 4, vc4, 2);
    cptr += ldc;
    vst1q_f32(cptr, vc3);
    vst1q_lane_f32(cptr + 4, vc4, 3);
    cptr += ldc;
    vst1q_f32(cptr, vcA);
    vst1q_lane_f32(cptr + 4, vcE, 0);
    cptr += ldc;
    vst1q_f32(cptr, vcB);
    vst1q_lane_f32(cptr + 4, vcE, 1);
    cptr += ldc;
    vst1q_f32(cptr, vcC);
    vst1q_lane_f32(cptr + 4, vcE, 2);
    cptr += ldc;
    vst1q_f32(cptr, vcD);
    vst1q_lane_f32(cptr + 4, vcE, 3);
}

void sgemm_8x6(int L, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float b4, b5;
    float32x4_t vb;
    float32x4_t va0, va1;
    float32x4_t vc0, vc1, vc2, vc3, vc4, vc5;
    //next 4 rows
    float32x4_t vcA, vcB, vcC, vcD, vcE, vcF;
    float32x4_t vzero = vdupq_n_f32(0.0f);
    vc4 = vc5 = vcE = vcF = vzero;
    //vc0 1 2 3 and A B C D hold row values.
    vc0 = vld1q_f32(cptr);
    //vc 4 5 6 and E F G hold column values.
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 0);
    vc5 =  vld1q_lane_f32(cptr + 5, vc5, 0);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 1);
    vc5 =  vld1q_lane_f32(cptr + 5, vc5, 1);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 2);
    vc5 =  vld1q_lane_f32(cptr + 5, vc5, 2);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 3);
    vc5 =  vld1q_lane_f32(cptr + 5, vc5, 3);
    cptr += ldc;
    vcA = vld1q_f32(cptr);
    vcE =  vld1q_lane_f32(cptr + 4, vcE, 0);
    vcF =  vld1q_lane_f32(cptr + 5, vcF, 0);
    cptr += ldc;
    vcB = vld1q_f32(cptr);
    vcE =  vld1q_lane_f32(cptr + 4, vcE, 1);
    vcF =  vld1q_lane_f32(cptr + 5, vcF, 1);
    cptr += ldc;
    vcC = vld1q_f32(cptr);
    vcE =  vld1q_lane_f32(cptr + 4, vcE, 2);
    vcF =  vld1q_lane_f32(cptr + 5, vcF, 2);
    cptr += ldc;
    vcD = vld1q_f32(cptr);
    vcE =  vld1q_lane_f32(cptr + 4, vcE, 3);
    vcF =  vld1q_lane_f32(cptr + 5, vcF, 3);


    for (int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        b4  = *(bptr + 4);
        b5  = *(bptr + 5);
        va0 = vld1q_f32(aptr);
        va1 = vld1q_f32(aptr + 4);

#if __aarch64__
        vc0 = vfmaq_laneq_f32(vc0, vb, va0, 0);
        vc1 = vfmaq_laneq_f32(vc1, vb, va0, 1);
        vc2 = vfmaq_laneq_f32(vc2, vb, va0, 2);
        vc3 = vfmaq_laneq_f32(vc3, vb, va0, 3);

        vcA = vfmaq_laneq_f32(vcA, vb, va1, 0);
        vcB = vfmaq_laneq_f32(vcB, vb, va1, 1);
        vcC = vfmaq_laneq_f32(vcC, vb, va1, 2);
        vcD = vfmaq_laneq_f32(vcD, vb, va1, 3);

        //A row in A multiplies a single value in B by column
        vc4 = vfmaq_n_f32(vc4, va0, b4);
        vc5 = vfmaq_n_f32(vc5, va0, b5);

        vcE = vfmaq_n_f32(vcE, va1, b4);
        vcF = vfmaq_n_f32(vcF, va1, b5);
#else
        vc0 = vmlaq_f32(vc0, vb, vld1q_dup_f32(aptr + 0));
        vc1 = vmlaq_f32(vc1, vb, vld1q_dup_f32(aptr + 0));
        vc2 = vmlaq_f32(vc2, vb, vld1q_dup_f32(aptr + 0));
        vc3 = vmlaq_f32(vc3, vb, vld1q_dup_f32(aptr + 0));

        vcA = vmlaq_f32(vcA, vb, vld1q_dup_f32(aptr + 4));
        vcB = vmlaq_f32(vcB, vb, vld1q_dup_f32(aptr + 5));
        vcC = vmlaq_f32(vcC, vb, vld1q_dup_f32(aptr + 6));
        vcD = vmlaq_f32(vcD, vb, vld1q_dup_f32(aptr + 7));

        //A row in A multiplies a single value in B by column
        vc4 = vmlaq_n_f32(vc4, va0, b4);
        vc5 = vmlaq_n_f32(vc5, va0, b5);

        vcE = vmlaq_n_f32(vcE, va1, b4);
        vcF = vmlaq_n_f32(vcF, va1, b5);
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }
    cptr = c;
    vst1q_f32(cptr, vc0);
    vst1q_lane_f32(cptr + 4, vc4, 0);
    vst1q_lane_f32(cptr + 5, vc5, 0);
    cptr += ldc;
    vst1q_f32(cptr, vc1);
    vst1q_lane_f32(cptr + 4, vc4, 1);
    vst1q_lane_f32(cptr + 5, vc5, 1);
    cptr += ldc;
    vst1q_f32(cptr, vc2);
    vst1q_lane_f32(cptr + 4, vc4, 2);
    vst1q_lane_f32(cptr + 5, vc5, 2);
    cptr += ldc;
    vst1q_f32(cptr, vc3);
    vst1q_lane_f32(cptr + 4, vc4, 3);
    vst1q_lane_f32(cptr + 5, vc5, 3);
    cptr += ldc;
    vst1q_f32(cptr, vcA);
    vst1q_lane_f32(cptr + 4, vcE, 0);
    vst1q_lane_f32(cptr + 5, vcF, 0);
    cptr += ldc;
    vst1q_f32(cptr, vcB);
    vst1q_lane_f32(cptr + 4, vcE, 1);
    vst1q_lane_f32(cptr + 5, vcF, 1);
    cptr += ldc;
    vst1q_f32(cptr, vcC);
    vst1q_lane_f32(cptr + 4, vcE, 2);
    vst1q_lane_f32(cptr + 5, vcF, 2);
    cptr += ldc;
    vst1q_f32(cptr, vcD);
    vst1q_lane_f32(cptr + 4, vcE, 3);
    vst1q_lane_f32(cptr + 5, vcF, 3);
}

void sgemm_8x7(int L, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float b4, b5, b6;
    float32x4_t vb;
    float32x4_t va0, va1;
    float32x4_t vc0, vc1, vc2, vc3, vc4, vc5, vc6;
    //next 4 rows
    float32x4_t vcA, vcB, vcC, vcD, vcE, vcF, vcG;
    float32x4_t vzero = vdupq_n_f32(0.0f);
    vc4 = vc5 = vc6 = vcE = vcF = vcG = vzero;
    //vc0 1 2 3 and A B C D hold row values.
    vc0 = vld1q_f32(cptr);
    //vc 4 5 6 and E F G hold column values.
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 0);
    vc5 =  vld1q_lane_f32(cptr + 5, vc5, 0);
    vc6 =  vld1q_lane_f32(cptr + 6, vc6, 0);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 1);
    vc5 =  vld1q_lane_f32(cptr + 5, vc5, 1);
    vc6 =  vld1q_lane_f32(cptr + 6, vc6, 1);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 2);
    vc5 =  vld1q_lane_f32(cptr + 5, vc5, 2);
    vc6 =  vld1q_lane_f32(cptr + 6, vc6, 2);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vc4 =  vld1q_lane_f32(cptr + 4, vc4, 3);
    vc5 =  vld1q_lane_f32(cptr + 5, vc5, 3);
    vc6 =  vld1q_lane_f32(cptr + 6, vc6, 3);
    cptr += ldc;
    vcA = vld1q_f32(cptr);
    vcE =  vld1q_lane_f32(cptr + 4, vcE, 0);
    vcF =  vld1q_lane_f32(cptr + 5, vcF, 0);
    vcG =  vld1q_lane_f32(cptr + 6, vcG, 0);
    cptr += ldc;
    vcB = vld1q_f32(cptr);
    vcE =  vld1q_lane_f32(cptr + 4, vcE, 1);
    vcF =  vld1q_lane_f32(cptr + 5, vcF, 1);
    vcG =  vld1q_lane_f32(cptr + 6, vcG, 1);
    cptr += ldc;
    vcC = vld1q_f32(cptr);
    vcE =  vld1q_lane_f32(cptr + 4, vcE, 2);
    vcF =  vld1q_lane_f32(cptr + 5, vcF, 2);
    vcG =  vld1q_lane_f32(cptr + 6, vcG, 2);
    cptr += ldc;
    vcD = vld1q_f32(cptr);
    vcE =  vld1q_lane_f32(cptr + 4, vcE, 3);
    vcF =  vld1q_lane_f32(cptr + 5, vcF, 3);
    vcG =  vld1q_lane_f32(cptr + 6, vcG, 3);

    for (int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        b4  = *(bptr + 4);
        b5  = *(bptr + 5);
        b6  = *(bptr + 6);
        va0 = vld1q_f32(aptr);
        va1 = vld1q_f32(aptr + 4);

#if __aarch64__
        vc0 = vfmaq_laneq_f32(vc0, vb, va0, 0);
        vc1 = vfmaq_laneq_f32(vc1, vb, va0, 1);
        vc2 = vfmaq_laneq_f32(vc2, vb, va0, 2);
        vc3 = vfmaq_laneq_f32(vc3, vb, va0, 3);

        vcA = vfmaq_laneq_f32(vcA, vb, va1, 0);
        vcB = vfmaq_laneq_f32(vcB, vb, va1, 1);
        vcC = vfmaq_laneq_f32(vcC, vb, va1, 2);
        vcD = vfmaq_laneq_f32(vcD, vb, va1, 3);

        //A row in A multiplies a single value in B by column
        vc4 = vfmaq_n_f32(vc4, va0, b4);
        vc5 = vfmaq_n_f32(vc5, va0, b5);
        vc6 = vfmaq_n_f32(vc6, va0, b6);

        vcE = vfmaq_n_f32(vcE, va1, b4);
        vcF = vfmaq_n_f32(vcF, va1, b5);
        vcG = vfmaq_n_f32(vcG, va1, b6);
#else
        vc0 = vmlaq_f32(vc0, vb, vld1q_dup_f32(aptr + 0));
        vc1 = vmlaq_f32(vc1, vb, vld1q_dup_f32(aptr + 1));
        vc2 = vmlaq_f32(vc2, vb, vld1q_dup_f32(aptr + 2));
        vc3 = vmlaq_f32(vc3, vb, vld1q_dup_f32(aptr + 3));

        vcA = vmlaq_f32(vcA, vb, vld1q_dup_f32(aptr + 4));
        vcB = vmlaq_f32(vcB, vb, vld1q_dup_f32(aptr + 5));
        vcC = vmlaq_f32(vcC, vb, vld1q_dup_f32(aptr + 6));
        vcD = vmlaq_f32(vcD, vb, vld1q_dup_f32(aptr + 7));

        //A row in A multiplies a single value in B by column
        vc4 = vmlaq_n_f32(vc4, va0, b4);
        vc5 = vmlaq_n_f32(vc5, va0, b5);
        vc6 = vmlaq_n_f32(vc6, va0, b6);

        vcE = vmlaq_n_f32(vcE, va1, b4);
        vcF = vmlaq_n_f32(vcF, va1, b5);
        vcG = vmlaq_n_f32(vcG, va1, b6);
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }
    cptr = c;
    vst1q_f32(cptr, vc0);
    vst1q_lane_f32(cptr + 4, vc4, 0);
    vst1q_lane_f32(cptr + 5, vc5, 0);
    vst1q_lane_f32(cptr + 6, vc6, 0);
    cptr += ldc;
    vst1q_f32(cptr, vc1);
    vst1q_lane_f32(cptr + 4, vc4, 1);
    vst1q_lane_f32(cptr + 5, vc5, 1);
    vst1q_lane_f32(cptr + 6, vc6, 1);
    cptr += ldc;
    vst1q_f32(cptr, vc2);
    vst1q_lane_f32(cptr + 4, vc4, 2);
    vst1q_lane_f32(cptr + 5, vc5, 2);
    vst1q_lane_f32(cptr + 6, vc6, 2);
    cptr += ldc;
    vst1q_f32(cptr, vc3);
    vst1q_lane_f32(cptr + 4, vc4, 3);
    vst1q_lane_f32(cptr + 5, vc5, 3);
    vst1q_lane_f32(cptr + 6, vc6, 3);
    cptr += ldc;
    vst1q_f32(cptr, vcA);
    vst1q_lane_f32(cptr + 4, vcE, 0);
    vst1q_lane_f32(cptr + 5, vcF, 0);
    vst1q_lane_f32(cptr + 6, vcG, 0);
    cptr += ldc;
    vst1q_f32(cptr, vcB);
    vst1q_lane_f32(cptr + 4, vcE, 1);
    vst1q_lane_f32(cptr + 5, vcF, 1);
    vst1q_lane_f32(cptr + 6, vcG, 1);
    cptr += ldc;
    vst1q_f32(cptr, vcC);
    vst1q_lane_f32(cptr + 4, vcE, 2);
    vst1q_lane_f32(cptr + 5, vcF, 2);
    vst1q_lane_f32(cptr + 6, vcG, 2);
    cptr += ldc;
    vst1q_f32(cptr, vcD);
    vst1q_lane_f32(cptr + 4, vcE, 3);
    vst1q_lane_f32(cptr + 5, vcF, 3);
    vst1q_lane_f32(cptr + 6, vcG, 3);
}

void block_sgemm_external_pack(int M, int N, int L, float *a, float *b, float *c)
{
    int eM = M + (4 - M % 4) % 4;
    switch (N % 8)
    {
        case 1:
            sgemm_tiny_scale = sgemm_4x1;
            break;
        case 2:
            sgemm_tiny_scale = sgemm_4x2;
            break;
        case 3:
            sgemm_tiny_scale = sgemm_4x3;
            break;
        case 4:
            sgemm_tiny_scale = sgemm_4x4;
            break;
        case 5:
            sgemm_tiny_scale = sgemm_4x5;
            break;
        case 6:
            sgemm_tiny_scale = sgemm_4x6;
            break;
        case 7:
            sgemm_tiny_scale = sgemm_4x7;
            break;
    }
    block_sgemm_pack(eM, N, L, a, L, b, N, c, N);
}

void block_sgemm(int M, int N, int L, float *a, float *b, float *c)
{
    switch (N % 8)
    {
        case 1:
            sgemm_tiny_scale = sgemm_4x1;
            break;
        case 2:
            sgemm_tiny_scale = sgemm_4x2;
            break;
        case 3:
            sgemm_tiny_scale = sgemm_4x3;
            break;
        case 4:
            sgemm_tiny_scale = sgemm_4x4;
            break;
        case 5:
            sgemm_tiny_scale = sgemm_4x5;
            break;
        case 6:
            sgemm_tiny_scale = sgemm_4x6;
            break;
        case 7:
            sgemm_tiny_scale = sgemm_4x7;
            break;
    }
    switch (M % 4)
    {
        case 0:
            internalPackA = internalPackA4;
            break;
        case 1:
            internalPackA = internalPackA1;
            break;
        case 2:
            internalPackA = internalPackA2;
            break;
        case 3:
            internalPackA = internalPackA3;
            break;
    }
    block_sgemm_internal_pack(M, N, L, a, L, b, N, c, N);
}

void sgemm_4x8_pack(int L, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;
    float32x4_t vb1, vb2;
    float32x4_t va0, va1, va2, va3;

    float32x4_t vc0 = vld1q_f32(cptr);
    float32x4_t vc4 = vld1q_f32(cptr + 4);
    cptr += ldc;
    float32x4_t vc1 = vld1q_f32(cptr);
    float32x4_t vc5 = vld1q_f32(cptr + 4);
    cptr += ldc;
    float32x4_t vc2 = vld1q_f32(cptr);
    float32x4_t vc6 = vld1q_f32(cptr + 4);
    cptr += ldc;
    float32x4_t vc3 = vld1q_f32(cptr);
    float32x4_t vc7 = vld1q_f32(cptr + 4);

    for (int p = 0; p < L; ++p)
    {
        vb1  = vld1q_f32(bptr);
        vb2  = vld1q_f32(bptr + 4);

        va0 = vld1q_dup_f32(aptr);
        va1 = vld1q_dup_f32(aptr + 1);
        va2 = vld1q_dup_f32(aptr + 2);
        va3 = vld1q_dup_f32(aptr + 3);

#if __aarch64__
        vc0 = vfmaq_f32(vc0, va0, vb1);
        vc1 = vfmaq_f32(vc1, va1, vb1);
        vc2 = vfmaq_f32(vc2, va2, vb1);
        vc3 = vfmaq_f32(vc3, va3, vb1);

        vc4 = vfmaq_f32(vc4, va0, vb2);
        vc5 = vfmaq_f32(vc5, va1, vb2);
        vc6 = vfmaq_f32(vc6, va2, vb2);
        vc7 = vfmaq_f32(vc7, va3, vb2);
#else
        vc0 = vmlaq_f32(vc0, va0, vb1);
        vc1 = vmlaq_f32(vc1, va1, vb1);
        vc2 = vmlaq_f32(vc2, va2, vb1);
        vc3 = vmlaq_f32(vc3, va3, vb1);

        vc4 = vmlaq_f32(vc4, va0, vb2);
        vc5 = vmlaq_f32(vc5, va1, vb2);
        vc6 = vmlaq_f32(vc6, va2, vb2);
        vc7 = vmlaq_f32(vc7, va3, vb2);
#endif // __aarch64__

        bptr += 8;
        aptr += 4;
    }

    cptr = c;
    vst1q_f32(cptr, vc0);
    vst1q_f32(cptr + 4, vc4);
    cptr += ldc;
    vst1q_f32(cptr, vc1);
    vst1q_f32(cptr + 4, vc5);
    cptr += ldc;
    vst1q_f32(cptr, vc2);
    vst1q_f32(cptr + 4, vc6);
    cptr += ldc;
    vst1q_f32(cptr, vc3);
    vst1q_f32(cptr + 4, vc7);
}

void SGEBP_externalPackA_tiny_scale(int M, int N, int L, float *a, int lda, float *b, int ldb, float *c, int ldc, float* packA, float* packB)
{
    //Align L to achieve better performance for better cache line alignment.
    int eL = L + (4 - L % 4) % 4;
    int remN = N % 8;
    int fN = N - remN;

    for (int i = 0; i < M; i += 4)
    {
        for (int j = 0; j < fN; j += 8)
        {
            if (i == 0)
                internalPackB8(L, packB + j * eL, b + j, ldb);
            sgemm_4x8_pack(L, a + i * L, lda, packB + j * eL, 8, c + i * ldc + j, ldc);
        }
        if (remN)
            sgemm_tiny_scale(L, a + i * L, lda, b + fN, ldb, c + i * ldc + fN, ldc);
    }
}

inline void sgemm_8x8_pack(int L, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;
    float32x4_t vb0, vb1;
    float32x4_t va0, va1;

    float32x4_t vc0 = vld1q_f32(cptr);
    float32x4_t vc8 = vld1q_f32(cptr + 4);
    cptr += ldc;
    float32x4_t vc1 = vld1q_f32(cptr);
    float32x4_t vc9 = vld1q_f32(cptr + 4);
    cptr += ldc;
    float32x4_t vc2 = vld1q_f32(cptr);
    float32x4_t vcA = vld1q_f32(cptr + 4);
    cptr += ldc;
    float32x4_t vc3 = vld1q_f32(cptr);
    float32x4_t vcB = vld1q_f32(cptr + 4);
    cptr += ldc;
    float32x4_t vc4 = vld1q_f32(cptr);
    float32x4_t vcC = vld1q_f32(cptr + 4);
    cptr += ldc;
    float32x4_t vc5 = vld1q_f32(cptr);
    float32x4_t vcD = vld1q_f32(cptr + 4);
    cptr += ldc;
    float32x4_t vc6 = vld1q_f32(cptr);
    float32x4_t vcE = vld1q_f32(cptr + 4);
    cptr += ldc;
    float32x4_t vc7 = vld1q_f32(cptr);
    float32x4_t vcF = vld1q_f32(cptr + 4);

    for (int p = 0; p < L; ++p)
    {
        vb0  = vld1q_f32(bptr);
        vb1  = vld1q_f32(bptr + 4);
        va0 = vld1q_f32(aptr);
        va1 = vld1q_f32(aptr + 4);

#if __aarch64__
        vc0 = vfmaq_laneq_f32(vc0, vb0, va0, 0);
        vc1 = vfmaq_laneq_f32(vc1, vb0, va0, 1);
        vc2 = vfmaq_laneq_f32(vc2, vb0, va0, 2);
        vc3 = vfmaq_laneq_f32(vc3, vb0, va0, 3);

        vc4 = vfmaq_laneq_f32(vc4, vb0, va1, 0);
        vc5 = vfmaq_laneq_f32(vc5, vb0, va1, 1);
        vc6 = vfmaq_laneq_f32(vc6, vb0, va1, 2);
        vc7 = vfmaq_laneq_f32(vc7, vb0, va1, 3);

        vc8 = vfmaq_laneq_f32(vc8, vb1, va0, 0);
        vc9 = vfmaq_laneq_f32(vc9, vb1, va0, 1);
        vcA = vfmaq_laneq_f32(vcA, vb1, va0, 2);
        vcB = vfmaq_laneq_f32(vcB, vb1, va0, 3);

        vcC = vfmaq_laneq_f32(vcC, vb1, va1, 0);
        vcD = vfmaq_laneq_f32(vcD, vb1, va1, 1);
        vcE = vfmaq_laneq_f32(vcE, vb1, va1, 2);
        vcF = vfmaq_laneq_f32(vcF, vb1, va1, 3);
#else
        vc0 = vmlaq_f32(vc0, vb0, vld1q_dup_f32(aptr + 0));
        vc1 = vmlaq_f32(vc1, vb0, vld1q_dup_f32(aptr + 1));
        vc2 = vmlaq_f32(vc2, vb0, vld1q_dup_f32(aptr + 2));
        vc3 = vmlaq_f32(vc3, vb0, vld1q_dup_f32(aptr + 3));

        vc4 = vmlaq_f32(vc4, vb0, vld1q_dup_f32(aptr + 4));
        vc5 = vmlaq_f32(vc5, vb0, vld1q_dup_f32(aptr + 5));
        vc6 = vmlaq_f32(vc6, vb0, vld1q_dup_f32(aptr + 6));
        vc7 = vmlaq_f32(vc7, vb0, vld1q_dup_f32(aptr + 7));

        vc8 = vmlaq_f32(vc8, vb1, vld1q_dup_f32(aptr + 0));
        vc9 = vmlaq_f32(vc9, vb1, vld1q_dup_f32(aptr + 1));
        vcA = vmlaq_f32(vcA, vb1, vld1q_dup_f32(aptr + 2));
        vcB = vmlaq_f32(vcB, vb1, vld1q_dup_f32(aptr + 3));

        vcC = vmlaq_f32(vcC, vb1, vld1q_dup_f32(aptr + 4));
        vcD = vmlaq_f32(vcD, vb1, vld1q_dup_f32(aptr + 5));
        vcE = vmlaq_f32(vcE, vb1, vld1q_dup_f32(aptr + 6));
        vcF = vmlaq_f32(vcF, vb1, vld1q_dup_f32(aptr + 7));
#endif // __aarch64__

        bptr += 8;
        aptr += 8;
    }
    cptr = c;
    vst1q_f32(cptr, vc0);
    vst1q_f32(cptr + 4, vc8);
    cptr += ldc;
    vst1q_f32(cptr, vc1);
    vst1q_f32(cptr + 4, vc9);
    cptr += ldc;
    vst1q_f32(cptr, vc2);
    vst1q_f32(cptr + 4, vcA);
    cptr += ldc;
    vst1q_f32(cptr, vc3);
    vst1q_f32(cptr + 4, vcB);
    cptr += ldc;
    vst1q_f32(cptr, vc4);
    vst1q_f32(cptr + 4, vcC);
    cptr += ldc;
    vst1q_f32(cptr, vc5);
    vst1q_f32(cptr + 4, vcD);
    cptr += ldc;
    vst1q_f32(cptr, vc6);
    vst1q_f32(cptr + 4, vcE);
    cptr += ldc;
    vst1q_f32(cptr, vc7);
    vst1q_f32(cptr + 4, vcF);
}

void SGEBP_externalPackA_tiny_scale_8x8(int M, int N, int L, float *a, int lda, float *b, int ldb, float *c, int ldc, float* packA, float* packB)
{
    //Align L to achieve better performance for better cache line alignment.
    int eL = L + (4 - L % 4) % 4;
    int remN = N % 8;
    int fN = N - remN;

    for (int i = 0; i < M; i += 8)
    {
        for (int j = 0; j < fN; j += 8)
        {
            if (i == 0)
                internalPackB8(L, packB + j * eL, b + j, ldb);
            sgemm_8x8_pack(L, a + i * L, lda, packB + j * eL, 8, c + i * ldc + j, ldc);
        }
        if (remN)
            sgemm_tiny_scale(L, a + i * L, lda, b + fN, ldb, c + i * ldc + fN, ldc);
    }
}

void SGEBP_internalPack_tiny_scale(int M, int N, int L, float *a, int lda, float *b, int ldb, float *c, int ldc, float* packA, float* packB)
{
    //Align L for better cache line alignment.
    int eL = L + (4 - L % 4) % 4;
    int remM = M % 4;
    int remN = N % 8;
    int fM = M - remM;
    int fN = N - remN;

    for (int i = 0; i < fM; i += 8)
    {
        internalPackA4(L, packA + i * eL, a + i * lda, lda);
        for (int j = 0; j < fN; j += 8)
        {
            if (i == 0)
                internalPackB8(L, packB + j * eL, b + j, ldb);
            sgemm_8x8_pack(L, packA + i * eL, lda, packB + j * eL, 8, c + i * ldc + j, ldc);
        }
        if (remN)
            sgemm_tiny_scale(L, packA + i * eL, lda, b + fN, ldb, c + i * ldc + fN, ldc);
    }
    //Compute last row in A
    if (remM)
    {
        internalPackA(L, packA + fM * eL, a + fM * lda, lda);
        for (int j = 0; j < fN; j += 8)
        {
            sgemm_8x8_pack(L, packA + fM * eL, lda, packB + j * eL, 8, c + fM * ldc + j, ldc);
        }
        if (remN)
            sgemm_tiny_scale(L, packA + fM * eL, lda, b + fN, ldb, c + fM * ldc + fN, ldc);
    }
}


void block_sgemm_pack(int M, int N, int L, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    for (int i = 0; i < M; ++i)
        memset(c + ldc * i, 0, sizeof(float) * N);
    float* packB = (float *)_mm_malloc(sizeof(float) * kc *  N, 16);
    if (NULL == packB) return;

    for (int l = 0; l < N; l += nc)
    {
        int lb = min(N - l, nc);
        float* packAptr = a;
        for (int i = 0; i < M; i += mc)
        {
            int ib = min(M - i, mc);
            for (int p = 0; p < L; p += kc)
            {
                int pb = min(L - p, kc);
                SGEBP_externalPackA_tiny_scale(ib, lb, pb, packAptr, lda, b + p * ldb + l, ldb, c + i * ldc + l, ldc, NULL, packB);
                packAptr += ib * pb;
            }
        }
    }
    _mm_free(packB);
}

void block_sgemm_pack_8x8(int M, int N, int L, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    for (int i = 0; i < M; ++i)
        memset(c + ldc * i, 0, sizeof(float) * N);
    float* packB = (float *)_mm_malloc(sizeof(float) * kc *  N, 16);
    if (NULL == packB) return;

    for (int l = 0; l < N; l += nc)
    {
        float* packAptr = a;
        for (int i = 0; i < M; i += mc)
        {
            for (int p = 0; p < L; p += kc)
            {
                int lb = min(N - l, nc);
                int ib = min(M - i, mc);
                int pb = min(L - p, kc);
                SGEBP_externalPackA_tiny_scale_8x8(ib, lb, pb, packAptr, lda, b + p * ldb + l, ldb, c + i * ldc + l, ldc, NULL, packB);
                packAptr += ib * pb;
            }
        }
    }
    _mm_free(packB);
}

void block_sgemm_internal_pack(int M, int N, int L, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    int eM = M + (4 - M % 4) % 4;
    for (int i = 0; i < M; ++i)
        memset(c + ldc * i, 0, sizeof(float) * N);
    float* packA = (float *)_mm_malloc(sizeof(float) * kc * eM, 16);
    if (NULL == packA) return;
    float* packB = (float *)_mm_malloc(sizeof(float) * kc *  N, 16);
    if (NULL == packB)
    {
        _mm_free(packA);
        return;
    }

    for (int l = 0; l < N; l += nc)
    {
        int lb = min(N - l, nc);
        for (int i = 0; i < M; i += mc)
        {
            int ib = min(M - i, mc);
            for (int p = 0; p < L; p += kc)
            {
                int pb = min(L - p, kc);
                SGEBP_internalPack_tiny_scale(ib, lb, pb, a + i * lda + p, lda, b + p * ldb + l, ldb, c + i * ldc + l, ldc, packA, packB);
            }
        }
    }
    _mm_free(packA);
    _mm_free(packB);
}


void externalPackA(int M, int L, float* packA, float* a, int lda)
{
    float* packAptr = packA;
    int remM = M % 4;
    int eM = M + (4 - M % 4) % 4;//Ceil

    void (*remPack)(int, float*, float*, int) = NULL;
    switch (remM)
    {
        case 0:
            remPack = internalPackA4;
            break;
        case 1:
            remPack = internalPackA1;
            break;
        case 2:
            remPack = internalPackA2;
            break;
        case 3:
            remPack = internalPackA3;
            break;
    }
    for (int i = 0; i < eM; i += mc)
    {
        const int ib = min(eM - i, mc);
        for (int p = 0; p < L; p += kc)
        {
            const int pb = min(L - p, kc);
            for (int k = 0; k < ib - 4; k += 4)
            {
                internalPackA4(pb, packAptr, a + i * lda + p + k * lda, lda);
                packAptr += 4 * pb;
            }
            remPack(pb, packAptr, a + i * lda + p + (ib - 4) * lda, lda);
            packAptr += 4 * pb;
        }
    }
}

void block_sgemm_external_pack_threading(int M, int N, int L, float *a, float *b, float *c, int num_threads)
{
    int eM = M + (4 - M % 4) % 4;
    switch (N % 8)
    {
        case 1:
            sgemm_tiny_scale = sgemm_4x1;
            break;
        case 2:
            sgemm_tiny_scale = sgemm_4x2;
            break;
        case 3:
            sgemm_tiny_scale = sgemm_4x3;
            break;
        case 4:
            sgemm_tiny_scale = sgemm_4x4;
            break;
        case 5:
            sgemm_tiny_scale = sgemm_4x5;
            break;
        case 6:
            sgemm_tiny_scale = sgemm_4x6;
            break;
        case 7:
            sgemm_tiny_scale = sgemm_4x7;
            break;
    }
    const int factor = 1;
    int tN = N / num_threads / factor;
    tN = tN + (8 - tN % 8) % 8;
    if (num_threads == 1 || N <= 8 || N - (num_threads * factor - 1) * tN <= 0)
    {
        block_sgemm_pack(eM, N, L, a, L, b, N, c, N);
    }
    else
    {
#pragma parallel for num_threads(num_threads)
        for (int i = 0; i < num_threads * factor; ++i)
        {
            int sN = (tN < N - i * tN) ? tN : N - i * tN;
            block_sgemm_pack(eM, sN, L, a, L, b + i * tN, N, c + i * tN, N);
        }
    }
}

void externalPackA8(int M, int L, float* packA, float* a, int lda)
{
    float* packAptr = packA;
    int eM = M + (8 - M % 8) % 8;

    for (int i = 0; i < eM; i += mc)
    {
        const int ib = min(eM - i, mc);
        for (int p = 0; p < L; p += kc)
        {
            const int pb = min(L - p, kc);
            for (int k = 0; k < ib; k += 8)
            {
                internalPackA8(pb, packAptr, a + i * lda + p + k * lda, lda);
                packAptr += 8 * pb;
            }
        }
    }
}

void block_sgemm_external_pack_threading_8x8(int M, int N, int L, float *a, float *b, float *c, int num_threads)
{
    int eM = M + (8 - M % 8) % 8;
    switch (N % 8)
    {
        case 1:
            sgemm_tiny_scale = sgemm_8x1;
            break;
        case 2:
            sgemm_tiny_scale = sgemm_8x2;
            break;
        case 3:
            sgemm_tiny_scale = sgemm_8x3;
            break;
        case 4:
            sgemm_tiny_scale = sgemm_8x4;
            break;
        case 5:
            sgemm_tiny_scale = sgemm_8x5;
            break;
        case 6:
            sgemm_tiny_scale = sgemm_8x6;
            break;
        case 7:
            sgemm_tiny_scale = sgemm_8x7;
            break;
    }

    if (num_threads > 8)   num_threads = 8;

    const int factor = 1;
    unsigned int tN = N / num_threads / factor;

    tN = (tN + 7) & 0xFFFFFFF8;
    unsigned int lastSN = N - (num_threads * factor - 1) * tN;
    while (lastSN <= 0)
    {
        --num_threads;
        lastSN = N - (num_threads * factor - 1) * tN;
    }
    num_threads = (num_threads <= 0) ? 1 : num_threads;
    //num_threads change to 1 -> SN and tN will not be used.
    //otherwise, num_threads is left unchanged and lastSN is still effective.

    if (num_threads == 1 || N <= 8 || N - (num_threads * factor - 1) * tN <= 0)
    {
        block_sgemm_pack_8x8(eM, N, L, a, L, b, N, c, N);
    }
    else
    {
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int sN = tN;
            if (tid == num_threads - 1)
                sN = N - tid * tN;
            block_sgemm_pack_8x8(eM, sN, L, a, L, b + tid * tN, N, c + tid * tN, N);
        }
    }
}
