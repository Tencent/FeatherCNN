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

#include "winograd_kernels.h"
#include "helper.h"
#include <stdlib.h>
#include <arm_neon.h>
#include <assert.h>
#include <string.h>


//#define FEATHER_USE_GCD

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef FEATHER_USE_GCD
#include <dispatch/dispatch.h>
#endif

//#define WINOGRAD_BENCH

static inline void TensorGEMMInnerKernel4x4x4(float* &WTp, const int &wstride, const float* &UTp, const float* &vp, const int &inChannels);
static inline void TensorGEMMInnerKernel4x3x4(float* &WTp, const int &wstride, const float* &UTp, const float* &vp, const int &inChannels);
static inline void TensorGEMMInnerKernel4x2x4(float* &WTp, const int &wstride, const float* &UTp, const float* &vp, const int &inChannels);
static inline void TensorGEMMInnerKernel4x1x4(float* &WTp, const int &wstride, const float* &UTp, const float* &vp, const int &inChannels);

void naive_gemm_temp(int M, int N, int L, float *A, float *B, float *C)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i * N + j] = 0.f;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < L; k++)
            {
                C[i * N + j] += A[i * L + k] * B[k * N + j];
            }
        }
    }
}

void transpose_temp(size_t m, size_t n, float *in, float *out) //  A[m][n] -> A[n][m]
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            out[j * m + i] = in[i * n + j];
}

static inline void neon_transpose4x4_inplace_f32_cpp(
    float32x4_t &row0,
    float32x4_t &row1,
    float32x4_t &row2,
    float32x4_t &row3)
{
    /*
     * row0 = ( x00 x01 x02 x03 )
     * row1 = ( x10 x11 x12 x13 )
     * row2 = ( x20 x21 x22 x23 )
     * row3 = ( x30 x31 x32 x33 )
     */
    /*
     * row01 = ( x00 x10 x02 x12 ), ( x01 x11 x03, x13 )
     * row23 = ( x20 x30 x22 x32 ), ( x21 x31 x23, x33 )
     */
    float32x4x2_t row01 = vtrnq_f32(row0, row1);
    float32x4x2_t row23 = vtrnq_f32(row2, row3);

    /*
     * row0 = ( x00 x10 x20 x30 )
     * row1 = ( x01 x11 x21 x31 )
     * row2 = ( x02 x12 x22 x32 )
     * row3 = ( x03 x13 x23 x33 )
     */
    row0 = vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0]));
    row1 = vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1]));
    row2 = vcombine_f32(vget_high_f32(row01.val[0]), vget_high_f32(row23.val[0]));
    row3 = vcombine_f32(vget_high_f32(row01.val[1]), vget_high_f32(row23.val[1]));
}

static inline void neon_transpose4x4_inplace_f32_fp(float *fp)
{
    /*
     * row0 = ( x00 x01 x02 x03 )
     * row1 = ( x10 x11 x12 x13 )
     * row2 = ( x20 x21 x22 x23 )
     * row3 = ( x30 x31 x32 x33 )
     */
    /*
     * row01 = ( x00 x10 x02 x12 ), ( x01 x11 x03, x13 )
     * row23 = ( x20 x30 x22 x32 ), ( x21 x31 x23, x33 )
     */
    float32x4_t row0 = vld1q_f32(fp);
    float32x4_t row1 = vld1q_f32(fp + 4);
    float32x4_t row2 = vld1q_f32(fp + 8);
    float32x4_t row3 = vld1q_f32(fp + 12);

    float32x4x2_t row01 = vtrnq_f32(row0, row1);
    float32x4x2_t row23 = vtrnq_f32(row2, row3);

    /*
     * row0 = ( x00 x10 x20 x30 )
     * row1 = ( x01 x11 x21 x31 )
     * row2 = ( x02 x12 x22 x32 )
     * row3 = ( x03 x13 x23 x33 )
     */
    row0 = vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0]));
    row1 = vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1]));
    row2 = vcombine_f32(vget_high_f32(row01.val[0]), vget_high_f32(row23.val[0]));
    row3 = vcombine_f32(vget_high_f32(row01.val[1]), vget_high_f32(row23.val[1]));

    vst1q_f32(fp, row0);
    vst1q_f32(fp + 4, row1);
    vst1q_f32(fp + 8, row2);
    vst1q_f32(fp + 12, row3);
}
/*
 AT =
 ⎡1  1  1   0⎤
 ⎢           ⎥
 ⎣0  1  -1  1⎦
 G =
 ⎡ 1    0     0 ⎤
 ⎢              ⎥
 ⎢1/2  1/2   1/2⎥
 ⎢              ⎥
 ⎢1/2  -1/2  1/2⎥
 ⎢              ⎥
 ⎣ 0    0     1 ⎦
 BT =
 ⎡1  0   -1  0⎤
 ⎢            ⎥
 ⎢0  1   1   0⎥
 ⎢            ⎥
 ⎢0  -1  1   0⎥
 ⎢            ⎥
 ⎣0  -1  0   1⎦
 */

inline void winogradOutputTransformInplace(float32x2_t *o0, float32x2_t *o1, float32x4_t *w0, float32x4_t *w1, float32x4_t *w2, float32x4_t *w3)
{
    float32x4_t s0, s1;
    float32x2_t d0, d1, d2, d3;
    float32x4_t m0 = *w0;
    float32x4_t m1 = *w1;
    float32x4_t m2 = *w2;
    float32x4_t m3 = *w3;

    s0 = m0 + m1 + m2;
    s1 = m1 - m2 + m3;

    float32x4x2_t rows = vtrnq_f32(s0, s1);

    d0 = vget_low_f32(rows.val[0]);
    d1 = vget_low_f32(rows.val[1]);
    d2 = vget_high_f32(rows.val[0]);
    d3 = vget_high_f32(rows.val[1]);

    *o0 = d0 + d1 + d2;
    *o1 = d1 - d2 + d3;
}

/*
 * Kernel Transform:
 * Layout k(outChannel, inChannel):
 * k(0, 0) k(0, 1) k(0, 2) k(0, 3) k(0, 4)....
 * k(1, 0) k(1, 1) k(1, 2) k(1, 3) k(1, 4)....
 * k(2, 0) k(2, 1) k(2, 2) k(2, 3) k(2, 4)....
 */

void winogradKernelTransform_F6x6_3x3(float *transKernel, float *kernel)
{
    float ktm[24] =
    {
        1.0f, 0.0f, 0.0f,
        -2.0f / 9, -2.0f / 9, -2.0f / 9,
        -2.0f / 9, 2.0f / 9, -2.0f / 9,
        1.0f / 90, 1.0f / 45, 2.0f / 45,
        1.0f / 90, -1.0f / 45, 2.0f / 45,
        1.0f / 45, 1.0f / 90, 1.0f / 180,
        1.0f / 45, -1.0f / 90, 1.0f / 180,
        0.0f, 0.0f, 1.0f
    };

    float midBlock[24];
    float outBlock[24];
    float bigBlock[64];
    float32x4_t w0, w1, w2, w3;

    //print_floats(kernel, 3, 3);
    naive_gemm_temp(8, 3, 3, ktm, kernel, midBlock);
    transpose_temp(8, 3, midBlock, outBlock);
    naive_gemm_temp(8, 8, 3, ktm, outBlock, bigBlock);

    for (int i = 0; i < 16; ++i)
    {
        float32x4_t reg;
        reg = vld1q_f32(bigBlock + i * 4);
        vst1q_f32(transKernel + i * 16, reg);
    }
    //print_floats(bigBlock, 8, 8);

}

void winogradKernelTransformPacked(float *transKernel, float *kernel, int stride, float* base, int oi, int oj)
{
    float ktm[24] =
    {
        1.0f, 0.0f, 0.0f,
        -2.0f / 9, -2.0f / 9, -2.0f / 9,
        -2.0f / 9, 2.0f / 9, -2.0f / 9,
        1.0f / 90, 1.0f / 45, 2.0f / 45,
        1.0f / 90, -1.0f / 45, 2.0f / 45,
        1.0f / 45, 1.0f / 90, 1.0f / 180,
        1.0f / 45, -1.0f / 90, 1.0f / 180,
        0.0f, 0.0f, 1.0f
    };

    float midBlock[24];
    float outBlock[24];
    float bigBlock[64];
    float32x4_t w0, w1, w2, w3;

    //print_floats(kernel, 3, 3);
    naive_gemm_temp(8, 3, 3, ktm, kernel, midBlock);
    transpose_temp(8, 3, midBlock, outBlock);
    naive_gemm_temp(8, 8, 3, ktm, outBlock, bigBlock);
    //print_floats(bigBlock, 8, 8);

    for (int i = 0; i < 16; ++i)
    {
        float32x4_t reg;
        reg = vld1q_f32(bigBlock + i * 4);
        vst1q_f32(transKernel + i * stride, reg);
        //printf("offset %d\n", i * stride);
        //printf("UTp offset %d i %d j %d\n", transKernel+i*stride - base, oi, oj);
    }
}

void transformKernel_F6x6_3x3(float *UT, float *kernel, int inChannels, int outChannels)
{
    //printf("UT %x kernel %x inChannels %zu outChannels %zu\n", UT, kernel, inChannels, outChannels);
    for (int i = 0; i < inChannels; ++i)
    {
        for (int j = 0; j < outChannels; ++j)
        {
            int cid = j * inChannels + i;
            //float* UTp = UT + 64 * (i & 0xFFFFFFFC) + (i & 0x3) * 4 + j / 4 * 64 * inChannels;
            float* UTp = UT + (j / 4) * (256 * inChannels) //Big block id for every 4 output channels.
                         + 16 * i    //Choose the line, which is the input channel offset.
                         + (j & 0x3) * 4;//Starting point in each block.
            winogradKernelTransformPacked(UTp, kernel + 9 * (j * inChannels + i), 16 * inChannels, UT, i, j);
        }
    }
}

/*
 * Input transform:
 *
 * First traverse all the output channels and then the input channels,
 * so that each input frame would be only transformed once.
 * The out buffer for UT x VT result would be heavily flushed.
 * We can do cache blocking against the output channels if necessary.
 */

/*
 * Reshape:
 * Dimensions: (nRowBlocks * nColBlocks * 16 / 16) x (Input channels * 16)
 * Illustrations:
 * INPUT CHANNEL IDS:   |   0   |   1   |   2   |   3   |...
 * 4x4 Data blocks:     |BLOCK 0|BLOCK 0|BLOCK 0|BLOCK 0|
 *                      |BLOCK 1|BLOCK 1|BLOCK 1|BLOCK 1|
 *                      |BLOCK 2|BLOCK 2|BLOCK 2|BLOCK 2|
 */

inline void input_transform(
    float32x4_t &r0,
    float32x4_t &r1,
    float32x4_t &r2,
    float32x4_t &r3,
    float32x4_t &r4,
    float32x4_t &r5,
    float32x4_t &r6,
    float32x4_t &r7,
    float32x4_t &t1,
    float32x4_t &t2,
    float32x4_t &s1,
    float32x4_t &s2,
    float32x4_t &p1,
    float32x4_t &p2,
    const float32x4_t &f5_25,
    const float32x4_t &f4_25,
    const float32x4_t &f4,
    const float32x4_t &f2_5,
    const float32x4_t &f2,
    const float32x4_t &f1_25,
    const float32x4_t &f0_5,
    const float32x4_t &f0_25
)
{
    r0 = r0 - r6 + (r4 - r2) * f5_25;
    r7 = r7 - r1 + (r3 - r5) * f5_25;

    //r6 - r4 * f5_25 can be reused
    //r1 - r3 * f5_25 can be reused

    t1 = r2 + r6 - r4 * f4_25;
    t2 = r1 + r5 - r3 * f4_25;

    s1 = r4 * f1_25;
    s2 = r3 * f2_5;

    p1 = r6 + (r2 * f0_25 - s1);
    p2 = r1 * f0_5 - s2 + r5 * f2;

    r3 = p1 + p2;
    r4 = p1 - p2;

    //2.5 * (r01 - r03 + r05)

    p1 = r6 + (r2 - s1) * f4;
    p2 = r1 * f2 - s2 + r5 * f0_5;

    r5 = p1 + p2;
    r6 = p1 - p2;

    r1 = vaddq_f32(t1, t2);
    r2 = vsubq_f32(t1, t2);
}

void winogradInputFrameTransformSeq(float *VT, int inChannels, float *input, int inputh, int inputw, int frameStride, int ldin, int nRowBlocks, int nColBlocks, int num_threads)
{
    //Constants in transformation matrices.
    const float32x4_t f5 = vdupq_n_f32(5.0f);
    const float32x4_t f4 = vdupq_n_f32(4.0f);
    const float32x4_t f2 = vdupq_n_f32(2.0f);
    const float32x4_t f2_5 = vdupq_n_f32(2.5f);
    const float32x4_t f5_25 = vdupq_n_f32(5.25f);
    const float32x4_t f4_25 = vdupq_n_f32(4.25f);
    const float32x4_t f1_25 = vdupq_n_f32(1.25f);
    const float32x4_t f0_5 = vdupq_n_f32(0.5f);
    const float32x4_t f0_25 = vdupq_n_f32(0.25f);
    const float32x4_t vZero = vdupq_n_f32(0.0f);

    const int nBlocks = nRowBlocks * nColBlocks;
    const int nBlocksAligned = nBlocks & 0xFFFFFFFC;
    const int rem = nBlocks & 0x3;
    memset(VT, 0, sizeof(float) * 64 * nBlocks * inChannels);
    int hdiff = nColBlocks * 6 + 2 - inputh;
    int wdiff = nRowBlocks * 6 + 2 - inputw;
    //diff ranges from 0 to 5
    //printf("diff %d, %d\n", hdiff, wdiff);
    //print_floats(input, inChannels* inputh , inputw);

#ifdef FEATHER_USE_GCD
    dispatch_apply(inChannels, dispatch_get_global_queue(0, 0), ^(size_t ic)
#else
#ifdef _OPENMP
    #pragma omp parallel for num_threads(num_threads) collapse(2) schedule(static)
#endif
    for (int ic = 0; ic < inChannels; ++ic)
#endif
    {
        for (int j = 0; j < nColBlocks; ++j)
        {
            float ext[64];
            float32x4_t d0, d1, d2, d3, d4, d5, d6, d7;
            float32x4_t l0, l1, l2, l3, l4, l5, l6, l7;
            float32x4_t r0, r1, r2, r3, r4, r5, r6, r7;
            float32x4_t m1, m2, s1, s2, t1, t2;//Auxiliary registers
            float *p0 = input + ic * frameStride + ldin * j * 6;
            float *p1 = p0 + ldin;
            float *p2 = p1 + ldin;
            float *p3 = p2 + ldin;
            float *p4 = p3 + ldin;
            float *p5 = p4 + ldin;
            float *p6 = p5 + ldin;
            float *p7 = p6 + ldin;

            for (int i = 0; i < nRowBlocks; ++i)
            {
                int bid = j * nRowBlocks + i;
                float *outp = VT + (ic * nBlocks + (bid & 0xFFFFFFFC)) * 64 + (bid & 0x3) * 4;
                if (((j * 6 + 8) > inputh) || ((i * 6 + 8) > inputw))
                {
                    for (int t = 0; t < 16; ++t)
                    {
                        vst1q_f32(ext + t * 4, vZero);
                    }
                    int step_h = inputh - j * 6;
                    int step_w = inputw - i * 6;
                    if (step_h > 8)
                        step_h = 8;
                    if (step_w > 8)
                        step_w = 8;
                    float* edge_blk = input + ic * frameStride + (j * 6) * ldin + (i * 6);
                    //printf("small blk offset %d\n", edge_blk - input);
                    for (int n = 0; n < step_h; ++n)
                        for (int m = 0; m < step_w; ++m)
                            ext[n * 8 + m] = *(edge_blk + n * ldin + m);

                    //printf("step hxw %dx%d\n", step_h, step_w);
                    //print_floats(ext, 8, 8);
                    l0 = vld1q_f32(ext);
                    r0 = vld1q_f32(ext + 4);
                    l1 = vld1q_f32(ext + 8);
                    r1 = vld1q_f32(ext + 12);
                    l2 = vld1q_f32(ext + 16);
                    r2 = vld1q_f32(ext + 20);
                    l3 = vld1q_f32(ext + 24);
                    r3 = vld1q_f32(ext + 28);
                    l4 = vld1q_f32(ext + 32);
                    r4 = vld1q_f32(ext + 36);
                    l5 = vld1q_f32(ext + 40);
                    r5 = vld1q_f32(ext + 44);
                    l6 = vld1q_f32(ext + 48);
                    r6 = vld1q_f32(ext + 52);
                    l7 = vld1q_f32(ext + 56);
                    r7 = vld1q_f32(ext + 60);
                }
                else
                {
                    l0 = vld1q_f32(p0);
                    r0 = vld1q_f32(p0 + 4);
                    p0 += 6;
                    l1 = vld1q_f32(p1);
                    r1 = vld1q_f32(p1 + 4);
                    p1 += 6;
                    l2 = vld1q_f32(p2);
                    r2 = vld1q_f32(p2 + 4);
                    p2 += 6;
                    l3 = vld1q_f32(p3);
                    r3 = vld1q_f32(p3 + 4);
                    p3 += 6;
                    l4 = vld1q_f32(p4);
                    r4 = vld1q_f32(p4 + 4);
                    p4 += 6;
                    l5 = vld1q_f32(p5);
                    r5 = vld1q_f32(p5 + 4);
                    p5 += 6;
                    l6 = vld1q_f32(p6);
                    r6 = vld1q_f32(p6 + 4);
                    p6 += 6;
                    l7 = vld1q_f32(p7);
                    r7 = vld1q_f32(p7 + 4);
                    p7 += 6;
                }

                input_transform(l0, l1, l2, l3, l4, l5, l6, l7, //Target
                                t1, t2, s1, s2, m1, m2, //Auxiliary
                                f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                neon_transpose4x4_inplace_f32_cpp(l0, l1, l2, l3);
                neon_transpose4x4_inplace_f32_cpp(l4, l5, l6, l7);
                input_transform(r0, r1, r2, r3, r4, r5, r6, r7, //Target
                                t1, t2, s1, s2, m1, m2, //Auxiliary
                                f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                neon_transpose4x4_inplace_f32_cpp(r0, r1, r2, r3);
                neon_transpose4x4_inplace_f32_cpp(r4, r5, r6, r7);
                input_transform(l0, l1, l2, l3, r0, r1, r2, r3, //Target
                                t1, t2, s1, s2, m1, m2, //Auxiliary
                                f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                input_transform(l4, l5, l6, l7, r4, r5, r6, r7, //Target
                                t1, t2, s1, s2, m1, m2, //Auxiliary
                                f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants

                //printf("outp offset %d\n", outp - VT);
                if (bid < nBlocksAligned)
                {
                    vst1q_f32(outp, l0);
                    vst1q_f32(outp + 16, l4);
                    vst1q_f32(outp + 32, l1);
                    vst1q_f32(outp + 48, l5);

                    vst1q_f32(outp + 64, l2);
                    vst1q_f32(outp + 80, l6);
                    vst1q_f32(outp + 96, l3);
                    vst1q_f32(outp + 112, l7);

                    vst1q_f32(outp + 128, r0);
                    vst1q_f32(outp + 144, r4);
                    vst1q_f32(outp + 160, r1);
                    vst1q_f32(outp + 176, r5);

                    vst1q_f32(outp + 192, r2);
                    vst1q_f32(outp + 208, r6);
                    vst1q_f32(outp + 224, r3);
                    vst1q_f32(outp + 240, r7);
                }
                else
                {
                    vst1q_f32(outp, l0);
                    vst1q_f32(outp + rem * 4, l4);
                    vst1q_f32(outp + rem * 8, l1);
                    vst1q_f32(outp + rem * 12, l5);

                    vst1q_f32(outp + rem * 16, l2);
                    vst1q_f32(outp + rem * 20, l6);
                    vst1q_f32(outp + rem * 24, l3);
                    vst1q_f32(outp + rem * 28, l7);

                    vst1q_f32(outp + rem * 32, r0);
                    vst1q_f32(outp + rem * 36, r4);
                    vst1q_f32(outp + rem * 40, r1);
                    vst1q_f32(outp + rem * 44, r5);

                    vst1q_f32(outp + rem * 48, r2);
                    vst1q_f32(outp + rem * 52, r6);
                    vst1q_f32(outp + rem * 56, r3);
                    vst1q_f32(outp + rem * 60, r7);

                }
            }
        }
#ifdef FEATHER_USE_GCD
    });
#else
    }
#endif
}

void TensorGEMM(float *WT, const float *VT, const float *UT, const int depth, const int inChannels, const int outChannels, const int nRowBlocks, const int nColBlocks, const int num_threads, float* pack_arr, const int cache_block)
{
    //Real depth in floating point is $depth * 4.
    const int nBlocks = nRowBlocks * nColBlocks;
    const int nBlocksAligned = nBlocks - nBlocks % 4;
    const int wstride = nBlocks * 4 * depth;
    const int vstride = nBlocks * 4 * depth;


    //assert(nBlocks % 4 == 0);
    assert(nBlocks >= 1);
    assert(outChannels % 4 == 0);
    //int pass = nBlocksAligned / cache_block;
    //int r = nBlocksAligned % cache_block;
    int pass = nBlocks / cache_block;
    int r = nBlocks % cache_block;
    if (r > 0)
        ++pass;
    //printf("nBlocks %d, pass block %d pass %d r %d\n", nBlocks, cache_block, pass, r);
    //TODO: Increase r value when it's too small.
    //We will be caching for 4 * $cache_block * $depth floats.
    for (int p = 0; p < pass; p++)
    {
        //int tid = omp_get_thread_num();
        //int tid = 0;

        int start_block_id = p * cache_block;
        int end_block_id = start_block_id + cache_block;

        end_block_id = (end_block_id > nBlocks) ? nBlocks : end_block_id;
        int end_block_id_aligned = end_block_id & 0xFFFFFFFC;
        const int rem = end_block_id % 4;


        /*I have no idea which packing method is faster, seeems that they are not the major bottleneck after loop swapping*/
#ifdef _OPENMP
        #pragma omp parallel num_threads(num_threads)
#endif
        {
#ifdef _OPENMP
            #pragma omp for collapse(2)
#endif
            for (int i = start_block_id; i < end_block_id_aligned + 4; i += 4)
            {
                for (int d = 0; d < depth; ++d)
                {
                    float *pack_workp = pack_arr + (i - start_block_id) * depth * inChannels * 4 + d * inChannels * 4 * ((i < end_block_id_aligned) ? 4 : rem);
                    float32x4_t v0, v1, v2, v3;
                    for (int ic = 0; ic < inChannels; ++ic)
                    {
                        if (i < end_block_id_aligned)
                        {
                            const float *svp = VT + i * 4 * depth + d * 4 * 4 + ic * vstride;
                            //print_floats(svp, 16);
                            v0 = vld1q_f32(svp);
                            v1 = vld1q_f32(svp + 4);
                            v2 = vld1q_f32(svp + 8);
                            v3 = vld1q_f32(svp + 12);
                            svp += vstride;
                            vst1q_f32(pack_workp, v0);
                            vst1q_f32(pack_workp +  4, v1);
                            vst1q_f32(pack_workp +  8, v2);
                            vst1q_f32(pack_workp + 12, v3);
                            pack_workp += 16;
                        }
                        else
                        {
                            //print_floats(svp, 4 * len);
                            const float *svp = VT + i * 4 * depth + d * 4 * rem + ic * vstride;
                            v0 = vld1q_f32(svp);
                            if (rem > 1)
                                v1 = vld1q_f32(svp + 4);
                            if (rem > 2)
                                v2 = vld1q_f32(svp + 8);
                            svp += vstride;

                            vst1q_f32(pack_workp, v0);
                            if (rem > 1)
                                vst1q_f32(pack_workp +  4, v1);
                            if (rem > 2)
                                vst1q_f32(pack_workp +  8, v2);
                            pack_workp += rem * 4;
                        }
                    }
                }
            }
#ifdef FEATHER_USE_GCD
            dispatch_apply(outChannels / 4, dispatch_get_global_queue(0, 0), ^(size_t dispatch_i)
            {
                int oc = dispatch_i * 4;

#else
#ifdef _OPENMP
            #pragma omp for collapse(3)
#endif
            for (int oc = 0; oc < outChannels; oc += 4)
            {
#endif
                for (int i = start_block_id; i < end_block_id_aligned + 4; i += 4)
                {
                    for (int d = 0; d < depth; ++d)
                    {
                        if (i < end_block_id_aligned)
                        {
                            const float *UTp = UT + d * 16 * inChannels + oc / 4 * inChannels * 16 * depth;
                            const float *vp = pack_arr
                                              + (i - start_block_id) * inChannels * depth * 4//which block
                                              + d * depth * inChannels;
                            float *WTp = WT + oc * wstride + i * depth * 4 + d * 16 + (i % 4) * 4;
                            TensorGEMMInnerKernel4x4x4(WTp, wstride, UTp, vp, inChannels);
                        }
                        else
                        {
                            int i = end_block_id & 0xFFFFFFC;
                            int len = end_block_id & 0x3;
                            //printf("end_block_id %d i %d len %d wstride %d\n", end_block_id, i, len, wstride);
                            //We are going to compute the remains here.
                            //for (int oc = 0; oc < outChannels; oc += 4)
                            //{
                            const float *UTp = UT + d * 16 * inChannels + oc / 4 * inChannels * 16 * depth;
                            const float *vp = pack_arr
                                              //+ tid * cache_block * inChannels * depth * 4//which thread
                                              + (i - start_block_id) * inChannels * depth * 4//which block
                                              + d * depth * inChannels * (4 * len) / 16;
                            float *WTp = WT + oc * wstride + i * depth * 4 + d * 4 * len + (i % 4) * 4;
                            if (len == 1)
                            {
                                TensorGEMMInnerKernel4x1x4(WTp, wstride, UTp, vp, inChannels);
                            }
                            if (len == 2)
                            {
                                TensorGEMMInnerKernel4x2x4(WTp, wstride, UTp, vp, inChannels);
                            }
                            if (len == 3)
                            {
                                TensorGEMMInnerKernel4x3x4(WTp, wstride, UTp, vp, inChannels);
                            }
                        }
                    }
                }
#ifdef FEATHER_USE_GCD
            });
#else
            }
#endif

        }
    }
}

static inline void TensorGEMMInnerKernel4x4x4(float* &WTp, const int &wstride, const float* &UTp, const float* &vp, const int &inChannels)
{
    float32x4_t vc00, vc01, vc02, vc03;
    float32x4_t vc10, vc11, vc12, vc13;
    float32x4_t vc20, vc21, vc22, vc23;
    float32x4_t vc30, vc31, vc32, vc33;
    float32x4_t u0, u1, u2, u3;
    float32x4_t v0, v1, v2, v3;
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc03 = vdupq_n_f32(0.f);
    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc13 = vdupq_n_f32(0.f);
    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    vc23 = vdupq_n_f32(0.f);
    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    vc32 = vdupq_n_f32(0.f);
    vc33 = vdupq_n_f32(0.f);
    const float *up = UTp;
    // printf("WTp offset %d\n", WTp - WT);
    //if(oc == 0)
    //      printf("vp offset %d i %d block offset %d depth offset %d\n", vp - (float*)pack_arr, i, (i - start_block_id) * inChannels * depth,d * depth * inChannels);
    // printf("up offset %d\n", up - UT);
    for (int ic = 0; ic < inChannels; ++ic)
    {
        //if(oc == 0){
        //print_floats(vp, 16);
        //print_floats(up, 16);
        //}
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        vp += 16;
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        up += 16;
#ifdef __aarch64__
        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);
        vc13 = vfmaq_f32(vc13, u1, v3);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);
        vc23 = vfmaq_f32(vc23, u2, v3);

        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);
        vc32 = vfmaq_f32(vc32, u3, v2);
        vc33 = vfmaq_f32(vc33, u3, v3);
#else
        vc00 = vmlaq_f32(vc00, u0, v0);
        vc01 = vmlaq_f32(vc01, u0, v1);
        vc02 = vmlaq_f32(vc02, u0, v2);
        vc03 = vmlaq_f32(vc03, u0, v3);

        vc10 = vmlaq_f32(vc10, u1, v0);
        vc11 = vmlaq_f32(vc11, u1, v1);
        vc12 = vmlaq_f32(vc12, u1, v2);
        vc13 = vmlaq_f32(vc13, u1, v3);

        vc20 = vmlaq_f32(vc20, u2, v0);
        vc21 = vmlaq_f32(vc21, u2, v1);
        vc22 = vmlaq_f32(vc22, u2, v2);
        vc23 = vmlaq_f32(vc23, u2, v3);

        vc30 = vmlaq_f32(vc30, u3, v0);
        vc31 = vmlaq_f32(vc31, u3, v1);
        vc32 = vmlaq_f32(vc32, u3, v2);
        vc33 = vmlaq_f32(vc33, u3, v3);
#endif
    }
    float *wp = WTp;
    vst1q_f32(wp, vc00);
    vst1q_f32(wp + 4, vc01);
    vst1q_f32(wp + 8, vc02);
    vst1q_f32(wp + 12, vc03);
    //print_floats(wp, 16);
    wp += wstride;
    vst1q_f32(wp, vc10);
    vst1q_f32(wp + 4, vc11);
    vst1q_f32(wp + 8, vc12);
    vst1q_f32(wp + 12, vc13);
    wp += wstride;
    vst1q_f32(wp, vc20);
    vst1q_f32(wp + 4, vc21);
    vst1q_f32(wp + 8, vc22);
    vst1q_f32(wp + 12, vc23);
    wp += wstride;
    vst1q_f32(wp, vc30);
    vst1q_f32(wp + 4, vc31);
    vst1q_f32(wp + 8, vc32);
    vst1q_f32(wp + 12, vc33);
}

static inline void TensorGEMMInnerKernel4x3x4(float* &WTp, const int &wstride, const float* &UTp, const float* &vp, const int &inChannels)
{
    float32x4_t vc00, vc01, vc02;
    float32x4_t vc10, vc11, vc12;
    float32x4_t vc20, vc21, vc22;
    float32x4_t vc30, vc31, vc32;
    float32x4_t u0, u1, u2, u3;
    float32x4_t v0, v1, v2;
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    vc32 = vdupq_n_f32(0.f);
    const float *up = UTp;
    // printf("WTp offset %d\n", WTp - WT);
    //if(oc == 0)
    //      printf("vp offset %d i %d block offset %d depth offset %d\n", vp - (float*)pack_arr, i, (i - start_block_id) * inChannels * depth,d * depth * inChannels);
    // printf("up offset %d\n", up - UT);
    for (int ic = 0; ic < inChannels; ++ic)
    {
        //if(oc == 0){
        //print_floats(vp, 16);
        //print_floats(up, 16);
        //}
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        vp += 12;
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        up += 16;
#ifdef __aarch64__
        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);

        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);
        vc32 = vfmaq_f32(vc32, u3, v2);
#else
        vc00 = vmlaq_f32(vc00, u0, v0);
        vc01 = vmlaq_f32(vc01, u0, v1);
        vc02 = vmlaq_f32(vc02, u0, v2);

        vc10 = vmlaq_f32(vc10, u1, v0);
        vc11 = vmlaq_f32(vc11, u1, v1);
        vc12 = vmlaq_f32(vc12, u1, v2);

        vc20 = vmlaq_f32(vc20, u2, v0);
        vc21 = vmlaq_f32(vc21, u2, v1);
        vc22 = vmlaq_f32(vc22, u2, v2);

        vc30 = vmlaq_f32(vc30, u3, v0);
        vc31 = vmlaq_f32(vc31, u3, v1);
        vc32 = vmlaq_f32(vc32, u3, v2);
#endif
    }
    float *wp = WTp;
    vst1q_f32(wp, vc00);
    vst1q_f32(wp + 4, vc01);
    vst1q_f32(wp + 8, vc02);
    wp += wstride;
    vst1q_f32(wp, vc10);
    vst1q_f32(wp + 4, vc11);
    vst1q_f32(wp + 8, vc12);
    wp += wstride;
    vst1q_f32(wp, vc20);
    vst1q_f32(wp + 4, vc21);
    vst1q_f32(wp + 8, vc22);
    wp += wstride;
    vst1q_f32(wp, vc30);
    vst1q_f32(wp + 4, vc31);
    vst1q_f32(wp + 8, vc32);
}

static inline void TensorGEMMInnerKernel4x2x4(float* &WTp, const int &wstride, const float* &UTp, const float* &vp, const int &inChannels)
{
    float32x4_t vc00, vc01;
    float32x4_t vc10, vc11;
    float32x4_t vc20, vc21;
    float32x4_t vc30, vc31;
    float32x4_t u0, u1, u2, u3;
    float32x4_t v0, v1;
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    const float *up = UTp;
    // printf("WTp offset %d\n", WTp - WT);
    //if(oc == 0)
    //      printf("vp offset %d i %d block offset %d depth offset %d\n", vp - (float*)pack_arr, i, (i - start_block_id) * inChannels * depth,d * depth * inChannels);
    // printf("up offset %d\n", up - UT);
    for (int ic = 0; ic < inChannels; ++ic)
    {
        //if(oc == 0){
        //print_floats(vp, 16);
        //print_floats(up, 16);
        //}
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        vp += 8;
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        up += 16;
#ifdef __aarch64__
        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);
#else
        vc00 = vmlaq_f32(vc00, u0, v0);
        vc01 = vmlaq_f32(vc01, u0, v1);
        vc10 = vmlaq_f32(vc10, u1, v0);
        vc11 = vmlaq_f32(vc11, u1, v1);
        vc20 = vmlaq_f32(vc20, u2, v0);
        vc21 = vmlaq_f32(vc21, u2, v1);
        vc30 = vmlaq_f32(vc30, u3, v0);
        vc31 = vmlaq_f32(vc31, u3, v1);
#endif
    }
    float *wp = WTp;
    vst1q_f32(wp, vc00);
    vst1q_f32(wp + 4, vc01);
    wp += wstride;
    vst1q_f32(wp, vc10);
    vst1q_f32(wp + 4, vc11);
    wp += wstride;
    vst1q_f32(wp, vc20);
    vst1q_f32(wp + 4, vc21);
    wp += wstride;
    vst1q_f32(wp, vc30);
    vst1q_f32(wp + 4, vc31);
}

static inline void TensorGEMMInnerKernel4x1x4(float* &WTp, const int &wstride, const float* &UTp, const float* &vp, const int &inChannels)
{
    float32x4_t vc00;
    float32x4_t vc10;
    float32x4_t vc20;
    float32x4_t vc30;
    float32x4_t u0, u1, u2, u3;
    float32x4_t v0;
    vc00 = vdupq_n_f32(0.f);
    vc10 = vdupq_n_f32(0.f);
    vc20 = vdupq_n_f32(0.f);
    vc30 = vdupq_n_f32(0.f);
    const float *up = UTp;
    // printf("WTp offset %d\n", WTp - WT);
    //if(oc == 0)
    //      printf("vp offset %d i %d block offset %d depth offset %d\n", vp - (float*)pack_arr, i, (i - start_block_id) * inChannels * depth,d * depth * inChannels);
    // printf("up offset %d\n", up - UT);
    for (int ic = 0; ic < inChannels; ++ic)
    {
        //if(oc == 0){
        //print_floats(vp, 4);
        //print_floats(up, 16);
        //}
        //printf("vp %d\n", vp);
        v0 = vld1q_f32(vp);
        vp += 4;
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        up += 16;
#ifdef __aarch64__
        vc00 = vfmaq_f32(vc00, u0, v0);
        vc10 = vfmaq_f32(vc10, u1, v0);
        vc20 = vfmaq_f32(vc20, u2, v0);
        vc30 = vfmaq_f32(vc30, u3, v0);
#else
        vc00 = vmlaq_f32(vc00, u0, v0);
        vc10 = vmlaq_f32(vc10, u1, v0);
        vc20 = vmlaq_f32(vc20, u2, v0);
        vc30 = vmlaq_f32(vc30, u3, v0);
#endif
    }
    float *wp = WTp;
    vst1q_f32(wp, vc00);
    wp += wstride;
    vst1q_f32(wp, vc10);
    wp += wstride;
    vst1q_f32(wp, vc20);
    wp += wstride;
    vst1q_f32(wp, vc30);
}


static inline void winograd_f6k3_output_transform_inplace(
    float32x4_t &m0,
    float32x4_t &m1,
    float32x4_t &m2,
    float32x4_t &m3,
    float32x4_t &m4,
    float32x4_t &m5,
    float32x4_t &m6,
    float32x4_t &m7)
{
    /*
     * s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
     * s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
     * s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
     * s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
     * s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
     * s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m7
     */

    const float32x4_t m1_add_m2 = m1 + m2;
    const float32x4_t m1_sub_m2 = m1 - m2;
    const float32x4_t m3_add_m4 = m3 + m4;
    const float32x4_t m3_sub_m4 = m3 - m4;
    const float32x4_t m5_add_m6 = m5 + m6;
    const float32x4_t m5_sub_m6 = m5 - m6;

    // Finised with M[0-6] as **inputs** here.
    m0 = m0 + m1_add_m2;
    m5 = m7 + m1_sub_m2;
    // Finised with M[0-7] as **inputs** here.

#ifdef __aarch64__
    const float32x4_t const_16 = vdupq_n_f32(16.0f);
    m1 = vfmaq_f32(m1_sub_m2, const_16, m5_sub_m6);
    m4 = vfmaq_f32(m1_add_m2, const_16, m3_add_m4);

    const float32x4_t const_8 = vdupq_n_f32(8.0f);
    m2 = vfmaq_f32(m1_add_m2, const_8, m5_add_m6);
    m3 = vfmaq_f32(m1_sub_m2, const_8, m3_sub_m4);

    const float32x4_t const_32 = vdupq_n_f32(32.0f);
    m0 = vfmaq_f32(m0, const_32, m5_add_m6);
    m0 += m3_add_m4;

    m5 = vfmaq_f32(m5, const_32, m3_sub_m4);
    m5 += m5_sub_m6;

    const float32x4_t const_2 = vdupq_n_f32(2.0f);
    m1 = vfmaq_f32(m1, m3_sub_m4, const_2);
    m4 = vfmaq_f32(m4, m5_add_m6, const_2);

    const float32x4_t const_4 = vdupq_n_f32(4.0f);
    m2 = vfmaq_f32(m2, m3_add_m4, const_4);
    m3 = vfmaq_f32(m3, m5_sub_m6, const_4);
#else
    const float32x4_t const_16 = vdupq_n_f32(16.0f);
    m1 = vmlaq_f32(m1_sub_m2, const_16, m5_sub_m6);
    m4 = vmlaq_f32(m1_add_m2, const_16, m3_add_m4);

    const float32x4_t const_8 = vdupq_n_f32(8.0f);
    m2 = vmlaq_f32(m1_add_m2, const_8, m5_add_m6);
    m3 = vmlaq_f32(m1_sub_m2, const_8, m3_sub_m4);

    const float32x4_t const_32 = vdupq_n_f32(32.0f);
    m0 = vmlaq_f32(m0, const_32, m5_add_m6);
    m0 += m3_add_m4;

    m5 = vmlaq_f32(m5, const_32, m3_sub_m4);
    m5 += m5_sub_m6;

    const float32x4_t const_2 = vdupq_n_f32(2.0f);
    m1 = vmlaq_f32(m1, m3_sub_m4, const_2);
    m4 = vmlaq_f32(m4, m5_add_m6, const_2);

    const float32x4_t const_4 = vdupq_n_f32(4.0f);
    m2 = vmlaq_f32(m2, m3_add_m4, const_4);
    m3 = vmlaq_f32(m3, m5_sub_m6, const_4);
#endif

    const float32x4_t const_0 = vdupq_n_f32(0.0f);
    m6 = const_0;
    m7 = const_0;
}


template<bool HAS_RELU, bool HAS_BIAS>
void winogradOutputTransform(float *output, int outputh, int outputw, int ldout, float *WT, int outChannels, int nRowBlocks, int nColBlocks, float* biasArr, int num_threads)
{
    const float32x4_t vZero = vdupq_n_f32(0.f);
    int nBlocks = nRowBlocks * nColBlocks;
    int nBlocksAligned = nBlocks & 0xFFFFFFFC;
    int rem = nBlocks & 0x3;
#ifdef FEATHER_USE_GCD
    dispatch_apply(outChannels, dispatch_get_global_queue(0, 0), ^(size_t oc)
#else
    #pragma omp parallel for num_threads(num_threads) schedule(static) collapse(3)
    for (int oc = 0; oc < outChannels; ++oc)
#endif
    {
        for (int j = 0; j < nColBlocks; ++j)
        {
            for (int i = 0; i < nRowBlocks; ++i)
            {
                float32x4_t vBias = vdupq_n_f32(biasArr[oc]);
                float ext[48];
                const int offset = nRowBlocks * nColBlocks * 64 * oc;
                float *wp;
                float32x4_t s0, s1, s2, s3;
                float32x2_t o0, o1;
                float32x2_t d0, d1, d2, d3;
                int bid = nRowBlocks * j + i;
                wp = WT + oc * nBlocks * 64 + (bid & 0xFFFFFFFC) * 64 + (bid & 0x3) * 4;
                float32x4_t l0, l1, l2, l3, l4, l5, l6, l7;
                float32x4_t r0, r1, r2, r3, r4, r5, r6, r7;
                if (bid < nBlocksAligned)
                {
                    l0 = vld1q_f32(wp);
                    r0 = vld1q_f32(wp + 16);
                    l1 = vld1q_f32(wp + 32);
                    r1 = vld1q_f32(wp + 48);
                    l2 = vld1q_f32(wp + 64);
                    r2 = vld1q_f32(wp + 80);
                    l3 = vld1q_f32(wp + 96);
                    r3 = vld1q_f32(wp + 112);
                    l4 = vld1q_f32(wp + 128);
                    r4 = vld1q_f32(wp + 144);
                    l5 = vld1q_f32(wp + 160);
                    r5 = vld1q_f32(wp + 176);
                    l6 = vld1q_f32(wp + 192);
                    r6 = vld1q_f32(wp + 208);
                    l7 = vld1q_f32(wp + 224);
                    r7 = vld1q_f32(wp + 240);
                }
                else
                {
                    //print_floats(wp, 64);
                    l0 = vld1q_f32(wp);
                    r0 = vld1q_f32(wp + rem * 4);
                    l1 = vld1q_f32(wp + rem * 8);
                    r1 = vld1q_f32(wp + rem * 12);
                    l2 = vld1q_f32(wp + rem * 16);
                    r2 = vld1q_f32(wp + rem * 20);
                    l3 = vld1q_f32(wp + rem * 24);
                    r3 = vld1q_f32(wp + rem * 28);
                    l4 = vld1q_f32(wp + rem * 32);
                    r4 = vld1q_f32(wp + rem * 36);
                    l5 = vld1q_f32(wp + rem * 40);
                    r5 = vld1q_f32(wp + rem * 44);
                    l6 = vld1q_f32(wp + rem * 48);
                    r6 = vld1q_f32(wp + rem * 52);
                    l7 = vld1q_f32(wp + rem * 56);
                    r7 = vld1q_f32(wp + rem * 60);
                }

                winograd_f6k3_output_transform_inplace(l0, l1, l2, l3, l4, l5, l6, l7);
                winograd_f6k3_output_transform_inplace(r0, r1, r2, r3, r4, r5, r6, r7);
                neon_transpose4x4_inplace_f32_cpp(l0, l1, l2, l3);
                neon_transpose4x4_inplace_f32_cpp(l4, l5, l6, l7);
                neon_transpose4x4_inplace_f32_cpp(r0, r1, r2, r3);
                neon_transpose4x4_inplace_f32_cpp(r4, r5, r6, r7);
                winograd_f6k3_output_transform_inplace(l0, l1, l2, l3, r0, r1, r2, r3);
                winograd_f6k3_output_transform_inplace(l4, l5, l6, l7, r4, r5, r6, r7);
                float *outFrame = output + oc * outputw * outputh + j * outputw * 6 + i * 6;
                //printf("block %d outFrame offset %d\n", bid, outFrame - output);

                if (HAS_BIAS)
                {
                    l0 = vaddq_f32(l0, vBias);
                    l1 = vaddq_f32(l1, vBias);
                    l2 = vaddq_f32(l2, vBias);
                    l3 = vaddq_f32(l3, vBias);
                    l4 = vaddq_f32(l4, vBias);
                    l5 = vaddq_f32(l5, vBias);
                    l6 = vaddq_f32(l6, vBias);
                    l7 = vaddq_f32(l7, vBias);
                    r0 = vaddq_f32(r0, vBias);
                    r1 = vaddq_f32(r1, vBias);
                    r4 = vaddq_f32(r4, vBias);
                    r5 = vaddq_f32(r5, vBias);
                }

                if (HAS_RELU)
                {
                    l0 = vmaxq_f32(l0, vZero);
                    l1 = vmaxq_f32(l1, vZero);
                    l2 = vmaxq_f32(l2, vZero);
                    l3 = vmaxq_f32(l3, vZero);
                    l4 = vmaxq_f32(l4, vZero);
                    l5 = vmaxq_f32(l5, vZero);
                    l6 = vmaxq_f32(l6, vZero);
                    l7 = vmaxq_f32(l7, vZero);
                    r0 = vmaxq_f32(r0, vZero);
                    r1 = vmaxq_f32(r1, vZero);
                    r4 = vmaxq_f32(r4, vZero);
                    r5 = vmaxq_f32(r5, vZero);
                }

                if (((j * 6 + 6) > outputh) || ((i * 6 + 6) > outputw))
                {
                    for (int t = 0; t < 12; ++t)
                    {
                        vst1q_f32(ext + t * 4, vZero);
                    }
                    int step_h = outputh - j * 6;
                    int step_w = outputw - i * 6;
                    if (step_h > 6)
                        step_h = 6;
                    if (step_w > 6)
                        step_w = 6;
                    //printf("step %dx%d\n", step_h, step_w);
                    vst1q_f32(ext, l0);
                    vst1q_f32(ext + 4, l4);
                    vst1q_f32(ext + 8, l1);
                    vst1q_f32(ext + 12, l5);
                    vst1q_f32(ext + 16, l2);
                    vst1q_f32(ext + 20, l6);
                    vst1q_f32(ext + 24, l3);
                    vst1q_f32(ext + 28, l7);
                    vst1q_f32(ext + 32, r0);
                    vst1q_f32(ext + 36, r4);
                    vst1q_f32(ext + 40, r1);
                    vst1q_f32(ext + 44, r5);
                    for (int n = 0; n < step_h; ++n)
                    {
                        for (int m = 0; m < step_w; ++m)
                        {
                            *(outFrame + (n * ldout + m)) = ext[n * 8 + m];
                        }
                    }

                }
                else
                {
                    vst1q_f32(outFrame, l0);
                    vst1_f32(outFrame + 4, vget_low_f32(l4));
                    outFrame += ldout;
                    vst1q_f32(outFrame, l1);
                    vst1_f32(outFrame + 4, vget_low_f32(l5));
                    outFrame += ldout;
                    vst1q_f32(outFrame, l2);
                    vst1_f32(outFrame + 4, vget_low_f32(l6));
                    outFrame += ldout;
                    vst1q_f32(outFrame, l3);
                    vst1_f32(outFrame + 4, vget_low_f32(l7));
                    outFrame += ldout;
                    vst1q_f32(outFrame, r0);
                    vst1_f32(outFrame + 4, vget_low_f32(r4));
                    outFrame += ldout;
                    vst1q_f32(outFrame, r1);
                    vst1_f32(outFrame + 4, vget_low_f32(r5));
                }
            }
        }
#ifdef FEATHER_USE_GCD
    });
#else
    }
#endif
}

size_t getPackArraySize_F6x6_3x3(int inChannels, int num_threads)
{
    return 32 * num_threads * inChannels *  64;/**depth in floats*/
}

void winogradNonFusedTransform_inner(float *output, int ldout, float *WT, float *VT, float *UT, int inChannels, int outChannels, float *input, int inputh, int inputw, int frameStride, int ldin, int nRowBlocks, int nColBlocks, WinogradOutType outType, float *biasArr, float* pack_array, int num_threads)
{
    int nBlocks = nRowBlocks * nColBlocks;
    int depth = 16;
    //float *pack_arr = (float*)malloc(cache_block * inChannels * depth * sizeof(float32x4_t));
#ifdef WINOGRAD_BENCH
    Timer tmr;
    tmr.startBench();
#endif
    winogradInputFrameTransformSeq(VT, inChannels, input, inputh, inputw, frameStride, ldin, nRowBlocks, nColBlocks, num_threads);
#ifdef WINOGRAD_BENCH
    tmr.endBench("Input Transform:");
    tmr.startBench();
#endif
    //printf("=====VT=====\n");
    //print_floats(VT, inChannels * 4 * nBlocks, 16);
    //print_floats(VT, inChannels , 4 * nBlocks* 16);
    //printf("=====UT=====\n");
    //print_floats(UT, inChannels * outChannels, 64);
    TensorGEMM(WT, VT, UT,
               16, inChannels, outChannels, nRowBlocks, nColBlocks, num_threads, pack_array, num_threads * 32);
    //printf("=============TensorGEMMOut==============\n");
    //print_floats(WT, outChannels , nBlocks * 4* 16);
#ifdef WINOGRAD_BENCH
    tmr.endBench("Multiplication:");
#endif
#ifdef WINOGRAD_BENCH
    tmr.startBench();
#endif
    //out type
#if 0
    winogradOutputTransform<false, false>(output, inputh - 2, inputw - 2, ldout, WT, outChannels, nRowBlocks, nColBlocks, biasArr, num_threads);
#else
    switch (outType)
    {
        case None:
            winogradOutputTransform<false, false>(output, inputh - 2, inputw - 2, ldout, WT, outChannels, nRowBlocks, nColBlocks, biasArr, num_threads);
            break;
        case ReLU:
            winogradOutputTransform<true, false>(output, inputh - 2, inputw - 2, ldout, WT, outChannels, nRowBlocks, nColBlocks, biasArr, num_threads);
            break;
        case Bias:
            winogradOutputTransform<false, true>(output, inputh - 2, inputw - 2, ldout, WT, outChannels, nRowBlocks, nColBlocks, biasArr, num_threads);
            break;
        case BiasReLU:
            winogradOutputTransform<true, true>(output, inputh - 2, inputw - 2, ldout, WT, outChannels, nRowBlocks, nColBlocks, biasArr, num_threads);
            break;
    }
#endif

#ifdef WINOGRAD_BENCH
    tmr.endBench("Output transform:");
#endif
    //free(pack_arr);
}

void winogradNonFusedTransform_F6x6_3x3(float *output, int outChannels, float *WT, float *VT, float *UT, float *input, int inChannels, int inputh, int inputw, WinogradOutType outType, float *biasArr, float* pack_array, int num_threads)
{
    //assert((inputw - 2) % 6 == 0);
    //assert((inputh - 2) % 6 == 0);
    const int inputFrameStride = inputw * inputh;
    const int nRowBlocks = (inputw + 3) / 6;//inputw - kernelw + 5
    const int nColBlocks = (inputh + 3) / 6;
    const int ldout = inputw - 2;
    //printf("-----------------\n");
    //printf("Block dim = %dx%d\n", nRowBlocks, nColBlocks);
    //printf("ldout = %d\n", ldout);
    winogradNonFusedTransform_inner(output, ldout, WT, VT, UT, inChannels, outChannels, input, inputh, inputw, inputFrameStride, inputw, nRowBlocks, nColBlocks, outType, biasArr, pack_array, num_threads);
}
