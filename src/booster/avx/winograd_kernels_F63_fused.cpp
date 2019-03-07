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

// #include <booster/winograd_kernels.h>
// #include <booster/helper.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include <booster/booster.h>
#include <booster/winograd_kernels.h>
#include <booster/helper.h>

#include <immintrin.h>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef FEATHER_USE_GCD
#include <dispatch/dispatch.h>
#endif

//GCC doesn't have these two macros.
#ifdef __GNUC__
#define _mm256_set_m128(/* __m128 */ hi, /* __m128 */ lo) \
    _mm256_insertf128_ps(_mm256_castps128_ps256(lo), (hi), 0x1)
#define _mm256_loadu2_m128(/* float const* */ hiaddr, \
                           /* float const* */ loaddr) \
    _mm256_set_m128(_mm_loadu_ps(hiaddr), _mm_loadu_ps(loaddr))
#endif

// #define WINOGRAD_BENCH

namespace Winograd_F63_Fused
{

static inline void TensorGEMMInnerKernel4x4x4_avx(float *&WTp, const float *&UTp, const float *&vp, const int &inChannels);

template <int zero_lane_cnt>
inline __m128i get_mm_tail_mask()
{
    if (zero_lane_cnt == -3)
        return _mm_set_epi32(0xFFFFFFFF, 0x0, 0x0, 0x0);
    else if (zero_lane_cnt == -2)
        return _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0);
    else if (zero_lane_cnt == -1)
        return _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0);
    else if (zero_lane_cnt == 0)
        return _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    else if (zero_lane_cnt == 1)
        return _mm_set_epi32(0x0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    else if (zero_lane_cnt == 2)
        return _mm_set_epi32(0x0, 0x0, 0xFFFFFFFF, 0xFFFFFFFF);
    else if (zero_lane_cnt == 3)
        return _mm_set_epi32(0x0, 0x0, 0x0, 0xFFFFFFFF);
    else
    {
        printf("Zero tail lanes should be -3, -2, -1, 0, 1, 2, 3, but given %d\n", zero_lane_cnt);
        exit(-1);
    }
}

inline __m128i get_mm_tail_mask(int zero_lane_cnt)
{
    if (zero_lane_cnt == -3)
        return _mm_set_epi32(0xFFFFFFFF, 0x0, 0x0, 0x0);
    else if (zero_lane_cnt == -2)
        return _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0);
    else if (zero_lane_cnt == -1)
        return _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0);
    else if (zero_lane_cnt == 0)
        return _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    else if (zero_lane_cnt == 1)
        return _mm_set_epi32(0x0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    else if (zero_lane_cnt == 2)
        return _mm_set_epi32(0x0, 0x0, 0xFFFFFFFF, 0xFFFFFFFF);
    else if (zero_lane_cnt == 3)
        return _mm_set_epi32(0x0, 0x0, 0x0, 0xFFFFFFFF);
    else
    {
        printf("Zero tail lanes should be -3, -2, -1, 0, 1, 2, 3, 4 but given %d\n", zero_lane_cnt);
        exit(-1);
    }
}

inline void mm_store_leftovers(float* &dst, __m128& left, __m128& right, const int& store_cnt, const __m128i& mask)
{
    if (store_cnt > 4)
    {
        _mm_storeu_ps(dst, left);
        _mm_maskstore_ps(dst + 4, mask, right);
    }
    else
    {
        _mm_maskstore_ps(dst, mask, left);
    }
}

inline void mm_load_leftovers(const float* &src, __m128& left, __m128& right, const int& load_cnt, const __m128i& mask)
{
    if (load_cnt > 4)
    {
        left = _mm_loadu_ps(src);
        right = _mm_maskload_ps(src + 4, mask);
    }
    else
    {
        left  = _mm_maskload_ps(src, mask);
        right = _mm_set1_ps(0.f);
    }
}
inline void transpose8_avx_ps(
    __m256 &r0,
    __m256 &r1,
    __m256 &r2,
    __m256 &r3,
    __m256 &r4,
    __m256 &r5,
    __m256 &r6,
    __m256 &r7)
{
    __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
    __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
    __t0 = _mm256_unpacklo_ps(r0, r1);
    __t1 = _mm256_unpackhi_ps(r0, r1);
    __t2 = _mm256_unpacklo_ps(r2, r3);
    __t3 = _mm256_unpackhi_ps(r2, r3);
    __t4 = _mm256_unpacklo_ps(r4, r5);
    __t5 = _mm256_unpackhi_ps(r4, r5);
    __t6 = _mm256_unpacklo_ps(r6, r7);
    __t7 = _mm256_unpackhi_ps(r6, r7);
    __tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0));
    __tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2));
    __tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0));
    __tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2));
    __tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0));
    __tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2));
    __tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0));
    __tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2));
    r0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
    r1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
    r2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
    r3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
    r4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
    r5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
    r6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
    r7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}


static inline void winograd_f6k3_output_transform_inplace_avx(
    __m256 &m0,
    __m256 &m1,
    __m256 &m2,
    __m256 &m3,
    __m256 &m4,
    __m256 &m5,
    __m256 &m6,
    __m256 &m7)
{
    /*
     * s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
     * s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
     * s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
     * s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
     * s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
     * s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m7
     */
    const __m256 m1_add_m2 = _mm256_add_ps(m1, m2);// m1 + m2;
    const __m256 m1_sub_m2 = _mm256_sub_ps(m1, m2);// m1 - m2;
    const __m256 m3_add_m4 = _mm256_add_ps(m3, m4);// m3 + m4;
    const __m256 m3_sub_m4 = _mm256_sub_ps(m3, m4);// m3 - m4;
    const __m256 m5_add_m6 = _mm256_add_ps(m5, m6);// m5 + m6;
    const __m256 m5_sub_m6 = _mm256_sub_ps(m5, m6);// m5 - m6;

    // Finised with M[0-6] as **inputs** here.
    m0 = _mm256_add_ps(m0, m1_add_m2);// m0 + m1_add_m2;
    m5 = _mm256_add_ps(m7, m1_sub_m2);// m7 + m1_sub_m2;
    // Finised with M[0-7] as **inputs** here.

    const __m256 const_16 = _mm256_set1_ps(16.0f);
    m1 = _mm256_fmadd_ps(const_16, m5_sub_m6, m1_sub_m2);
    m4 = _mm256_fmadd_ps(const_16, m3_add_m4, m1_add_m2);

    const __m256 const_8 = _mm256_set1_ps(8.0f);
    m2 = _mm256_fmadd_ps(const_8, m5_add_m6, m1_add_m2);
    m3 = _mm256_fmadd_ps(const_8, m3_sub_m4, m1_sub_m2);

    const __m256 const_32 = _mm256_set1_ps(32.0f);
    m0 = _mm256_fmadd_ps(const_32, m5_add_m6, m0);
    m0 = _mm256_add_ps(m0, m3_add_m4);// m0 += m3_add_m4;

    m5 = _mm256_fmadd_ps(const_32, m3_sub_m4, m5);
    m5 = _mm256_add_ps(m5, m5_sub_m6);// m5 += m5_sub_m6;

    const __m256 const_2 = _mm256_set1_ps(2.0f);
    m1 = _mm256_fmadd_ps(m3_sub_m4, const_2, m1);
    m4 = _mm256_fmadd_ps(m5_add_m6, const_2, m4);

    const __m256 const_4 = _mm256_set1_ps(4.0f);
    m2 = _mm256_fmadd_ps(m3_add_m4, const_4, m2);
    m3 = _mm256_fmadd_ps(m5_sub_m6, const_4, m3);

    const __m256 const_0 = _mm256_set1_ps(0.0f);
    m6 = const_0;
    m7 = const_0;
}

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

    //print_floats(kernel, 3, 3);
    naive_gemm_temp(8, 3, 3, ktm, kernel, midBlock);
    transpose_temp(8, 3, midBlock, outBlock);
    naive_gemm_temp(8, 8, 3, ktm, outBlock, bigBlock);

    for (int i = 0; i < 16; ++i)
    {
        __m128 reg;
        reg = _mm_load_ps(bigBlock + i * 4);
        _mm_store_ps(transKernel + i * 16, reg);
    }
    //print_floats(bigBlock, 8, 8);
}

void winogradKernelTransformPackedNaive(float *transKernel, float *kernel, int stride)
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

    //print_floats(kernel, 3, 3);
    naive_gemm_temp(8, 3, 3, ktm, kernel, midBlock);
    transpose_temp(8, 3, midBlock, outBlock);
    naive_gemm_temp(8, 8, 3, ktm, outBlock, bigBlock);
    //print_floats(bigBlock, 8, 8);

    for (int i = 0; i < 16; ++i)
    {
        __m128 reg;
        reg = _mm_load_ps(bigBlock + i * 4);
        _mm_store_ps(transKernel + i * stride, reg);
        //printf("offset %d\n", i * stride);
        //printf("UTp offset %d i %d j %d\n", transKernel+i*stride - base, oi, oj);
    }
}

// inline void winogradKernelTransformF6x6_3x3()

void winogradKernelTransformPacked(float *transKernel, float *kernel, int stride)
{
#if 0
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


    //print_floats(kernel, 3, 3);
    naive_gemm_temp(8, 3, 3, ktm, kernel, midBlock);
    transpose_temp(8, 3, midBlock, outBlock);
    naive_gemm_temp(8, 8, 3, ktm, outBlock, bigBlock);
    //print_floats(bigBlock, 8, 8);
#else
    float bigBlock[64];
    __m256 k0, k1, k2; // different rows of kernel.
    k0 = _mm256_loadu_ps(kernel);
    k1 = _mm256_loadu_ps(kernel + 3);
    k2 = _mm256_loadu_ps(kernel + 6);
    // Kernel transformation.
#if 1
    __m256 t0, t1, t2, t3, t4, t5, t6, t7, c0;
    t0 = k0;
    t1 = k0 + k1 + k2;
    t2 = k0 - k1 + k2;
    c0 = _mm256_set1_ps(-2.0f / 9);
    t1 = _mm256_mul_ps(c0, t1);
    t2 = _mm256_mul_ps(c0, t2);
    t3 = k0 + 2 * k1 + 4 * k2;
    t4 = k0 - 2 * k1 + 4 * k2;
    c0 = _mm256_set1_ps(1.0f / 90);
    t3 = _mm256_mul_ps(c0, t3);
    t4 = _mm256_mul_ps(c0, t4);
    t5 = 4 * k0 + 2 * k1 + k2;
    t6 = 4 * k0 - 2 * k1 + k2;
    c0 = _mm256_set1_ps(1.0f / 180);
    t5 = _mm256_mul_ps(c0, t5);
    t6 = _mm256_mul_ps(c0, t6);
    t7 = k2;

    transpose8_avx_ps(t0, t1, t2, t3, t4, t5, t6, t7);

    k0 = t0;
    k1 = t1;
    k2 = t2;

    t0 = k0;
    t1 = k0 + k1 + k2;
    t2 = k0 - k1 + k2;
    c0 = _mm256_set1_ps(-2.0f / 9);
    t1 = _mm256_mul_ps(c0, t1);
    t2 = _mm256_mul_ps(c0, t2);
    t3 = k0 + 2 * k1 + 4 * k2;
    t4 = k0 - 2 * k1 + 4 * k2;
    c0 = _mm256_set1_ps(1.0f / 90);
    t3 = _mm256_mul_ps(c0, t3);
    t4 = _mm256_mul_ps(c0, t4);
    t5 = 4 * k0 + 2 * k1 + k2;
    t6 = 4 * k0 - 2 * k1 + k2;
    c0 = _mm256_set1_ps(1.0f / 180);
    t5 = _mm256_mul_ps(c0, t5);
    t6 = _mm256_mul_ps(c0, t6);
    t7 = k2;
    #else
    #endif    
    // transpose_temp(8, 3, midBlock, outBlock);
    // naive_gemm_temp(8, 8, 3, ktm, outBlock, bigBlock);
    _mm256_store_ps(bigBlock, t0);
    _mm256_store_ps(bigBlock + 8, t1);
    _mm256_store_ps(bigBlock + 16, t2);
    _mm256_store_ps(bigBlock + 24, t3);
    _mm256_store_ps(bigBlock + 32, t4);
    _mm256_store_ps(bigBlock + 40, t5);
    _mm256_store_ps(bigBlock + 48, t6);
    _mm256_store_ps(bigBlock + 56, t7);
    
    
#endif
// print_floats(bigBlock, 8, 8);  
    for (int i = 0; i < 16; ++i)
    {
        __m128 reg;
        reg = _mm_load_ps(bigBlock + i * 4);
        _mm_store_ps(transKernel + i * stride, reg);
        //printf("offset %d\n", i * stride);
        //printf("UTp offset %d i %d j %d\n", transKernel+i*stride - base, oi, oj);
    }
}

void winogradKernelTransformPack2Outch(float *transKernel, float *kernel, int kernel_stride, int stride)
{
    __m256 k0, k1, k2; // different rows of kernel.
    k0 = _mm256_loadu_ps(kernel);
    k1 = _mm256_loadu_ps(kernel + 3);
    k2 = _mm256_loadu_ps(kernel + 6);
    // Kernel transformation.
    // We are using eight registers for a single transformation.
    
    __m256 t0, t1, t2, t3, t4, t5, t6, t7, c0;
    t0 = k0;
    t1 = k0 + k1 + k2;
    t2 = k0 - k1 + k2;
    c0 = _mm256_set1_ps(-2.0f / 9);
    t1 = _mm256_mul_ps(c0, t1);
    t2 = _mm256_mul_ps(c0, t2);
    t3 = k0 + 2 * k1 + 4 * k2;
    t4 = k0 - 2 * k1 + 4 * k2;
    c0 = _mm256_set1_ps(1.0f / 90);
    t3 = _mm256_mul_ps(c0, t3);
    t4 = _mm256_mul_ps(c0, t4);
    t5 = 4 * k0 + 2 * k1 + k2;
    t6 = 4 * k0 - 2 * k1 + k2;
    c0 = _mm256_set1_ps(1.0f / 180);
    t5 = _mm256_mul_ps(c0, t5);
    t6 = _mm256_mul_ps(c0, t6);
    t7 = k2;

    transpose8_avx_ps(t0, t1, t2, t3, t4, t5, t6, t7);

    k0 = t0;
    k1 = t1;
    k2 = t2;

    t0 = k0;
    t1 = k0 + k1 + k2;
    t2 = k0 - k1 + k2;
    c0 = _mm256_set1_ps(-2.0f / 9);
    t1 = _mm256_mul_ps(c0, t1);
    t2 = _mm256_mul_ps(c0, t2);
    t3 = k0 + 2 * k1 + 4 * k2;
    t4 = k0 - 2 * k1 + 4 * k2;
    c0 = _mm256_set1_ps(1.0f / 90);
    t3 = _mm256_mul_ps(c0, t3);
    t4 = _mm256_mul_ps(c0, t4);
    t5 = 4 * k0 + 2 * k1 + k2;
    t6 = 4 * k0 - 2 * k1 + k2;
    c0 = _mm256_set1_ps(1.0f / 180);
    t5 = _mm256_mul_ps(c0, t5);
    t6 = _mm256_mul_ps(c0, t6);
    t7 = k2;


    k0 = _mm256_loadu_ps(kernel + kernel_stride);
    k1 = _mm256_loadu_ps(kernel + kernel_stride + 3);
    k2 = _mm256_loadu_ps(kernel + kernel_stride + 6);

    __m256 s0, s1, s2, s3, s4, s5, s6, s7;
    s0 = k0;
    s1 = k0 + k1 + k2;
    s2 = k0 - k1 + k2;
    c0 = _mm256_set1_ps(-2.0f / 9);
    s1 = _mm256_mul_ps(c0, s1);
    s2 = _mm256_mul_ps(c0, s2);
    s3 = k0 + 2 * k1 + 4 * k2;
    s4 = k0 - 2 * k1 + 4 * k2;
    c0 = _mm256_set1_ps(1.0f / 90);
    s3 = _mm256_mul_ps(c0, s3);
    s4 = _mm256_mul_ps(c0, s4);
    s5 = 4 * k0 + 2 * k1 + k2;
    s6 = 4 * k0 - 2 * k1 + k2;
    c0 = _mm256_set1_ps(1.0f / 180);
    s5 = _mm256_mul_ps(c0, s5);
    s6 = _mm256_mul_ps(c0, s6);
    s7 = k2;

    transpose8_avx_ps(s0, s1, s2, s3, s4, s5, s6, s7);

    k0 = s0;
    k1 = s1;
    k2 = s2;

    s0 = k0;
    s1 = k0 + k1 + k2;
    s2 = k0 - k1 + k2;
    c0 = _mm256_set1_ps(-2.0f / 9);
    s1 = _mm256_mul_ps(c0, s1);
    s2 = _mm256_mul_ps(c0, s2);
    s3 = k0 + 2 * k1 + 4 * k2;
    s4 = k0 - 2 * k1 + 4 * k2;
    c0 = _mm256_set1_ps(1.0f / 90);
    s3 = _mm256_mul_ps(c0, s3);
    s4 = _mm256_mul_ps(c0, s4);
    s5 = 4 * k0 + 2 * k1 + k2;
    s6 = 4 * k0 - 2 * k1 + k2;
    c0 = _mm256_set1_ps(1.0f / 180);
    s5 = _mm256_mul_ps(c0, s5);
    s6 = _mm256_mul_ps(c0, s6);
    s7 = k2;

    // Interleave register contents in order to fit TensorGEMM accessing pattern.
    k0 = _mm256_permute2f128_ps(t0, s0, 0x20);
    k1 = _mm256_permute2f128_ps(t0, s0, 0x31);
    _mm256_store_ps(transKernel, k0);
    _mm256_store_ps(transKernel + stride, k1);

    k0 = _mm256_permute2f128_ps(t1, s1, 0x20);
    k1 = _mm256_permute2f128_ps(t1, s1, 0x31);
    _mm256_store_ps(transKernel + 2 * stride, k0);
    _mm256_store_ps(transKernel + 3 * stride, k1);

    k0 = _mm256_permute2f128_ps(t2, s2, 0x20);
    k1 = _mm256_permute2f128_ps(t2, s2, 0x31);
    _mm256_store_ps(transKernel + 4 * stride, k0);
    _mm256_store_ps(transKernel + 5 * stride, k1);

    k0 = _mm256_permute2f128_ps(t3, s3, 0x20);
    k1 = _mm256_permute2f128_ps(t3, s3, 0x31);
    _mm256_store_ps(transKernel + 6 * stride, k0);
    _mm256_store_ps(transKernel + 7 * stride, k1);

    k0 = _mm256_permute2f128_ps(t4, s4, 0x20);
    k1 = _mm256_permute2f128_ps(t4, s4, 0x31);
    _mm256_store_ps(transKernel + 8 * stride, k0);
    _mm256_store_ps(transKernel + 9 * stride, k1);

    k0 = _mm256_permute2f128_ps(t5, s5, 0x20);
    k1 = _mm256_permute2f128_ps(t5, s5, 0x31);
    _mm256_store_ps(transKernel + 10 * stride, k0);
    _mm256_store_ps(transKernel + 11 * stride, k1);

    k0 = _mm256_permute2f128_ps(t6, s6, 0x20);
    k1 = _mm256_permute2f128_ps(t6, s6, 0x31);
    _mm256_store_ps(transKernel + 12 * stride, k0);
    _mm256_store_ps(transKernel + 13 * stride, k1);

    k0 = _mm256_permute2f128_ps(t7, s7, 0x20);
    k1 = _mm256_permute2f128_ps(t7, s7, 0x31);
    _mm256_store_ps(transKernel + 14 * stride, k0);
    _mm256_store_ps(transKernel + 15 * stride, k1);
}



void winogradKernelTransformPack2OutchSeq(float *transKernel, float *packed_kernel, int stride)
{
    __m256 k0, k1, k2; // different rows of kernel.
    k0 = _mm256_loadu_ps(packed_kernel);
    k1 = _mm256_loadu_ps(packed_kernel + 3);
    k2 = _mm256_loadu_ps(packed_kernel + 6);
    // Kernel transformation.
    // We are using eight registers for a single transformation.
    
    __m256 t0, t1, t2, t3, t4, t5, t6, t7, c0;
    t0 = k0;
    t1 = k0 + k1 + k2;
    t2 = k0 - k1 + k2;
    c0 = _mm256_set1_ps(-2.0f / 9);
    t1 = _mm256_mul_ps(c0, t1);
    t2 = _mm256_mul_ps(c0, t2);
    t3 = k0 + 2 * k1 + 4 * k2;
    t4 = k0 - 2 * k1 + 4 * k2;
    c0 = _mm256_set1_ps(1.0f / 90);
    t3 = _mm256_mul_ps(c0, t3);
    t4 = _mm256_mul_ps(c0, t4);
    t5 = 4 * k0 + 2 * k1 + k2;
    t6 = 4 * k0 - 2 * k1 + k2;
    c0 = _mm256_set1_ps(1.0f / 180);
    t5 = _mm256_mul_ps(c0, t5);
    t6 = _mm256_mul_ps(c0, t6);
    t7 = k2;

    transpose8_avx_ps(t0, t1, t2, t3, t4, t5, t6, t7);

    k0 = t0;
    k1 = t1;
    k2 = t2;

    t0 = k0;
    t1 = k0 + k1 + k2;
    t2 = k0 - k1 + k2;
    c0 = _mm256_set1_ps(-2.0f / 9);
    t1 = _mm256_mul_ps(c0, t1);
    t2 = _mm256_mul_ps(c0, t2);
    t3 = k0 + 2 * k1 + 4 * k2;
    t4 = k0 - 2 * k1 + 4 * k2;
    c0 = _mm256_set1_ps(1.0f / 90);
    t3 = _mm256_mul_ps(c0, t3);
    t4 = _mm256_mul_ps(c0, t4);
    t5 = 4 * k0 + 2 * k1 + k2;
    t6 = 4 * k0 - 2 * k1 + k2;
    c0 = _mm256_set1_ps(1.0f / 180);
    t5 = _mm256_mul_ps(c0, t5);
    t6 = _mm256_mul_ps(c0, t6);
    t7 = k2;


    k0 = _mm256_loadu_ps(packed_kernel + 9);
    k1 = _mm256_loadu_ps(packed_kernel + 12);
    k2 = _mm256_loadu_ps(packed_kernel + 15);

    __m256 s0, s1, s2, s3, s4, s5, s6, s7;
    s0 = k0;
    s1 = k0 + k1 + k2;
    s2 = k0 - k1 + k2;
    c0 = _mm256_set1_ps(-2.0f / 9);
    s1 = _mm256_mul_ps(c0, s1);
    s2 = _mm256_mul_ps(c0, s2);
    s3 = k0 + 2 * k1 + 4 * k2;
    s4 = k0 - 2 * k1 + 4 * k2;
    c0 = _mm256_set1_ps(1.0f / 90);
    s3 = _mm256_mul_ps(c0, s3);
    s4 = _mm256_mul_ps(c0, s4);
    s5 = 4 * k0 + 2 * k1 + k2;
    s6 = 4 * k0 - 2 * k1 + k2;
    c0 = _mm256_set1_ps(1.0f / 180);
    s5 = _mm256_mul_ps(c0, s5);
    s6 = _mm256_mul_ps(c0, s6);
    s7 = k2;

    transpose8_avx_ps(s0, s1, s2, s3, s4, s5, s6, s7);

    k0 = s0;
    k1 = s1;
    k2 = s2;

    s0 = k0;
    s1 = k0 + k1 + k2;
    s2 = k0 - k1 + k2;
    c0 = _mm256_set1_ps(-2.0f / 9);
    s1 = _mm256_mul_ps(c0, s1);
    s2 = _mm256_mul_ps(c0, s2);
    s3 = k0 + 2 * k1 + 4 * k2;
    s4 = k0 - 2 * k1 + 4 * k2;
    c0 = _mm256_set1_ps(1.0f / 90);
    s3 = _mm256_mul_ps(c0, s3);
    s4 = _mm256_mul_ps(c0, s4);
    s5 = 4 * k0 + 2 * k1 + k2;
    s6 = 4 * k0 - 2 * k1 + k2;
    c0 = _mm256_set1_ps(1.0f / 180);
    s5 = _mm256_mul_ps(c0, s5);
    s6 = _mm256_mul_ps(c0, s6);
    s7 = k2;

    // Interleave register contents in order to fit TensorGEMM accessing pattern.
    k0 = _mm256_permute2f128_ps(t0, s0, 0x20);
    k1 = _mm256_permute2f128_ps(t0, s0, 0x31);
    _mm256_store_ps(transKernel, k0);
    _mm256_store_ps(transKernel + stride, k1);

    k0 = _mm256_permute2f128_ps(t1, s1, 0x20);
    k1 = _mm256_permute2f128_ps(t1, s1, 0x31);
    _mm256_store_ps(transKernel + 2 * stride, k0);
    _mm256_store_ps(transKernel + 3 * stride, k1);

    k0 = _mm256_permute2f128_ps(t2, s2, 0x20);
    k1 = _mm256_permute2f128_ps(t2, s2, 0x31);
    _mm256_store_ps(transKernel + 4 * stride, k0);
    _mm256_store_ps(transKernel + 5 * stride, k1);

    k0 = _mm256_permute2f128_ps(t3, s3, 0x20);
    k1 = _mm256_permute2f128_ps(t3, s3, 0x31);
    _mm256_store_ps(transKernel + 6 * stride, k0);
    _mm256_store_ps(transKernel + 7 * stride, k1);

    k0 = _mm256_permute2f128_ps(t4, s4, 0x20);
    k1 = _mm256_permute2f128_ps(t4, s4, 0x31);
    _mm256_store_ps(transKernel + 8 * stride, k0);
    _mm256_store_ps(transKernel + 9 * stride, k1);

    k0 = _mm256_permute2f128_ps(t5, s5, 0x20);
    k1 = _mm256_permute2f128_ps(t5, s5, 0x31);
    _mm256_store_ps(transKernel + 10 * stride, k0);
    _mm256_store_ps(transKernel + 11 * stride, k1);

    k0 = _mm256_permute2f128_ps(t6, s6, 0x20);
    k1 = _mm256_permute2f128_ps(t6, s6, 0x31);
    _mm256_store_ps(transKernel + 12 * stride, k0);
    _mm256_store_ps(transKernel + 13 * stride, k1);

    k0 = _mm256_permute2f128_ps(t7, s7, 0x20);
    k1 = _mm256_permute2f128_ps(t7, s7, 0x31);
    _mm256_store_ps(transKernel + 14 * stride, k0);
    _mm256_store_ps(transKernel + 15 * stride, k1);
}


void winogradKernelTransformPackedPartial(float *transKernel, float *kernel, int depth_id)
{
    // float bigBlock[64];
    __m256 k0, k1, k2; // different rows of kernel.
    k0 = _mm256_loadu_ps(kernel);
    k1 = _mm256_loadu_ps(kernel + 3);
    k2 = _mm256_loadu_ps(kernel + 6);
    // Kernel transformation.
    __m256 t0, t1, t2, t3, t4, t5, t6, t7, c0;
    
    #if 1
    t0 = k0;
    t1 = k0 + k1 + k2;
    t2 = k0 - k1 + k2;
    
    
    c0 = _mm256_set1_ps(-2.0f / 9);
    t1 = _mm256_mul_ps(c0, t1);
    t2 = _mm256_mul_ps(c0, t2);
    t3 = k0 + 2 * k1 + 4 * k2;
    t4 = k0 - 2 * k1 + 4 * k2;
    c0 = _mm256_set1_ps(1.0f / 90);
    t3 = _mm256_mul_ps(c0, t3);
    t4 = _mm256_mul_ps(c0, t4);
    t5 = 4 * k0 + 2 * k1 + k2;
    t6 = 4 * k0 - 2 * k1 + k2;
    c0 = _mm256_set1_ps(1.0f / 180);
    t5 = _mm256_mul_ps(c0, t5);
    t6 = _mm256_mul_ps(c0, t6);
    t7 = k2;

    transpose8_avx_ps(t0, t1, t2, t3, t4, t5, t6, t7);

    k0 = t0;
    k1 = t1;
    k2 = t2;

    // Kernel transformation again and store.
    __m128 reg;
    int vec_id = depth_id / 2;
    const int st_id = depth_id % 2;
    if (vec_id == 0)
    {
        t0 = k0;    
        if (st_id == 0)
            reg = _mm256_extractf128_ps(t0, 0);
        else
            reg = _mm256_extractf128_ps(t0, 1);
        _mm_store_ps(transKernel, reg);
    }
    c0 = _mm256_set1_ps(-2.0f / 9);
    if (vec_id == 1)
    {
        t1 = k0 + k1 + k2;
        t1 = _mm256_mul_ps(c0, t1);
        if (st_id == 0)
            reg = _mm256_extractf128_ps(t1, 0);
        else
            reg = _mm256_extractf128_ps(t1, 1);
        _mm_store_ps(transKernel, reg);
    }
    if (vec_id == 2)
    {
        t2 = k0 - k1 + k2;
        t2 = _mm256_mul_ps(c0, t2);
        if (st_id == 0)
            reg = _mm256_extractf128_ps(t2, 0);
        else
            reg = _mm256_extractf128_ps(t2, 1);
        _mm_store_ps(transKernel, reg);
    }
    c0 = _mm256_set1_ps(1.0f / 90);
    if (vec_id == 3)
    {
        t3 = k0 + 2 * k1 + 4 * k2;
        t3 = _mm256_mul_ps(c0, t3);
        if (st_id == 0)
            reg = _mm256_extractf128_ps(t3, 0);
        else
            reg = _mm256_extractf128_ps(t3, 1);
        _mm_store_ps(transKernel, reg);
    }
    if (vec_id == 4)
    {
        t4 = k0 - 2 * k1 + 4 * k2;
        t4 = _mm256_mul_ps(c0, t4);
        if (st_id == 0)
            reg = _mm256_extractf128_ps(t4, 0);
        else
            reg = _mm256_extractf128_ps(t4, 1);
        _mm_store_ps(transKernel, reg);
    }
    c0 = _mm256_set1_ps(1.0f / 180);
    if (vec_id == 5)
    {
        t5 = 4 * k0 + 2 * k1 + k2;
        t5 = _mm256_mul_ps(c0, t5);
        if (st_id == 0)
            reg = _mm256_extractf128_ps(t5, 0);
        else
            reg = _mm256_extractf128_ps(t5, 1);
        _mm_store_ps(transKernel, reg);
    }
    if (vec_id == 6)
    {
        t6 = 4 * k0 - 2 * k1 + k2;
        t6 = _mm256_mul_ps(c0, t6);
        if (st_id == 0)
            reg = _mm256_extractf128_ps(t6, 0);
        else
            reg = _mm256_extractf128_ps(t6, 1);
        _mm_store_ps(transKernel, reg);
    }
    if (vec_id == 7)
    {
        t7 = k2;
        if (st_id == 0)
            reg = _mm256_extractf128_ps(t7, 0);
        else
            reg = _mm256_extractf128_ps(t7, 1);
        _mm_store_ps(transKernel, reg);
    }
    #else
    t0 = k0;
    t1 = k0 + k1 + k2;
    t2 = k0 - k1 + k2;
    t3 = k0;
    t4 = k0 + k1 + k2;
    t5 = k0 - k1 + k2;
    t6 = k0;
    t7 = k0 + k1 + k2;
    __m128 reg;
    reg = _mm256_extractf128_ps(t2, 1);
    _mm_store_ps(transKernel, reg);
    #endif
}


void transformKernel_F6x6_3x3_orig(float *UT, float *kernel, int input_channels, int outChannels)
{
    /* Data layout for kernel transformation (UT):
     *
     * 4 output channels always batched together
     * input channel block
     * remaining output channel loop
     * remaining input channel block loop
     */
    const int inch_cache_block = 192;
    const int outch_cache_block = outChannels;
    int inch_pass = input_channels / inch_cache_block;

    if (input_channels % inch_cache_block > 0)
        ++inch_pass;
    int inChannels = inch_cache_block;
    for (int q = 0; q < inch_pass; ++q)
    {
        int cur_inch = q * inch_cache_block;
        if (q == inch_pass - 1)
            inChannels = input_channels - cur_inch;
        for (int i = 0; i < inChannels; ++i)
        {
            for (int j = 0; j < outChannels; ++j)
            {
                int cid = j * inChannels + i;
#if 1               
                float *UTp = UT;
                UTp += cur_inch * outChannels * 16 * 4; // inch cache block, each with 64 floats.
                UTp += (j / 4) * (16 * inChannels);     //Big block id for every 4 output channels.
                UTp += 16 * i;                          //input channel offset.
                UTp += (j & 0x3) * 4;                   //Starting point in each 4 outch batch.
                winogradKernelTransformPacked(UTp, kernel + 9 * (j * input_channels + i + cur_inch), 4 * inChannels * outChannels);
#else
                float *UTp = UT + (j / 4) * (256 * inChannels) //Big block id for every 4 output channels.
                             + 16 * i                          //input channel offset.
                             + (j & 0x3) * 4                  //Starting point in each 4 outch batch.
                             + q * (outChannels * 64 * inch_cache_block);
                winogradKernelTransformPacked(UTp, kernel + 9 * (j * input_channels + i + cur_inch), 16 * inChannels, UT, i, j);
#endif
            }
        }
    }
}

inline void input_transform_avx(
    __m256 &r0,
    __m256 &r1,
    __m256 &r2,
    __m256 &r3,
    __m256 &r4,
    __m256 &r5,
    __m256 &r6,
    __m256 &r7,
    __m256 &t1,
    __m256 &t2,
    __m256 &s1,
    __m256 &s2,
    __m256 &p1,
    __m256 &p2,
    const __m256 &f5_25,
    const __m256 &f4_25,
    const __m256 &f4,
    const __m256 &f2_5,
    const __m256 &f2,
    const __m256 &f1_25,
    const __m256 &f0_5,
    const __m256 &f0_25)
{
    r0 = _mm256_add_ps(_mm256_sub_ps(r0, r6), _mm256_mul_ps(_mm256_sub_ps(r4, r2), f5_25));// r0 - r6 + (r4 - r2) * f5_25;
    r7 = _mm256_add_ps(_mm256_sub_ps(r7, r1), _mm256_mul_ps(_mm256_sub_ps(r3, r5), f5_25));// r7 - r1 + (r3 - r5) * f5_25;

    //r6 - r4 * f5_25 can be reused
    //r1 - r3 * f5_25 can be reused

    t1 = _mm256_sub_ps(_mm256_add_ps(r2, r6), _mm256_mul_ps(r4, f4_25));// r2 + r6 - r4 * f4_25;
    t2 = _mm256_sub_ps(_mm256_add_ps(r1, r5), _mm256_mul_ps(r3, f4_25));// r1 + r5 - r3 * f4_25;

    s1 = _mm256_mul_ps(r4, f1_25);// r4 * f1_25;
    s2 = _mm256_mul_ps(r3, f2_5);// r3 * f2_5;

    p1 = _mm256_add_ps(r6, _mm256_sub_ps(_mm256_mul_ps(r2, f0_25), s1));// r6 + (r2 * f0_25 - s1);
    p2 = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(r1, f0_5), s2), _mm256_mul_ps(r5, f2));// r1 * f0_5 - s2 + r5 * f2;

    r3 = _mm256_add_ps(p1, p2);// p1 + p2;
    r4 = _mm256_sub_ps(p1, p2);// p1 - p2;

    //2.5 * (r01 - r03 + r05)

    p1 = _mm256_add_ps(r6, _mm256_mul_ps(_mm256_sub_ps(r2, s1), f4));// r6 + (r2 - s1) * f4;
    p2 = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(r1, f2), s2), _mm256_mul_ps(r5, f0_5));// r1 * f2 - s2 + r5 * f0_5;

    r5 = _mm256_add_ps(p1, p2);// p1 + p2;
    r6 = _mm256_sub_ps(p1, p2);// p1 - p2;

    r1 = _mm256_add_ps(t1, t2);
    r2 = _mm256_sub_ps(t1, t2);
}

union MM256_ARRAY_UNION
{
    int arr[8];
    __m256i vec;
};

__m256i feather_mm256_get_mask(const int left_zeros, const int right_zeros)
{
    MM256_ARRAY_UNION mask;
    mask.vec = _mm256_set1_epi32(0xFFFFFFFF);
    for (int i = 0; i < left_zeros; ++i)
    {
        mask.arr[i] = 0x0;
    }
    for (int i = 0; i < right_zeros; ++i)
    {
        mask.arr[7 - i] = 0x0;
    }
    return mask.vec;
}

inline void feather_avx_load_padded_registers_8x8(
    const float* ptr,
    const int ldin,
    __m256 &r0,
    __m256 &r1,
    __m256 &r2,
    __m256 &r3,
    __m256 &r4,
    __m256 &r5,
    __m256 &r6,
    __m256 &r7,
    const int pad_left,
    const int pad_bottom,
    const int pad_right,
    const int pad_top
)
{

    //ptr points to a virtual address where data doesn't exist.
    __m256i mask = feather_mm256_get_mask(pad_left, pad_right);
    if (pad_top > 0 || pad_bottom > 7)
        r0 = _mm256_set1_ps(0.f);
    else
        r0 = _mm256_maskload_ps(ptr, mask);
    ptr += ldin;

    if (pad_top > 1 || pad_bottom > 6)
        r1 = _mm256_set1_ps(0.f);
    else
        r1 = _mm256_maskload_ps(ptr, mask);
    ptr += ldin;

    if (pad_top > 2 || pad_bottom > 5)
        r2 = _mm256_set1_ps(0.f);
    else
        r2 = _mm256_maskload_ps(ptr, mask);
    ptr += ldin;

    if (pad_top > 3 || pad_bottom > 4)
        r3 = _mm256_set1_ps(0.f);
    else
        r3 = _mm256_maskload_ps(ptr, mask);
    ptr += ldin;

    if (pad_top > 4 || pad_bottom > 3)
        r4 = _mm256_set1_ps(0.f);
    else
        r4 = _mm256_maskload_ps(ptr, mask);
    ptr += ldin;

    if (pad_top > 5 || pad_bottom > 2)
        r5 = _mm256_set1_ps(0.f);
    else
        r5 = _mm256_maskload_ps(ptr, mask);
    ptr += ldin;

    if (pad_top > 6 || pad_bottom > 1)
        r6 = _mm256_set1_ps(0.f);
    else
        r6 = _mm256_maskload_ps(ptr, mask);
    ptr += ldin;

    if (pad_top > 7 || pad_bottom > 0)
        r7 = _mm256_set1_ps(0.f);
    else
        r7 = _mm256_maskload_ps(ptr, mask);
    ptr += ldin;
}

void winogradInputTransformSeqFusedAVX4(booster::ConvParam* conv_param, float *VT, const float *input, int startIdx, int endIdx, int inch_cache_block)
{
    //Constants in transformation matrices.
    int nRowBlocks = (conv_param->output_w + 5) / 6;
    int nColBlocks = (conv_param->output_h + 5) / 6;

    const int ldin = conv_param->input_w;
    // const int inChannels = conv_param->input_channels;
    const int inChannels = inch_cache_block;
    // printf("Input channels %d\n", inChannels);
    //Constants in transformation matrices.
    const __m256 f5    = _mm256_set1_ps(5.0f);
    const __m256 f4    = _mm256_set1_ps(4.0f);
    const __m256 f2    = _mm256_set1_ps(2.0f);
    const __m256 f2_5  = _mm256_set1_ps(2.5f);
    const __m256 f5_25 = _mm256_set1_ps(5.25f);
    const __m256 f4_25 = _mm256_set1_ps(4.25f);
    const __m256 f1_25 = _mm256_set1_ps(1.25f);
    const __m256 f0_5  = _mm256_set1_ps(0.5f);
    const __m256 f0_25 = _mm256_set1_ps(0.25f);
    const __m256 vZero = _mm256_set1_ps(0.0f);

    const int frameStride = conv_param->input_h * conv_param->input_w;

    FEATHER_MEN_ALIGN(32) float quad_blocks[256];

    for (int ic = 0; ic < inChannels; ++ic)
    {
        for (int bid = startIdx; bid < endIdx; bid += 4)
        {
            __m256 l0, l1, l2, l3, l4, l5, l6, l7;
            __m256 m1, m2, s1, s2, t1, t2; //Auxiliary registers

            for (int t = 0; t < 4; ++t)
            {
                int i = (bid + t) % nRowBlocks; //x coord along the rows
                int j = (bid + t) / nRowBlocks; //y coord along the columns
                const float *p0 = input + ic * frameStride + ldin * (j * 6 - conv_param->pad_top) + i * 6 - conv_param->pad_left;
                const float *p1 = p0 + ldin;
                const float *p2 = p1 + ldin;
                const float *p3 = p2 + ldin;
                const float *p4 = p3 + ldin;
                const float *p5 = p4 + ldin;
                const float *p6 = p5 + ldin;
                const float *p7 = p6 + ldin;

                int block_pad_left = 0;
                int block_pad_right = 0;
                int block_pad_top = 0;
                int block_pad_bottom = 0;

                int row_offset  = i * 6 - conv_param->pad_left;

                if (row_offset < 0)
                {
                    block_pad_left = 0 - row_offset;
                }
                if (row_offset + 8 >= conv_param->input_w)
                {
                    block_pad_right = row_offset + 8 - conv_param->input_w;
                }

                int col_offset  = j * 6 - conv_param->pad_top;

                if (col_offset < 0)
                {
                    block_pad_top = 0 - col_offset;
                }
                if (col_offset + 8 >= conv_param->input_h)
                {
                    block_pad_bottom = col_offset + 8 - conv_param->input_h;
                }
                if (block_pad_left || block_pad_right || block_pad_top || block_pad_bottom)
                {
                    feather_avx_load_padded_registers_8x8(
                        p0, ldin, l0, l1, l2, l3, l4, l5, l6, l7,
                        block_pad_left, block_pad_bottom, block_pad_right, block_pad_top);
                }
                else
                {
                    l0 = _mm256_loadu_ps(p0);
                    l1 = _mm256_loadu_ps(p1);
                    l2 = _mm256_loadu_ps(p2);
                    l3 = _mm256_loadu_ps(p3);
                    l4 = _mm256_loadu_ps(p4);
                    l5 = _mm256_loadu_ps(p5);
                    l6 = _mm256_loadu_ps(p6);
                    l7 = _mm256_loadu_ps(p7);
                }

                input_transform_avx(l0, l1, l2, l3, l4, l5, l6, l7,                  //Target
                                    t1, t2, s1, s2, m1, m2,                          //Auxiliary
                                    f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                transpose8_avx_ps(l0, l1, l2, l3, l4, l5, l6, l7);
                input_transform_avx(l0, l1, l2, l3, l4, l5, l6, l7,                  //Target
                                    t1, t2, s1, s2, m1, m2,                          //Auxiliary
                                    f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                size_t inner_offset = t << 2;
                float* block_outp = quad_blocks + inner_offset;
                /*
                 * Packed output index:
                 * 1) Every four blocks (tiles) are batched together: (bid & 0x3) * 4
                 * 2) Elements from different channels are packed together: ic * 16
                 * 3) In each block, vectors should offset by 4 * inChannels.
                 * 4) The 4 blocks batches should be offset by 4 * 4 * inChannels:
                 *    (bid / 4) * 16 * inChannels
                 */
                _mm_store_ps(block_outp, _mm256_extractf128_ps(l0, 0x0));
                _mm_store_ps(block_outp + 16, _mm256_extractf128_ps(l0, 0x1));
                _mm_store_ps(block_outp + 32, _mm256_extractf128_ps(l1, 0x0));
                _mm_store_ps(block_outp + 48, _mm256_extractf128_ps(l1, 0x1));

                _mm_store_ps(block_outp + 64, _mm256_extractf128_ps(l2, 0x0));
                _mm_store_ps(block_outp + 80, _mm256_extractf128_ps(l2, 0x1));
                _mm_store_ps(block_outp + 96, _mm256_extractf128_ps(l3, 0x0));
                _mm_store_ps(block_outp + 112, _mm256_extractf128_ps(l3, 0x1));

                _mm_store_ps(block_outp + 128, _mm256_extractf128_ps(l4, 0x0));
                _mm_store_ps(block_outp + 144, _mm256_extractf128_ps(l4, 0x1));
                _mm_store_ps(block_outp + 160, _mm256_extractf128_ps(l5, 0x0));
                _mm_store_ps(block_outp + 176, _mm256_extractf128_ps(l5, 0x1));

                _mm_store_ps(block_outp + 192, _mm256_extractf128_ps(l6, 0x0));
                _mm_store_ps(block_outp + 208, _mm256_extractf128_ps(l6, 0x1));
                _mm_store_ps(block_outp + 224, _mm256_extractf128_ps(l7, 0x0));
                _mm_store_ps(block_outp + 240, _mm256_extractf128_ps(l7, 0x1));
            }
            size_t bid_offset = bid - startIdx;
            float *outp = VT + ic * 16 + (bid_offset / 4) * 16 * 16 * inChannels;
#pragma unroll
            for (int t = 0; t < 16; ++t)
            {
                _mm256_storeu_ps(outp + inChannels * 16 * t, _mm256_load_ps(quad_blocks + t * 16));
                _mm256_storeu_ps(outp + inChannels * 16 * t + 8, _mm256_load_ps(quad_blocks + t * 16 + 8));
            }
        }
    }
}

template<bool HAS_RELU, bool HAS_BIAS>
void WinogradOutputTransformBlockAVX(const float *WT, float *output, const int ldout, const int ldchannel, const int vx, const int vy, const int block_stride, const float bias)
{
    __m256 _l0, _l1, _l2, _l3, _l4, _l5, _l6, _l7, vZero, vBias;
    vZero = _mm256_set1_ps(0.f);
    vBias = _mm256_set1_ps(bias);
    const float *wp = WT;
    const int stride = block_stride;

    _l0 = _mm256_loadu2_m128(wp + 1 * stride, wp);
    _l1 = _mm256_loadu2_m128(wp + 3 * stride, wp + 2 * stride);
    _l2 = _mm256_loadu2_m128(wp + 5 * stride, wp + 4 * stride);
    _l3 = _mm256_loadu2_m128(wp + 7 * stride, wp + 6 * stride);
    _l4 = _mm256_loadu2_m128(wp + 9 * stride, wp + 8 * stride);
    _l5 = _mm256_loadu2_m128(wp + 11 * stride, wp + 10 * stride);
    _l6 = _mm256_loadu2_m128(wp + 13 * stride, wp + 12 * stride);
    _l7 = _mm256_loadu2_m128(wp + 15 * stride, wp + 14 * stride);

    winograd_f6k3_output_transform_inplace_avx(_l0, _l1, _l2, _l3, _l4, _l5, _l6, _l7);
    transpose8_avx_ps(_l0, _l1, _l2, _l3, _l4, _l5, _l6, _l7);
    winograd_f6k3_output_transform_inplace_avx(_l0, _l1, _l2, _l3, _l4, _l5, _l6, _l7);


    float *outp = output;

    if (HAS_BIAS)
    {
        _l0 = _mm256_add_ps(_l0, vBias);
        _l1 = _mm256_add_ps(_l1, vBias);
        _l2 = _mm256_add_ps(_l2, vBias);
        _l3 = _mm256_add_ps(_l3, vBias);
        _l4 = _mm256_add_ps(_l4, vBias);
        _l5 = _mm256_add_ps(_l5, vBias);
        _l6 = _mm256_add_ps(_l6, vBias);
        _l7 = _mm256_add_ps(_l7, vBias);
    }

    if (HAS_RELU)
    {
        _l0 = _mm256_max_ps(_l0, vZero);
        _l1 = _mm256_max_ps(_l1, vZero);
        _l2 = _mm256_max_ps(_l2, vZero);
        _l3 = _mm256_max_ps(_l3, vZero);
        _l4 = _mm256_max_ps(_l4, vZero);
        _l5 = _mm256_max_ps(_l5, vZero);
        _l6 = _mm256_max_ps(_l6, vZero);
        _l7 = _mm256_max_ps(_l7, vZero);
    }

    if (vx == 0 && vy == 0)
    {
        __m256i fixed_mask6 = _mm256_set_epi32(0x0, 0x0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        _mm256_maskstore_ps(outp,             fixed_mask6, _l0);
        _mm256_maskstore_ps(outp + ldout,     fixed_mask6, _l1);
        _mm256_maskstore_ps(outp + 2 * ldout, fixed_mask6, _l2);
        _mm256_maskstore_ps(outp + 3 * ldout, fixed_mask6, _l3);
        _mm256_maskstore_ps(outp + 4 * ldout, fixed_mask6, _l4);
        _mm256_maskstore_ps(outp + 5 * ldout, fixed_mask6, _l5);
    }
    else
    {
        int leftover_x = 6 + vx;
        int leftover_y = 6 + vy;
        //printf("leftover %d %d\n", leftover_x, leftover_y);
        __m256i leftover_mask = feather_mm256_get_mask(0, 8 - leftover_x);
        if (leftover_y >= 1)
        {
            _mm256_maskstore_ps(outp, leftover_mask, _l0);
            outp += ldout;
        }
        if (leftover_y >= 2)
        {
            _mm256_maskstore_ps(outp, leftover_mask, _l1);
            outp += ldout;
        }
        if (leftover_y >= 3)
        {
            _mm256_maskstore_ps(outp, leftover_mask, _l2);
            outp += ldout;
        }
        if (leftover_y >= 4)
        {
            _mm256_maskstore_ps(outp, leftover_mask, _l3);
            outp += ldout;
        }
        if (leftover_y >= 5)
        {
            _mm256_maskstore_ps(outp, leftover_mask, _l4);
            outp += ldout;
        }
        if (leftover_y >= 6)
        {
            _mm256_maskstore_ps(outp, leftover_mask, _l5);
        }
    }
}

static inline void TensorGEMMInnerKernel4x4x4_avx(float *WTp, const float *UTp, const float *vp, const int inChannels, bool set_zero)
{
    __m256 vc00, vc01;
    __m256 vc10, vc11;
    __m256 vc20, vc21;
    __m256 vc30, vc31;
    __m256 u0, u1, u2, u3;
    __m256 v0, v1;
    if (set_zero)
    {
        vc00 = _mm256_set1_ps(0.f);
        vc01 = _mm256_set1_ps(0.f);
        vc10 = _mm256_set1_ps(0.f);
        vc11 = _mm256_set1_ps(0.f);
        vc20 = _mm256_set1_ps(0.f);
        vc21 = _mm256_set1_ps(0.f);
        vc30 = _mm256_set1_ps(0.f);
        vc31 = _mm256_set1_ps(0.f);
    }
    else
    {
        vc00 = _mm256_load_ps(WTp);
        vc01 = _mm256_load_ps(WTp + 8);
        vc10 = _mm256_load_ps(WTp + 16);
        vc11 = _mm256_load_ps(WTp + 24);
        vc20 = _mm256_load_ps(WTp + 32);
        vc21 = _mm256_load_ps(WTp + 40);
        vc30 = _mm256_load_ps(WTp + 48);
        vc31 = _mm256_load_ps(WTp + 56);
    }
    //const float *up = UTp;
    const __m128* vup = (const __m128*) UTp;
    u0 = _mm256_broadcast_ps(vup);
    u1 = _mm256_broadcast_ps(vup + 1);
    int inChannels_aligned = inChannels & 0xFFFFFFFE;
    for (int ic = 0; ic < inChannels_aligned; ic += 2)
    {
        v0 = _mm256_load_ps(vp);
        v1 = _mm256_load_ps(vp + 8);

        u2 = _mm256_broadcast_ps(vup + 2);
        u3 = _mm256_broadcast_ps(vup + 3);

        vc00 = _mm256_fmadd_ps(u0, v0, vc00);
        vc01 = _mm256_fmadd_ps(u0, v1, vc01);
        vc10 = _mm256_fmadd_ps(u1, v0, vc10);
        vc11 = _mm256_fmadd_ps(u1, v1, vc11);
        
        u0 = _mm256_broadcast_ps(vup + 4);
        u1 = _mm256_broadcast_ps(vup + 5);
        vc20 = _mm256_fmadd_ps(u2, v0, vc20);
        vc21 = _mm256_fmadd_ps(u2, v1, vc21);
        vc30 = _mm256_fmadd_ps(u3, v0, vc30);
        vc31 = _mm256_fmadd_ps(u3, v1, vc31);


        v0 = _mm256_load_ps(vp + 16);
        v1 = _mm256_load_ps(vp + 24);
        u2 = _mm256_broadcast_ps(vup + 6);
        u3 = _mm256_broadcast_ps(vup + 7);

        vp += 32;
        vup += 8;

        vc00 = _mm256_fmadd_ps(u0, v0, vc00);
        vc01 = _mm256_fmadd_ps(u0, v1, vc01);
        vc10 = _mm256_fmadd_ps(u1, v0, vc10);
        vc11 = _mm256_fmadd_ps(u1, v1, vc11);
        u0 = _mm256_broadcast_ps(vup);
        u1 = _mm256_broadcast_ps(vup + 1);
        vc20 = _mm256_fmadd_ps(u2, v0, vc20);
        vc21 = _mm256_fmadd_ps(u2, v1, vc21);
        vc30 = _mm256_fmadd_ps(u3, v0, vc30);
        vc31 = _mm256_fmadd_ps(u3, v1, vc31);

    }
    if (inChannels & 0x1) //Odd numbers
    {
        // printf("ic channels %d\n", inChannels);
        v0 = _mm256_load_ps(vp);
        v1 = _mm256_load_ps(vp + 8);
        u0 = _mm256_broadcast_ps(vup);
        u1 = _mm256_broadcast_ps(vup + 1);
        u2 = _mm256_broadcast_ps(vup + 2);
        u3 = _mm256_broadcast_ps(vup + 3);

        vc00 = _mm256_fmadd_ps(u0, v0, vc00);
        vc01 = _mm256_fmadd_ps(u0, v1, vc01);
        vc10 = _mm256_fmadd_ps(u1, v0, vc10);
        vc11 = _mm256_fmadd_ps(u1, v1, vc11);

        vc20 = _mm256_fmadd_ps(u2, v0, vc20);
        vc21 = _mm256_fmadd_ps(u2, v1, vc21);
        vc30 = _mm256_fmadd_ps(u3, v0, vc30);
        vc31 = _mm256_fmadd_ps(u3, v1, vc31);
    }
    float *wp = WTp;
    _mm256_store_ps(wp, vc00);
    _mm256_store_ps(wp + 8, vc01);
    _mm256_store_ps(wp + 16, vc10);
    _mm256_store_ps(wp + 24, vc11);
    _mm256_store_ps(wp + 32, vc20);
    _mm256_store_ps(wp + 40, vc21);
    _mm256_store_ps(wp + 48, vc30);
    _mm256_store_ps(wp + 56, vc31);
}

template <bool HAS_RELU, bool HAS_BIAS>
void ComputeCacheBlockFused(booster::ConvParam *conv_param, const float* UT, float* VT, float* WT, int inch_cache_block, int inch_pass, int outch_cache_block, int img_cache_block, int start_outch_id, int end_outch_id, int start_block_id, int end_block_id, float* UT_tmp_arr)
{
    // printf("allocating UT tmp arr %fKB\n", 4.0 * sizeof(float) * inch_cache_block * outch_cache_block * 64 / 1024.0);
    
    int nRowBlocks = (conv_param->output_w + 5) / 6;
    const int depth = 16;
    int cur_inch_cache_block = inch_cache_block;
    int cur_outch_cache_block = end_outch_id - start_outch_id;
    // const float* UTpt = UT;

    for (int q = 0; q < inch_pass; ++q)
    {
        int cur_inch = q * inch_cache_block;
        const float *input_at_channel_block = conv_param->input_fp32 + cur_inch * conv_param->input_h * conv_param->input_w;
        if (q == inch_pass - 1)
            cur_inch_cache_block = conv_param->input_channels - cur_inch;
        //Winograd Input Transform
        winogradInputTransformSeqFusedAVX4(conv_param, VT, input_at_channel_block, start_block_id, end_block_id, cur_inch_cache_block);

#ifdef BLOCK_KERNEL_TRANSFORM
        /* Kernel transform */
        // {
        //     const int inch_block = cur_inch_cache_block;
        //     for (int ic = 0; ic < inch_block; ++ic)
        //     {
        //         for (int oc = start_outch_id; oc < end_outch_id; ++oc)
        //         {
        //             // const int inch_block = cur_inch_cache_block;
        //             const int oc_offset = oc - start_outch_id;
        //             // float *UTp = UT_tmp_arr + start_outch_id * conv_param->input_channels * 64; // outch block.
        //             float* UTp = UT_tmp_arr;
        //             // UTp += cur_inch * cur_outch_block * 64;  // inch cache block, each kernel is transformed into one 8x8 tile.
        //             UTp += (oc_offset / 4) * (16 * inch_block); // Big block id for every 4 output channels.
        //             UTp += 16 * ic;                             // input channel offset.
        //             UTp += (oc_offset & 0x3) * 4;               // Starting point in each 4 outch batch.
        //             const int cid = oc * conv_param->input_channels + ic + cur_inch;
        //             winogradKernelTransformPacked(UTp, conv_param->kernel_fp32 + 9 * cid, 4 * inch_block * (end_outch_id - start_outch_id));
        //         }
        //     }
        // }

        // for(int d = 0; d < depth; ++d)
        // {  
        //     printf("===============d=%d================\n", d);
        //     print_floats(UT_tmp_arr + d * cur_inch_cache_block * cur_outch_cache_block * 4, 64, 16);
            
        // }
#endif
        for (int oc = start_outch_id; oc < end_outch_id; oc += 4)
        {
            // const float *UTp = UT + cur_inch * cur_outch_cache_block * 64 + d * cur_inch_cache_block * cur_outch_cache_block * 4;
            // const float *UTpt = UTp;
            // printf("cur inch %d\n", cur_inch);
            //Kernel transformation
            {
                const int inch_block = cur_inch_cache_block;
                for (int ic = 0; ic < inch_block; ++ic)
                {
                    // for (int oc = start_outch_id; oc < end_outch_id; ++oc)
                    for (int t = 0; t < 2; ++t)
                    {
                        // int oc = oc;
                        // const int inch_block = cur_inch_cache_block;
                        // int oc_offset = t * 2;
                        // float *UTp = UT_tmp_arr + start_outch_id * conv_param->input_channels * 64; // outch block.
                        float *UTp = UT_tmp_arr;
                        // UTp += cur_inch * cur_outch_block * 64;  // inch cache block, each kernel is transformed into one 8x8 tile.
                        // UTp += (oc_offset / 4) * (16 * inch_block); // Big block id for every 4 output channels.
                        UTp += 16 * ic;                             // input channel offset.
                        UTp += t * 8;               // Starting point in each 4 outch batch.
#if 0
                        const int cid = (oc + t + t) * conv_param->input_channels + ic + cur_inch;
                        winogradKernelTransformPack2Outch(UTp, conv_param->kernel_fp32 + 9 * cid, conv_param->input_channels * 9, 4 * inch_block * 4);//4 outch together
#else
#if 0
                        const int cid = oc * conv_param->input_channels + (ic + cur_inch) * 4 + t + t;
#else
                        const int cid = cur_inch * conv_param->output_channels + oc * cur_inch_cache_block + ic * 4 + t + t;
#endif
                        winogradKernelTransformPack2OutchSeq(UTp, conv_param->processed_kernel_fp32 + 9 * cid, 4 * inch_block * 4); //4 outch together
#endif
                        // printf("kernel offset %d\n", cid * 9);
                        // printf("UT offset %d\n", UTp - UT_tmp_arr);
                    }
                }
            }

            // for(int d = 0; d < depth; ++d)
            // {  
            //     printf("===============d=%d================\n", d);
            //     print_floats(UT_tmp_arr + d * cur_inch_cache_block * 4, 64, 16);
            
            // }
            const float *UTpt = UT_tmp_arr;
            ////////////////////////

            // printf("offset %d\n", cur_inch * cur_outch_cache_block * 64 + d * cur_inch_cache_block * cur_outch_cache_block * 4);
            for (int d = 0; d < depth; ++d)
            {
                //Range in a small cache block. I hope this part of VT resides in L1d cache (32KB).
                for (int i = start_block_id; i < end_block_id; i += 4)
                {
                    /* VT pointer offsets:
                                    * 1) 4 tiles are batched together
                                    * 2) First 4 floats in each tile from all inChannels are consecutive.
                                    * Therefore, depth should stride by 16 * inChannels.
                                    * 3) 4 tiles have 16 * inChannels * 16 floats in total:
                                    *     bid / 4 * (inChannels * 16 * 16)
                                    */
                    const float *vp = VT + d * 16 * cur_inch_cache_block + ((i - start_block_id) / 4) * cur_inch_cache_block * 64 * 4;

                    /* WT layout by fused very small buffer
                                    * 1) Each time access 4 (output channels) * 16 (tile elements)
                                    * 2) 4 tiles are computed in 16 (depth) loops.
                                    */
                    float *WTp = WT + 64 * d + (i - start_block_id) * 256 + (oc - start_outch_id) * img_cache_block * 256;
                    TensorGEMMInnerKernel4x4x4_avx(WTp, UTpt, vp, cur_inch_cache_block, q == 0);
                }
                UTpt += cur_inch_cache_block * 16;
            }
        }
    }
    /* Output Transform */
    for (int oc = start_outch_id; oc < end_outch_id; oc += 4)
    {
        // printf("oc %d\n", oc);
        for (int i = start_block_id; i < end_block_id; i += 4)
        {
            for (int tc = 0; tc < 4; ++tc)
            {
                for (int ti = 0; ti < 4; ++ti)
                {
                    const int ldout = conv_param->output_w;
                    const int ldchannel = conv_param->output_h * conv_param->output_w;
                    int bidx = (i + ti) % nRowBlocks;
                    int bidy = (i + ti) / nRowBlocks;
                    float *outp = conv_param->output_fp32 + bidx * 6 + bidy * 6 * ldout + (oc + tc) * ldchannel;
                    int vx = conv_param->output_h - bidx * 6 - 6;
                    int vy = conv_param->output_w - bidy * 6 - 6;
                    vx = std::min<int>(vx, 0);
                    vy = std::min<int>(vy, 0);
                    float bias_value = 0.f;
                    if (conv_param->bias_term)
                        bias_value = conv_param->bias_fp32[oc + tc];
                    if (vx < -6 || vy < -6)
                        continue;
                    float *WTp = WT + (i - start_block_id) * 256 + ti * 4 + tc * 16 + (oc - start_outch_id) * img_cache_block * 256;
                    WinogradOutputTransformBlockAVX<HAS_RELU, HAS_BIAS>(WTp, outp, ldout, ldchannel, vx, vy, 64, bias_value);
                }
            }
        }
    }
}


#define OUTCH_BLOCK_TEST
template <bool HAS_RELU, bool HAS_BIAS>
void WinogradF63Fused(booster::ConvParam *conv_param, float *output, const float *input, const float *transformed_weights, const float *bias_arr, float *buffers)
{
    // print_floats(conv_param->processed_kernel_fp32, 16 * 16, 64);
    assert(input == conv_param->input_fp32);
    // int num_threads = thpool->threadNum();
    int num_threads = conv_param->thpool->threadNum();
    // printf("num threads %d\n", num_threads);
    int nRowBlocks = (conv_param->output_w + 5) / 6;
    int nColBlocks = (conv_param->output_h + 5) / 6;
    int nBlocks = nRowBlocks * nColBlocks;
    /*
     * The AVX impl origins from an SSE approach.
     * Each 256 vector as composition of dual 128-bit vectors. 
     * The depth is 16 due to 64 elems in each tile is held in 16 128-bit vectors.
     * Each mm256 vector holds tensors for 2 Winograd tiles.
     */
    const int depth = 16;
    const int img_cache_block = 24;
    const int inch_cache_block = 16;
#ifdef OUTCH_BLOCK_TEST
    // const int outch_cache_block = 104;
    const int outch_cache_block = 88;
#else
    const int outch_cache_block = conv_param->output_channels;
#endif
    // The buffer size for each thread.
    // const int thread_buffer_stride = img_cache_block * 64 * conv_param->input_channels * 4 + conv_param->output_channels * 64 * 4 * img_cache_block;
    const int thread_buffer_stride = img_cache_block * 64 * inch_cache_block * 4 + outch_cache_block * 64 * 4 * img_cache_block;

    // The UT buffer offset after VT start pos, which is the size of VT.
    // const int UT_offset = img_cache_block * 64 * conv_param->input_channels * 4;
    const int VT_offset = img_cache_block * 64 * inch_cache_block * 4;
    // Slice the images into blocks.
    int img_pass = nBlocks / img_cache_block;
    int inch_pass = conv_param->input_channels / inch_cache_block;
    int outch_pass = conv_param->output_channels / outch_cache_block;
    if (nBlocks % img_cache_block > 0)
        ++img_pass;
    if (conv_param->input_channels % inch_cache_block > 0)
        ++inch_pass;
    if (conv_param->output_channels % outch_cache_block > 0)
        ++outch_pass;
    int cur_outch_cache_block = outch_cache_block;
 
    float* UT_tmp_arr_base = (float*) _mm_malloc(sizeof(float) * 4 * inch_cache_block * outch_cache_block * 64 * conv_param->thpool->threadNum(), 32);

    std::vector<std::future<void> > thread_futures;
    thread_futures.reserve(img_pass * outch_pass);    

    for (int r = 0; r < outch_pass; ++r)
    {
        for (int p = 0; p < img_pass; p++)
        {
            int start_block_id = p * img_cache_block;
            int end_block_id = start_block_id + img_cache_block;
            end_block_id = std::min<int>(end_block_id, nBlocks);

            int start_outch_id = r * outch_cache_block;
            int end_outch_id = (r + 1) * outch_cache_block;
            end_outch_id = std::min<int>(end_outch_id, conv_param->output_channels);

            thread_futures.emplace_back(
                conv_param->thpool->enqueue([conv_param, thread_buffer_stride, inch_cache_block, inch_pass, outch_cache_block, img_cache_block, start_outch_id, end_outch_id, start_block_id, end_block_id, UT_tmp_arr_base]
                {
                int tid = conv_param->thpool->threadID(std::this_thread::get_id());
                float *VT = conv_param->common_buffer_fp32 + tid * thread_buffer_stride;
                float *WT = VT + VT_offset;
                const float *UT = conv_param->processed_kernel_fp32 + start_outch_id * conv_param->input_channels * 64;
                float* UT_tmp_arr = UT_tmp_arr_base + tid * 4 * inch_cache_block * outch_cache_block * 64;
                // printf("outch %d %d\n", start_outch_id, end_outch_id);
                
                ComputeCacheBlockFused<HAS_RELU, HAS_BIAS>(conv_param, UT, VT, WT,
                                                      inch_cache_block, inch_pass,
                                                      outch_cache_block, img_cache_block,
                                                      start_outch_id, end_outch_id,
                                                      start_block_id, end_block_id, UT_tmp_arr);
                }));
        }
    }
    for (auto &&result : thread_futures)
    {
        result.get();
    }
    thread_futures.clear();
    _mm_free(UT_tmp_arr_base);
}

template void WinogradF63Fused<false, false>(booster::ConvParam* conv_param, float* output, const float* input, const float* transformed_weights, const float* bias, float* buffers);
template void WinogradF63Fused<false, true>(booster::ConvParam* conv_param, float* output, const float* input, const float* transformed_weights, const float* bias, float* buffers);
template void WinogradF63Fused<true, false>(booster::ConvParam* conv_param, float* output, const float* input, const float* transformed_weights, const float* bias, float* buffers);
template void WinogradF63Fused<true, true>(booster::ConvParam* conv_param, float* output, const float* input, const float* transformed_weights, const float* bias, float* buffers);


template <bool HAS_RELU, bool HAS_BIAS>
void ComputeCacheBlock(booster::ConvParam *conv_param, const float* UT, float* VT, float* WT, int inch_cache_block, int inch_pass, int outch_cache_block, int img_cache_block, int start_outch_id, int end_outch_id, int start_block_id, int end_block_id)
{
    int nRowBlocks = (conv_param->output_w + 5) / 6;
    const int depth = 16;
    int cur_inch_cache_block = inch_cache_block;
    int cur_outch_cache_block = end_outch_id - start_outch_id;
    // const float* UTpt = UT;

    for (int q = 0; q < inch_pass; ++q)
    {
        int cur_inch = q * inch_cache_block;
        const float *input_at_channel_block = conv_param->input_fp32 + cur_inch * conv_param->input_h * conv_param->input_w;
        if (q == inch_pass - 1)
            cur_inch_cache_block = conv_param->input_channels - cur_inch;
        // printf("inch %d\n", cur_inch_cache_block);
        //Winograd Input Transform
        winogradInputTransformSeqFusedAVX4(conv_param, VT, input_at_channel_block, start_block_id, end_block_id, cur_inch_cache_block);

        /* In order to replay the VT cache, the depth loop is placed outside the outch loop .*/

        // const float* UTp = UT + cur_inch * cur_outch_cache_block * 64;

        for (int d = 0; d < depth; ++d)
        {
            const float *UTp = UT + cur_inch * cur_outch_cache_block * 64 + d * cur_inch_cache_block * cur_outch_cache_block * 4;
            const float *UTpt = UTp;
            for (int oc = start_outch_id; oc < end_outch_id; oc += 4)
            {
                //Range in a small cache block. I hope this part of VT resides in L1d cache (32KB).
                for (int i = start_block_id; i < end_block_id; i += 4)
                {
                    /* VT pointer offsets:
                                    * 1) 4 tiles are batched together
                                    * 2) First 4 floats in each tile from all inChannels are consecutive.
                                    * Therefore, depth should stride by 16 * inChannels.
                                    * 3) 4 tiles have 16 * inChannels * 16 floats in total:
                                    *     bid / 4 * (inChannels * 16 * 16)
                                    */
                    const float *vp = VT + d * 16 * cur_inch_cache_block + ((i - start_block_id) / 4) * cur_inch_cache_block * 64 * 4;

                    /* WT layout by fused very small buffer
                                    * 1) Each time access 4 (output channels) * 16 (tile elements)
                                    * 2) 4 tiles are computed in 16 (depth) loops.
                                    */
                    float *WTp = WT + 64 * d + (i - start_block_id) * 256 + (oc - start_outch_id) * img_cache_block * 256;
                    TensorGEMMInnerKernel4x4x4_avx(WTp, UTpt, vp, cur_inch_cache_block, q == 0);
                }
                UTpt += cur_inch_cache_block * 16;
                // UTpt += cur_inch_cache_block * 8;
            }
        }
    }
    #if 1
    /* Output Transform */
    for (int oc = start_outch_id; oc < end_outch_id; oc += 4)
    {
        // printf("oc %d\n", oc);
        for (int i = start_block_id; i < end_block_id; i += 4)
        {
            for (int tc = 0; tc < 4; ++tc)
            {
                for (int ti = 0; ti < 4; ++ti)
                {
                    const int ldout = conv_param->output_w;
                    const int ldchannel = conv_param->output_h * conv_param->output_w;
                    int bidx = (i + ti) % nRowBlocks;
                    int bidy = (i + ti) / nRowBlocks;
                    float *outp = conv_param->output_fp32 + bidx * 6 + bidy * 6 * ldout + (oc + tc) * ldchannel;
                    int vx = conv_param->output_h - bidx * 6 - 6;
                    int vy = conv_param->output_w - bidy * 6 - 6;
                    vx = std::min<int>(vx, 0);
                    vy = std::min<int>(vy, 0);
                    float bias_value = 0.f;
                    if (conv_param->bias_term)
                        bias_value = conv_param->bias_fp32[oc + tc];
                    if (vx < -6 || vy < -6)
                        continue;
                    float *WTp = WT + (i - start_block_id) * 256 + ti * 4 + tc * 16 + (oc - start_outch_id) * img_cache_block * 256;
                    WinogradOutputTransformBlockAVX<HAS_RELU, HAS_BIAS>(WTp, outp, ldout, ldchannel, vx, vy, 64, bias_value);
                }
            }
        }
    }
    #endif
}

#define OUTCH_BLOCK_TEST
template <bool HAS_RELU, bool HAS_BIAS>
void WinogradF63Fused0(booster::ConvParam *conv_param, float *output, const float *input, const float *transformed_weights, const float *bias_arr, float *buffers)
{
    // print_floats(conv_param->processed_kernel_fp32, 16 * 16, 64);
    assert(input == conv_param->input_fp32);
    // int num_threads = thpool->threadNum();
    int num_threads = conv_param->thpool->threadNum();
    // printf("num threads %d\n", num_threads);
    int nRowBlocks = (conv_param->output_w + 5) / 6;
    int nColBlocks = (conv_param->output_h + 5) / 6;
    int nBlocks = nRowBlocks * nColBlocks;
    /*
     * The AVX impl origins from an SSE approach.
     * Each 256 vector as composition of dual 128-bit vectors. 
     * The depth is 16 due to 64 elems in each tile is held in 16 128-bit vectors.
     * Each mm256 vector holds tensors for 2 Winograd tiles.
     */
    const int depth = 16;
    const int img_cache_block = 32;
    const int inch_cache_block = 192;
#ifdef OUTCH_BLOCK_TEST
    const int outch_cache_block = 96;
#else
    const int outch_cache_block = conv_param->output_channels;
#endif
    // The buffer size for each thread.
    // const int thread_buffer_stride = img_cache_block * 64 * conv_param->input_channels * 4 + conv_param->output_channels * 64 * 4 * img_cache_block;
    const int thread_buffer_stride = img_cache_block * 64 * inch_cache_block * 4 + outch_cache_block * 64 * 4 * img_cache_block;

    // The UT buffer offset after VT start pos, which is the size of VT.
    // const int UT_offset = img_cache_block * 64 * conv_param->input_channels * 4;
    const int VT_offset = img_cache_block * 64 * inch_cache_block * 4;
    // Slice the images into blocks.
    int img_pass = nBlocks / img_cache_block;
    int inch_pass = conv_param->input_channels / inch_cache_block;
    int outch_pass = conv_param->output_channels / outch_cache_block;
    if (nBlocks % img_cache_block > 0)
        ++img_pass;
    if (conv_param->input_channels % inch_cache_block > 0)
        ++inch_pass;
    if (conv_param->output_channels % outch_cache_block > 0)
        ++outch_pass;
    int cur_outch_cache_block = outch_cache_block;

    std::vector<std::future<void> > thread_futures;
    thread_futures.reserve(img_pass * outch_pass);    

    for (int r = 0; r < outch_pass; ++r)
    {
        for (int p = 0; p < img_pass; p++)
        {
            int start_block_id = p * img_cache_block;
            int end_block_id = start_block_id + img_cache_block;
            end_block_id = std::min<int>(end_block_id, nBlocks);

            int start_outch_id = r * outch_cache_block;
            int end_outch_id = (r + 1) * outch_cache_block;
            end_outch_id = std::min<int>(end_outch_id, conv_param->output_channels);

            thread_futures.emplace_back(
                conv_param->thpool->enqueue([conv_param, thread_buffer_stride, inch_cache_block, inch_pass, outch_cache_block, img_cache_block, start_outch_id, end_outch_id, start_block_id, end_block_id]
                {
                int tid = conv_param->thpool->threadID(std::this_thread::get_id());
                float *VT = conv_param->common_buffer_fp32 + tid * thread_buffer_stride;
                float *WT = VT + VT_offset;
                const float *UT = conv_param->processed_kernel_fp32 + start_outch_id * conv_param->input_channels * 64;
                // printf("outch %d %d\n", start_outch_id, end_outch_id);
                
                ComputeCacheBlock<HAS_RELU, HAS_BIAS>(conv_param, UT, VT, WT,
                                                      inch_cache_block, inch_pass,
                                                      outch_cache_block, img_cache_block,
                                                      start_outch_id, end_outch_id,
                                                      start_block_id, end_block_id);
                }));
        }
    }
    for (auto &&result : thread_futures)
    {
        result.get();
    }
    thread_futures.clear();
}

template void WinogradF63Fused0<false, false>(booster::ConvParam* conv_param, float* output, const float* input, const float* transformed_weights, const float* bias, float* buffers);
template void WinogradF63Fused0<false, true>(booster::ConvParam* conv_param, float* output, const float* input, const float* transformed_weights, const float* bias, float* buffers);
template void WinogradF63Fused0<true, false>(booster::ConvParam* conv_param, float* output, const float* input, const float* transformed_weights, const float* bias, float* buffers);
template void WinogradF63Fused0<true, true>(booster::ConvParam* conv_param, float* output, const float* input, const float* transformed_weights, const float* bias, float* buffers);

void transformKernel_F6x6_3x3(float *UT, float *kernel, int input_channels, int output_channels)
{
    const int inch_cache_block = 16;
    int inch_pass = input_channels / inch_cache_block;
    if (input_channels % inch_cache_block > 0)
        ++inch_pass;
    
    //Reshape to let every 4 kernels from the sam outch placed together.
    for (int q = 0; q < inch_pass; ++q)
    {
        int cur_inch = q * inch_cache_block;
        int cur_inch_block = std::min<int>(inch_cache_block, input_channels - cur_inch);
        for (int oc = 0; oc < output_channels; oc += 4)
        {
            for (int ic = 0; ic < cur_inch_block; ++ic)
            {
                for (int t = 0; t < 4; ++t)
                {
                    const float *ldp = kernel + ((oc + t) * input_channels + ic + cur_inch) * 9;
                    float *wp = UT + (cur_inch * output_channels + oc * cur_inch_block + ic * 4 + t) * 9;
                    memcpy(wp, ldp, 9 * sizeof(float));
                }
            }
        }
    }
    // print_floats(kernel, output_channels * input_channels, 9);
    // print_floats(UT, output_channels * input_channels, 9);
}

void transformKernel_F6x6_3x3_1(float *UT, float *kernel, int input_channels, int output_channels)
{
    //Reshape to let every 4 kernels from the sam outch placed together.
    for (int oc = 0; oc < output_channels; oc += 4)
    {
        for (int ic = 0; ic < input_channels; ++ic)
        {
            for (int t = 0; t < 4; ++t)
            {
                const float* ldp = kernel + ((oc + t) * input_channels + ic) * 9;
                float* wp = UT + (oc * input_channels + ic * 4 + t) * 9;
                memcpy(wp, ldp, 9 * sizeof(float));
            }
        }
    }
    // print_floats(kernel, output_channels * input_channels, 9);
    // print_floats(UT, output_channels * input_channels, 9);
}

void transformKernel_F6x6_3x3_0(float *UT, float *kernel, int input_channels, int output_channels)
{
    /* Data layout for kernel transformation (UT):
     *
     * 4 output channels always batched together
     * input channel block
     * remaining output channel loop
     * remaining input channel block loop
     */
    const int inch_cache_block = 192;
#ifdef OUTCH_BLOCK_TEST
    const int outch_cache_block = 96;
#else
    const int outch_cache_block = output_channels;
#endif
    int inch_pass = input_channels / inch_cache_block;
    int outch_pass = output_channels / outch_cache_block;
    if (input_channels % inch_cache_block > 0)
        ++inch_pass;
    if (output_channels % outch_cache_block > 0)
        ++outch_pass;


    for (int r = 0; r < outch_pass; ++r)
    {
        int start_outch_id = r * outch_cache_block;
        int end_outch_id = (r + 1) * outch_cache_block;
        end_outch_id = std::min<int>(end_outch_id, output_channels);
        int cur_outch_block = end_outch_id - start_outch_id;
        int inch_block = inch_cache_block;

        for (int q = 0; q < inch_pass; ++q)
        {
            int cur_inch = q * inch_cache_block;
            if (q == inch_pass - 1)
                inch_block = input_channels - cur_inch;
            // printf("inch block %d\n", inch_block);
            for (int ic = 0; ic < inch_block; ++ic)
            {
                for (int oc = start_outch_id; oc < end_outch_id; ++oc)
                {
                    const int oc_offset = oc - start_outch_id;
                    float *UTp = UT + start_outch_id * input_channels * 64; // outch block.
                    UTp += cur_inch * cur_outch_block * 64;     // inch cache block, each kernel is transformed into one 8x8 tile.
                    UTp += (oc_offset / 4) * (16 * inch_block); // Big block id for every 4 output channels.
                    UTp += 16 * ic;                             // input channel offset.
                    UTp += (oc_offset & 0x3) * 4;               // Starting point in each 4 outch batch.
                    const int cid = oc * input_channels + ic + cur_inch;
                    winogradKernelTransformPacked(UTp, kernel + 9 * cid, 4 * inch_block * cur_outch_block);
                }
            }
        }
    }
}
}; // namespace fused