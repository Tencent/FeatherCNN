#include <booster/booster.h>
#include <booster/sgeconv.h>
#include <booster/helper.h>

// #include <immintrin.h>
#include <arm_neon.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

namespace booster
{
int align_ceil(int num, int align)
{
    return num + (align - (num % align)) % align;
}

typedef void (*InnerKernel)(int K, float *packA, float *packB, float *c, int ldc);

void memcpy_floats_neon(float* dst, float* src, int len)
{
    int len_floor = len - len % 16;
    int last_len = len % 16;
    for (int i = 0; i < len_floor; i += 16)
    {
        vst1q_f32(dst,      vld1q_f32(src));
        vst1q_f32(dst +  4, vld1q_f32(src + 4));
        vst1q_f32(dst +  8, vld1q_f32(src + 8));
        vst1q_f32(dst + 12, vld1q_f32(src + 12));
        src += 16;
        dst += 16;
    }
    int offset = 0;
    if (last_len >= 12)
    {
        vst1q_f32(dst,      vld1q_f32(src));
        vst1q_f32(dst +  4, vld1q_f32(src + 4));
        vst1q_f32(dst +  8, vld1q_f32(src + 8));
        offset = 12;
        src += 12;
        dst += 12;
    }
    else if (last_len >= 8)
    {
        vst1q_f32(dst,      vld1q_f32(src));
        vst1q_f32(dst +  4, vld1q_f32(src + 4));
        offset = 8;
        src += 8;
        dst += 8;
    }
    else if (last_len >= 4)
    {
        vst1q_f32(dst,      vld1q_f32(src));
        offset = 4;
        src += 4;
        dst += 4;
    }
    last_len -= offset;
    if (last_len > 0)
    {
        for (int i = 0; i < last_len; ++i)
        {
            dst[i] = src[i];
        }
    }
}

void memset_floats_neon(float* dst, int len)
{
    float32x4_t vZero = vdupq_n_f32(0.f);
    int len_floor = len - len % 16;
    int last_len = len % 16;
    for (int i = 0; i < len_floor; i += 16)
    {
        vst1q_f32(dst,      vZero);
        vst1q_f32(dst +  4, vZero);
        vst1q_f32(dst +  8, vZero);
        vst1q_f32(dst + 12, vZero);
        dst += 16;
    }
    int offset = 0;
    if (last_len >= 12)
    {
        vst1q_f32(dst,      vZero);
        vst1q_f32(dst +  4, vZero);
        vst1q_f32(dst +  8, vZero);
        offset = 12;
        dst += 12;
    }
    else if (last_len >= 8)
    {
        vst1q_f32(dst,      vZero);
        vst1q_f32(dst +  4, vZero);
        offset = 8;
        dst += 8;
    }
    else if (last_len >= 4)
    {
        vst1q_f32(dst,      vZero);
        offset = 4;
        dst += 4;
    }
    last_len -= offset;
    for (int i = 0; i < last_len; ++i)
    {
        dst[i] = 0.f;
    }
}

void inner_kernel_4x12(int K, float *packA, float *packB, float *c, int ldc)
{
    float *aptr = packA;
    float *bptr = packB;
    float *cptr = c;
    float32x4_t va;
    float32x4_t vb0, vb1, vb2;
    float32x4_t vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vcA, vcB;

    vc0 = vld1q_f32(cptr);
    vc1 = vld1q_f32(cptr + 4);
    vc2 = vld1q_f32(cptr + 8);
    cptr += ldc;

    vc3 = vld1q_f32(cptr);
    vc4 = vld1q_f32(cptr + 4);
    vc5 = vld1q_f32(cptr + 8);
    cptr += ldc;

    vc6 = vld1q_f32(cptr);
    vc7 = vld1q_f32(cptr + 4);
    vc8 = vld1q_f32(cptr + 8);
    cptr += ldc;

    vc9 = vld1q_f32(cptr);
    vcA = vld1q_f32(cptr + 4);
    vcB = vld1q_f32(cptr + 8);
    cptr += ldc;

    vb0 = vld1q_f32(bptr);
    vb1 = vld1q_f32(bptr + 4);
    vb2 = vld1q_f32(bptr + 8);

    for (int p = 0; p < (K - 1); ++p)
    {
        va = vld1q_dup_f32(aptr);
        vc0 = vfmaq_f32(vc0, vb0, va);
        vc1 = vfmaq_f32(vc1, vb1, va);
        vc2 = vfmaq_f32(vc2, vb2, va);
        va = vld1q_dup_f32(aptr + 1);
        vc3 = vfmaq_f32(vc3, vb0, va);
        vc4 = vfmaq_f32(vc4, vb1, va);
        vc5 = vfmaq_f32(vc5, vb2, va);

        va = vld1q_dup_f32(aptr + 2);

        vc6 = vfmaq_f32(vc6, vb0, va);
        vc7 = vfmaq_f32(vc7, vb1, va);
        vc8 = vfmaq_f32(vc8, vb2, va);

        va = vld1q_dup_f32(aptr + 3);
        vc9 = vfmaq_f32(vc9, vb0, va);
        vcA = vfmaq_f32(vcA, vb1, va);
        vcB = vfmaq_f32(vcB, vb2, va);

        bptr += 12;
        vb0 = vld1q_f32(bptr);
        vb1 = vld1q_f32(bptr + 4);
        vb2 = vld1q_f32(bptr + 8);

        aptr += 4;
    }
    cptr = c;
    va  = vld1q_dup_f32(aptr);
    vc0 = vfmaq_f32(vc0, vb0, va);
    vc1 = vfmaq_f32(vc1, vb1, va);
    vc2 = vfmaq_f32(vc2, vb2, va);
    vst1q_f32(cptr, vc0);
    vst1q_f32(cptr + 4, vc1);
    vst1q_f32(cptr + 8, vc2);
    cptr += ldc;

    va = vld1q_dup_f32(aptr + 1);
    vc3 = vfmaq_f32(vc3, vb0, va);
    vc4 = vfmaq_f32(vc4, vb1, va);
    vc5 = vfmaq_f32(vc5, vb2, va);
    vst1q_f32(cptr, vc3);
    vst1q_f32(cptr + 4, vc4);
    vst1q_f32(cptr + 8, vc5);
    cptr += ldc;

    va  = vld1q_dup_f32(aptr + 2);
    vc6 = vfmaq_f32(vc6, vb0, va);
    vc7 = vfmaq_f32(vc7, vb1, va);
    vc8 = vfmaq_f32(vc8, vb2, va);
    vst1q_f32(cptr, vc6);
    vst1q_f32(cptr + 4, vc7);
    vst1q_f32(cptr + 8, vc8);
    cptr += ldc;

    va = vld1q_dup_f32(aptr + 3);
    vc9 = vfmaq_f32(vc9, vb0, va);
    vcA = vfmaq_f32(vcA, vb1, va);
    vcB = vfmaq_f32(vcB, vb2, va);
    vst1q_f32(cptr, vc9);
    vst1q_f32(cptr + 4, vcA);
    vst1q_f32(cptr + 8, vcB);
    cptr += ldc;
}
void inner_kernel_8x12(int K, float *packA, float *packB, float *c, int ldc)
{
    float *aptr = packA;
    float *bptr = packB;
    float *cptr = c;
    float32x4_t va, va1;
    float32x4_t vb0, vb1, vb2;
    float32x4_t vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vcA, vcB;
    float32x4_t vd0, vd1, vd2, vd3, vd4, vd5, vd6, vd7, vd8, vd9, vdA, vdB;
    vc0 = vld1q_f32(cptr);
    vc1 = vld1q_f32(cptr + 4);
    vc2 = vld1q_f32(cptr + 8);
    cptr += ldc;

    vc3 = vld1q_f32(cptr);
    vc4 = vld1q_f32(cptr + 4);
    vc5 = vld1q_f32(cptr + 8);
    cptr += ldc;

    vc6 = vld1q_f32(cptr);
    vc7 = vld1q_f32(cptr + 4);
    vc8 = vld1q_f32(cptr + 8);
    cptr += ldc;

    vc9 = vld1q_f32(cptr);
    vcA = vld1q_f32(cptr + 4);
    vcB = vld1q_f32(cptr + 8);
    cptr += ldc;

    vd0 = vld1q_f32(cptr);
    vd1 = vld1q_f32(cptr + 4);
    vd2 = vld1q_f32(cptr + 8);
    cptr += ldc;

    vd3 = vld1q_f32(cptr);
    vd4 = vld1q_f32(cptr + 4);
    vd5 = vld1q_f32(cptr + 8);
    cptr += ldc;

    vd6 = vld1q_f32(cptr);
    vd7 = vld1q_f32(cptr + 4);
    vd8 = vld1q_f32(cptr + 8);
    cptr += ldc;

    vd9 = vld1q_f32(cptr);
    vdA = vld1q_f32(cptr + 4);
    vdB = vld1q_f32(cptr + 8);

    vb0 = vld1q_f32(bptr);
    vb1 = vld1q_f32(bptr + 4);
    vb2 = vld1q_f32(bptr + 8);

    for (int p = 0; p < (K - 1); ++p)
    {
        va = vld1q_dup_f32(aptr);
        va1 = vld1q_dup_f32(aptr + 1);
        vc0 = vfmaq_f32(vc0, vb0, va);
        vc1 = vfmaq_f32(vc1, vb1, va);
        vc2 = vfmaq_f32(vc2, vb2, va);
        vc3 = vfmaq_f32(vc3, vb0, va1);
        vc4 = vfmaq_f32(vc4, vb1, va1);
        vc5 = vfmaq_f32(vc5, vb2, va1);

        va = vld1q_dup_f32(aptr + 2);
        va1 = vld1q_dup_f32(aptr + 3);

        vc6 = vfmaq_f32(vc6, vb0, va);
        vc7 = vfmaq_f32(vc7, vb1, va);
        vc8 = vfmaq_f32(vc8, vb2, va);
        vc9 = vfmaq_f32(vc9, vb0, va1);
        vcA = vfmaq_f32(vcA, vb1, va1);
        vcB = vfmaq_f32(vcB, vb2, va1);

        va = vld1q_dup_f32(aptr + 4);
        va1 = vld1q_dup_f32(aptr + 5);
        vd0 = vfmaq_f32(vd0, vb0, va);
        vd1 = vfmaq_f32(vd1, vb1, va);
        vd2 = vfmaq_f32(vd2, vb2, va);
        vd3 = vfmaq_f32(vd3, vb0, va1);
        vd4 = vfmaq_f32(vd4, vb1, va1);
        vd5 = vfmaq_f32(vd5, vb2, va1);

        va = vld1q_dup_f32(aptr + 6);
        va1 = vld1q_dup_f32(aptr + 7);

        vd6 = vfmaq_f32(vd6, vb0, va);
        vd7 = vfmaq_f32(vd7, vb1, va);
        vd8 = vfmaq_f32(vd8, vb2, va);
        vd9 = vfmaq_f32(vd9, vb0, va1);
        vdA = vfmaq_f32(vdA, vb1, va1);
        vdB = vfmaq_f32(vdB, vb2, va1);

        bptr += 12;
        vb0 = vld1q_f32(bptr);
        vb1 = vld1q_f32(bptr + 4);
        vb2 = vld1q_f32(bptr + 8);

        aptr += 8;
    }
    cptr = c;
    va  = vld1q_dup_f32(aptr);
    va1 = vld1q_dup_f32(aptr + 1);
    vc0 = vfmaq_f32(vc0, vb0, va);
    vc1 = vfmaq_f32(vc1, vb1, va);
    vc2 = vfmaq_f32(vc2, vb2, va);
    vst1q_f32(cptr, vc0);
    vst1q_f32(cptr + 4, vc1);
    vst1q_f32(cptr + 8, vc2);
    cptr += ldc;

    vc3 = vfmaq_f32(vc3, vb0, va1);
    vc4 = vfmaq_f32(vc4, vb1, va1);
    vc5 = vfmaq_f32(vc5, vb2, va1);
    vst1q_f32(cptr, vc3);
    vst1q_f32(cptr + 4, vc4);
    vst1q_f32(cptr + 8, vc5);
    cptr += ldc;

    va  = vld1q_dup_f32(aptr + 2);
    va1 = vld1q_dup_f32(aptr + 3);
    vc6 = vfmaq_f32(vc6, vb0, va);
    vc7 = vfmaq_f32(vc7, vb1, va);
    vc8 = vfmaq_f32(vc8, vb2, va);
    vst1q_f32(cptr, vc6);
    vst1q_f32(cptr + 4, vc7);
    vst1q_f32(cptr + 8, vc8);
    cptr += ldc;

    vc9 = vfmaq_f32(vc9, vb0, va1);
    vcA = vfmaq_f32(vcA, vb1, va1);
    vcB = vfmaq_f32(vcB, vb2, va1);
    vst1q_f32(cptr, vc9);
    vst1q_f32(cptr + 4, vcA);
    vst1q_f32(cptr + 8, vcB);
    cptr += ldc;

    va  = vld1q_dup_f32(aptr + 4);
    va1 = vld1q_dup_f32(aptr + 5);
    vd0 = vfmaq_f32(vd0, vb0, va);
    vd1 = vfmaq_f32(vd1, vb1, va);
    vd2 = vfmaq_f32(vd2, vb2, va);
    vst1q_f32(cptr, vd0);
    vst1q_f32(cptr + 4, vd1);
    vst1q_f32(cptr + 8, vd2);
    cptr += ldc;

    vd3 = vfmaq_f32(vd3, vb0, va1);
    vd4 = vfmaq_f32(vd4, vb1, va1);
    vd5 = vfmaq_f32(vd5, vb2, va1);
    vst1q_f32(cptr, vd3);
    vst1q_f32(cptr + 4, vd4);
    vst1q_f32(cptr + 8, vd5);
    cptr += ldc;

    va  = vld1q_dup_f32(aptr + 6);
    va1 = vld1q_dup_f32(aptr + 7);
    vd6 = vfmaq_f32(vd6, vb0, va);
    vd7 = vfmaq_f32(vd7, vb1, va);
    vd8 = vfmaq_f32(vd8, vb2, va);
    vst1q_f32(cptr, vd6);
    vst1q_f32(cptr + 4, vd7);
    vst1q_f32(cptr + 8, vd8);
    cptr += ldc;

    vd9 = vfmaq_f32(vd9, vb0, va1);
    vdA = vfmaq_f32(vdA, vb1, va1);
    vdB = vfmaq_f32(vdB, vb2, va1);
    vst1q_f32(cptr, vd9);
    vst1q_f32(cptr + 4, vdA);
    vst1q_f32(cptr + 8, vdB);
}


template<int N>
void inner_kernel_Nx12_template(int K, float *packA, float *packB, float *c, int ldc)
{
    float *aptr = packA;
    float *bptr = packB;
    float *cptr = c;
    float32x4_t va, va1;
    float32x4_t vb0, vb1, vb2;
    float32x4_t vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vcA, vcB;
    float32x4_t vd0, vd1, vd2, vd3, vd4, vd5, vd6, vd7, vd8, vd9, vdA, vdB;
    {
        vc0 = vld1q_f32(cptr);
        vc1 = vld1q_f32(cptr + 4);
        vc2 = vld1q_f32(cptr + 8);
    }
    if (N > 1)
    {
        cptr += ldc;
        vc3 = vld1q_f32(cptr);
        vc4 = vld1q_f32(cptr + 4);
        vc5 = vld1q_f32(cptr + 8);
    }
    if (N > 2)
    {
        cptr += ldc;
        vc6 = vld1q_f32(cptr);
        vc7 = vld1q_f32(cptr + 4);
        vc8 = vld1q_f32(cptr + 8);
    }
    if (N > 3)
    {
        cptr += ldc;
        vc9 = vld1q_f32(cptr);
        vcA = vld1q_f32(cptr + 4);
        vcB = vld1q_f32(cptr + 8);
    }
    if (N > 4)
    {
        vd0 = vld1q_f32(cptr);
        vd1 = vld1q_f32(cptr + 4);
        vd2 = vld1q_f32(cptr + 8);
    }
    if (N > 5)
    {
        cptr += ldc;
        vd3 = vld1q_f32(cptr);
        vd4 = vld1q_f32(cptr + 4);
        vd5 = vld1q_f32(cptr + 8);
    }
    if (N > 6)
    {
        cptr += ldc;
        vd6 = vld1q_f32(cptr);
        vd7 = vld1q_f32(cptr + 4);
        vd8 = vld1q_f32(cptr + 8);
    }
    if (N > 7)
    {
        cptr += ldc;
        vd9 = vld1q_f32(cptr);
        vdA = vld1q_f32(cptr + 4);
        vdB = vld1q_f32(cptr + 8);
    }
    vb0 = vld1q_f32(bptr);
    vb1 = vld1q_f32(bptr + 4);
    vb2 = vld1q_f32(bptr + 8);

    for (int p = 0; p < (K - 1); ++p)
    {
        {
            va  = vld1q_dup_f32(aptr);
            vc0 = vfmaq_f32(vc0, vb0, va);
            vc1 = vfmaq_f32(vc1, vb1, va);
            vc2 = vfmaq_f32(vc2, vb2, va);
        }
        if (N > 1)
        {
            va1 = vld1q_dup_f32(aptr + 1);
            vc3 = vfmaq_f32(vc3, vb0, va1);
            vc4 = vfmaq_f32(vc4, vb1, va1);
            vc5 = vfmaq_f32(vc5, vb2, va1);
        }
        if (N > 2)
        {
            va  = vld1q_dup_f32(aptr + 2);
            vc6 = vfmaq_f32(vc6, vb0, va);
            vc7 = vfmaq_f32(vc7, vb1, va);
            vc8 = vfmaq_f32(vc8, vb2, va);
        }
        if (N > 3)
        {
            va1 = vld1q_dup_f32(aptr + 3);
            vc9 = vfmaq_f32(vc9, vb0, va1);
            vcA = vfmaq_f32(vcA, vb1, va1);
            vcB = vfmaq_f32(vcB, vb2, va1);

        }
        if (N > 4)
        {
            va  = vld1q_dup_f32(aptr + 4);
            vd0 = vfmaq_f32(vd0, vb0, va);
            vd1 = vfmaq_f32(vd1, vb1, va);
            vd2 = vfmaq_f32(vd2, vb2, va);
        }
        if (N > 5)
        {
            va1 = vld1q_dup_f32(aptr + 5);
            vd3 = vfmaq_f32(vd3, vb0, va1);
            vd4 = vfmaq_f32(vd4, vb1, va1);
            vd5 = vfmaq_f32(vd5, vb2, va1);
        }
        if (N > 6)
        {
            va  = vld1q_dup_f32(aptr + 6);
            vd6 = vfmaq_f32(vd6, vb0, va);
            vd7 = vfmaq_f32(vd7, vb1, va);
            vd8 = vfmaq_f32(vd8, vb2, va);
        }
        if (N > 7)
        {
            va1 = vld1q_dup_f32(aptr + 7);
            vd9 = vfmaq_f32(vd9, vb0, va1);
            vdA = vfmaq_f32(vdA, vb1, va1);
            vdB = vfmaq_f32(vdB, vb2, va1);
        }

        bptr += 12;
        vb0 = vld1q_f32(bptr);
        vb1 = vld1q_f32(bptr + 4);
        vb2 = vld1q_f32(bptr + 8);

        aptr += N;
    }

    cptr = c;
    {
        va = vld1q_dup_f32(aptr);
        vc0 = vfmaq_f32(vc0, vb0, va);
        vc1 = vfmaq_f32(vc1, vb1, va);
        vc2 = vfmaq_f32(vc2, vb2, va);
        vst1q_f32(cptr, vc0);
        vst1q_f32(cptr + 4, vc1);
        vst1q_f32(cptr + 8, vc2);
    }
    if (N > 1)
    {
        cptr += ldc;
        va = vld1q_dup_f32(aptr + 1);
        vc3 = vfmaq_f32(vc3, vb0, va);
        vc4 = vfmaq_f32(vc4, vb1, va);
        vc5 = vfmaq_f32(vc5, vb2, va);
        vst1q_f32(cptr, vc3);
        vst1q_f32(cptr + 4, vc4);
        vst1q_f32(cptr + 8, vc5);
    }
    if (N > 2)
    {
        cptr += ldc;
        va = vld1q_dup_f32(aptr + 2);
        vc6 = vfmaq_f32(vc6, vb0, va);
        vc7 = vfmaq_f32(vc7, vb1, va);
        vc8 = vfmaq_f32(vc8, vb2, va);
        vst1q_f32(cptr, vc6);
        vst1q_f32(cptr + 4, vc7);
        vst1q_f32(cptr + 8, vc8);
    }
    if (N > 3)
    {
        cptr += ldc;
        va = vld1q_dup_f32(aptr + 3);
        vc9 = vfmaq_f32(vc9, vb0, va);
        vcA = vfmaq_f32(vcA, vb1, va);
        vcB = vfmaq_f32(vcB, vb2, va);
        vst1q_f32(cptr, vc9);
        vst1q_f32(cptr + 4, vcA);
        vst1q_f32(cptr + 8, vcB);
    }
    if (N > 4)
    {
        va = vld1q_dup_f32(aptr + 4);
        vd0 = vfmaq_f32(vd0, vb0, va);
        vd1 = vfmaq_f32(vd1, vb1, va);
        vd2 = vfmaq_f32(vd2, vb2, va);
        vst1q_f32(cptr, vd0);
        vst1q_f32(cptr + 4, vd1);
        vst1q_f32(cptr + 8, vd2);
    }
    if (N > 5)
    {
        cptr += ldc;
        va = vld1q_dup_f32(aptr + 5);
        vd3 = vfmaq_f32(vd3, vb0, va);
        vd4 = vfmaq_f32(vd4, vb1, va);
        vd5 = vfmaq_f32(vd5, vb2, va);
        vst1q_f32(cptr, vd3);
        vst1q_f32(cptr + 4, vd4);
        vst1q_f32(cptr + 8, vd5);
    }
    if (N > 6)
    {
        cptr += ldc;
        va = vld1q_dup_f32(aptr + 6);
        vd6 = vfmaq_f32(vd6, vb0, va);
        vd7 = vfmaq_f32(vd7, vb1, va);
        vd8 = vfmaq_f32(vd8, vb2, va);
        vst1q_f32(cptr, vd6);
        vst1q_f32(cptr + 4, vd7);
        vst1q_f32(cptr + 8, vd8);
    }
    if (N > 7)
    {
        cptr += ldc;
        va = vld1q_dup_f32(aptr + 7);
        vd9 = vfmaq_f32(vd9, vb0, va);
        vdA = vfmaq_f32(vdA, vb1, va);
        vdB = vfmaq_f32(vdB, vb2, va);
        vst1q_f32(cptr, vd9);
        vst1q_f32(cptr + 4, vdA);
        vst1q_f32(cptr + 8, vdB);
    }
}

void inner_kernel_8x8(int K, float *packA, float *packB, float *c, int ldc)
{
    float *aptr = packA;
    float *bptr = packB;
    float *cptr = c;
    float32x4_t va, va1;
    float32x4_t vb0, vb1;
    float32x4_t vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vcA, vcB, vcC, vcD, vcE, vcF;

    vc0 = vld1q_f32(cptr);
    vc1 = vld1q_f32(cptr + 4);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc3 = vld1q_f32(cptr + 4);
    cptr += ldc;
    vc4 = vld1q_f32(cptr);
    vc5 = vld1q_f32(cptr + 4);
    cptr += ldc;
    vc6 = vld1q_f32(cptr);
    vc7 = vld1q_f32(cptr + 4);
    cptr += ldc;
    vc8 = vld1q_f32(cptr);
    vc9 = vld1q_f32(cptr + 4);
    cptr += ldc;
    vcA = vld1q_f32(cptr);
    vcB = vld1q_f32(cptr + 4);
    cptr += ldc;
    vcC = vld1q_f32(cptr);
    vcD = vld1q_f32(cptr + 4);
    cptr += ldc;
    vcE = vld1q_f32(cptr);
    vcF = vld1q_f32(cptr + 4);
    cptr += ldc;
    vb0 = vld1q_f32(bptr);
    vb1 = vld1q_f32(bptr + 4);
    for (int p = 0; p < (K - 1); ++p)
    {
        va = vld1q_f32(aptr);
        va1 = vld1q_f32(aptr + 4);
        vc0 = vfmaq_laneq_f32(vc0, vb0, va, 0);
        vc1 = vfmaq_laneq_f32(vc1, vb1, va, 0);
        vc2 = vfmaq_laneq_f32(vc2, vb0, va, 1);
        vc3 = vfmaq_laneq_f32(vc3, vb1, va, 1);
        vc4 = vfmaq_laneq_f32(vc4, vb0, va, 2);
        vc5 = vfmaq_laneq_f32(vc5, vb1, va, 2);
        vc6 = vfmaq_laneq_f32(vc6, vb0, va, 3);
        vc7 = vfmaq_laneq_f32(vc7, vb1, va, 3);
        vc8 = vfmaq_laneq_f32(vc8, vb0, va1, 0);
        vc9 = vfmaq_laneq_f32(vc9, vb1, va1, 0);
        vcA = vfmaq_laneq_f32(vcA, vb0, va1, 1);
        vcB = vfmaq_laneq_f32(vcB, vb1, va1, 1);
        vcC = vfmaq_laneq_f32(vcC, vb0, va1, 2);
        vcD = vfmaq_laneq_f32(vcD, vb1, va1, 2);
        vcE = vfmaq_laneq_f32(vcE, vb0, va1, 3);
        vcF = vfmaq_laneq_f32(vcF, vb1, va1, 3);

        vb0 = vld1q_f32(bptr + 8);
        vb1 = vld1q_f32(bptr + 12);
        bptr += 8;
        aptr += 8;
    }

    cptr = c;
    va = vld1q_f32(aptr);
    va1 = vld1q_f32(aptr + 4);
    vc0 = vfmaq_laneq_f32(vc0, vb0, va, 0);
    vc1 = vfmaq_laneq_f32(vc1, vb1, va, 0);
    vst1q_f32(cptr, vc0);
    vst1q_f32(cptr + 4, vc1);
    vc2 = vfmaq_laneq_f32(vc2, vb0, va, 1);
    vc3 = vfmaq_laneq_f32(vc3, vb1, va, 1);
    cptr += ldc;
    vst1q_f32(cptr, vc2);
    vst1q_f32(cptr + 4, vc3);
    vc4 = vfmaq_laneq_f32(vc4, vb0, va, 2);
    vc5 = vfmaq_laneq_f32(vc5, vb1, va, 2);
    cptr += ldc;
    vst1q_f32(cptr, vc4);
    vst1q_f32(cptr + 4, vc5);
    vc6 = vfmaq_laneq_f32(vc6, vb0, va, 3);
    vc7 = vfmaq_laneq_f32(vc7, vb1, va, 3);
    cptr += ldc;
    vst1q_f32(cptr, vc6);
    vst1q_f32(cptr + 4, vc7);
    vc8 = vfmaq_laneq_f32(vc8, vb0, va1, 0);
    vc9 = vfmaq_laneq_f32(vc9, vb1, va1, 0);
    cptr += ldc;
    vst1q_f32(cptr, vc8);
    vst1q_f32(cptr + 4, vc9);
    vcA = vfmaq_laneq_f32(vcA, vb0, va1, 1);
    vcB = vfmaq_laneq_f32(vcB, vb1, va1, 1);
    cptr += ldc;
    vst1q_f32(cptr, vcA);
    vst1q_f32(cptr + 4, vcB);
    vcC = vfmaq_laneq_f32(vcC, vb0, va1, 2);
    vcD = vfmaq_laneq_f32(vcD, vb1, va1, 2);
    cptr += ldc;
    vst1q_f32(cptr, vcC);
    vst1q_f32(cptr + 4, vcD);
    vcE = vfmaq_laneq_f32(vcE, vb0, va1, 3);
    vcF = vfmaq_laneq_f32(vcF, vb1, va1, 3);
    cptr += ldc;
    vst1q_f32(cptr, vcE);
    vst1q_f32(cptr + 4, vcF);
}

template<int N>
void inner_kernel_Nx8_template(int K, float *packA, float *packB, float *c, int ldc)
{
    float *aptr = packA;
    float *bptr = packB;
    float *cptr = c;
    float32x4_t va, va1;
    float32x4_t vb0, vb1;
    float32x4_t vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vcA, vcB, vcC, vcD, vcE, vcF;

    vc0 = vld1q_f32(cptr);
    vc1 = vld1q_f32(cptr + 4);
    cptr += ldc;
    if (N > 1)
    {
        vc2 = vld1q_f32(cptr);
        vc3 = vld1q_f32(cptr + 4);
        cptr += ldc;
    }
    if (N > 2)
    {
        vc4 = vld1q_f32(cptr);
        vc5 = vld1q_f32(cptr + 4);
        cptr += ldc;
    }
    if (N > 3)
    {
        vc6 = vld1q_f32(cptr);
        vc7 = vld1q_f32(cptr + 4);
        cptr += ldc;
    }
    if (N > 4)
    {
        vc8 = vld1q_f32(cptr);
        vc9 = vld1q_f32(cptr + 4);
        cptr += ldc;
    }
    if (N > 5)
    {
        vcA = vld1q_f32(cptr);
        vcB = vld1q_f32(cptr + 4);
        cptr += ldc;
    }
    if (N > 6)
    {
        vcC = vld1q_f32(cptr);
        vcD = vld1q_f32(cptr + 4);
        cptr += ldc;
    }
    if (N > 7)
    {
        vcE = vld1q_f32(cptr);
        vcF = vld1q_f32(cptr + 4);
        cptr += ldc;
    }
    vb0 = vld1q_f32(bptr);
    vb1 = vld1q_f32(bptr + 4);
    for (int p = 0; p < (K - 1); ++p)
    {
        va = vdupq_n_f32(aptr[0]);
        vc0 = vfmaq_f32(vc0, vb0, va);
        vc1 = vfmaq_f32(vc1, vb1, va);
        if (N > 1)
        {
            va = vdupq_n_f32(aptr[1]);
            vc2 = vfmaq_f32(vc2, vb0, va);
            vc3 = vfmaq_f32(vc3, vb1, va);
        }
        if (N > 2)
        {
            va = vdupq_n_f32(aptr[2]);
            vc4 = vfmaq_f32(vc4, vb0, va);
            vc5 = vfmaq_f32(vc5, vb1, va);
        }
        if (N > 3)
        {
            va = vdupq_n_f32(aptr[3]);
            vc6 = vfmaq_f32(vc6, vb0, va);
            vc7 = vfmaq_f32(vc7, vb1, va);
        }
        if (N > 4)
        {
            va = vdupq_n_f32(aptr[4]);
            vc8 = vfmaq_f32(vc8, vb0, va);
            vc9 = vfmaq_f32(vc9, vb1, va);
        }
        if (N > 5)
        {
            va = vdupq_n_f32(aptr[5]);
            vcA = vfmaq_f32(vcA, vb0, va);
            vcB = vfmaq_f32(vcB, vb1, va);
        }
        if (N > 6)
        {
            va = vdupq_n_f32(aptr[6]);
            vcC = vfmaq_f32(vcC, vb0, va);
            vcD = vfmaq_f32(vcD, vb1, va);
        }
        if (N > 7)
        {
            va = vdupq_n_f32(aptr[7]);
            vcE = vfmaq_f32(vcE, vb0, va);
            vcF = vfmaq_f32(vcF, vb1, va);
        }

        vb0 = vld1q_f32(bptr + 8);
        vb1 = vld1q_f32(bptr + 12);
        //tool::print_floats(bptr + 8, 8);
        bptr += 8;
        aptr += N;
    }

    cptr = c;
    if (N > 0)
    {
        va = vdupq_n_f32(aptr[0]);
        vc0 = vfmaq_f32(vc0, vb0, va);
        vc1 = vfmaq_f32(vc1, vb1, va);
        vst1q_f32(cptr, vc0);
        vst1q_f32(cptr + 4, vc1);
    }
    if (N > 1)
    {
        va = vdupq_n_f32(aptr[1]);
        vc2 = vfmaq_f32(vc2, vb0, va);
        vc3 = vfmaq_f32(vc3, vb1, va);
        cptr += ldc;
        vst1q_f32(cptr, vc2);
        vst1q_f32(cptr + 4, vc3);
    }
    if (N > 2)
    {
        va = vdupq_n_f32(aptr[2]);
        vc4 = vfmaq_f32(vc4, vb0, va);
        vc5 = vfmaq_f32(vc5, vb1, va);
        cptr += ldc;
        vst1q_f32(cptr, vc4);
        vst1q_f32(cptr + 4, vc5);
    }
    if (N > 3)
    {
        va = vdupq_n_f32(aptr[3]);
        vc6 = vfmaq_f32(vc6, vb0, va);
        vc7 = vfmaq_f32(vc7, vb1, va);
        cptr += ldc;
        vst1q_f32(cptr, vc6);
        vst1q_f32(cptr + 4, vc7);
    }
    if (N > 4)
    {
        va = vdupq_n_f32(aptr[4]);
        vc8 = vfmaq_f32(vc8, vb0, va);
        vc9 = vfmaq_f32(vc9, vb1, va);
        cptr += ldc;
        vst1q_f32(cptr, vc8);
        vst1q_f32(cptr + 4, vc9);
    }
    if (N > 5)
    {
        va = vdupq_n_f32(aptr[5]);
        vcA = vfmaq_f32(vcA, vb0, va);
        vcB = vfmaq_f32(vcB, vb1, va);
        cptr += ldc;
        vst1q_f32(cptr, vcA);
        vst1q_f32(cptr + 4, vcB);
    }
    if (N > 6)
    {
        va = vdupq_n_f32(aptr[6]);
        vcC = vfmaq_f32(vcC, vb0, va);
        vcD = vfmaq_f32(vcD, vb1, va);
        cptr += ldc;
        vst1q_f32(cptr, vcC);
        vst1q_f32(cptr + 4, vcD);
    }
    if (N > 7)
    {
        va = vdupq_n_f32(aptr[7]);
        vcE = vfmaq_f32(vcE, vb0, va);
        vcF = vfmaq_f32(vcF, vb1, va);
        cptr += ldc;
        vst1q_f32(cptr, vcE);
        vst1q_f32(cptr + 4, vcF);
    }
}


InnerKernel get_kernel_Nx8(int k)
{
    switch (k)
    {
        case 1:
            return inner_kernel_Nx8_template<1>;
        case 2:
            return inner_kernel_Nx8_template<2>;
        case 3:
            return inner_kernel_Nx8_template<3>;
        case 4:
            return inner_kernel_Nx8_template<4>;
        case 5:
            return inner_kernel_Nx8_template<5>;
        case 6:
            return inner_kernel_Nx8_template<6>;
        case 7:
            return inner_kernel_Nx8_template<7>;
        default:
            return inner_kernel_Nx8_template<8>;
    }
}

InnerKernel get_kernel_Nx12(int k)
{
    switch (k)
    {
        case 1:
            return inner_kernel_Nx12_template<1>;
        case 2:
            return inner_kernel_Nx12_template<2>;
        case 3:
            return inner_kernel_Nx12_template<3>;
        default:
            return inner_kernel_Nx12_template<4>;
            // case 4:
            //  return inner_kernel_Nx12_template<4>;
            // case 5:
            //  return inner_kernel_Nx12_template<5>;
            // case 6:
            //  return inner_kernel_Nx12_template<6>;
            // case 7:
            //  return inner_kernel_Nx12_template<7>;
            // default:
            //  return inner_kernel_Nx12_template<8>;
    }
}

template<bool fuseBias, bool fuseRelu>
inline void activate(int rows, float* C, int ldc, float* bias)
{
    float32x4_t vZero = vdupq_n_f32(0.f);
    for (int i = 0; i < rows; ++i)
    {
        float32x4_t vBias = vld1q_dup_f32(bias + i);
        float32x4_t vec1, vec2, vec3;
        vec1 = vld1q_f32(C + i * ldc);
        vec2 = vld1q_f32(C + i * ldc + 4);
        vec3 = vld1q_f32(C + i * ldc + 8);
        if (fuseBias)
        {
            vec1 = vaddq_f32(vec1, vBias);
            vec2 = vaddq_f32(vec2, vBias);
            vec3 = vaddq_f32(vec3, vBias);
        }
        if (fuseRelu)
        {
            vec1 = vmaxq_f32(vec1, vZero);
            vec2 = vmaxq_f32(vec2, vZero);
            vec3 = vmaxq_f32(vec3, vZero);
        }
        vst1q_f32(C + i * ldc, vec1);
        vst1q_f32(C + i * ldc + 4, vec2);
        vst1q_f32(C + i * ldc + 8, vec3);
    }
}

template<bool fuseBias, bool fuseRelu>
inline void compute_block_activation_reduce_ldst(int M, int nc, int kc, float* packA, float* packB, float *C, int ldc, float* bias, int bias_len, InnerKernel inner_kernel_local)
{
    const int COL_BATCH = 12;
    const int ROW_BATCH = 4;
    float C_buf[COL_BATCH * ROW_BATCH];

    const int nc_ceil = align_ceil(nc, COL_BATCH);
    const int n_len = nc % COL_BATCH;
    const int nc_floor_col = nc - nc % COL_BATCH;

    for (int i = 0; i < M - M % ROW_BATCH; i += ROW_BATCH)
    {
        float* rC = C + i * ldc;
        float* pA = packA + i * kc;
        for (int j = 0; j < nc_floor_col; j += COL_BATCH)
        {
            float* pC = rC + j;
            float* pB = packB + j * kc;
            inner_kernel_4x12(kc, pA, pB, pC, ldc);
            if (fuseBias || fuseRelu)
                activate<fuseBias, fuseRelu>(ROW_BATCH, pC, ldc, bias + i);
        }
        if (n_len)
        {
            int j = nc_floor_col;
            float* pC = rC + j;
            float* pB = packB + j * kc;
            float* pL = C_buf;
            for (int m = 0; m < ROW_BATCH; ++m)
            {
                for (int n = 0; n < n_len; ++n)
                {
                    pL[n] = pC[n];
                }
                pL += COL_BATCH;
                pC += ldc;
            }
            inner_kernel_4x12(kc, pA, pB, C_buf, COL_BATCH);
            if (fuseBias || fuseRelu)
                activate<fuseBias, fuseRelu>(ROW_BATCH, C_buf, COL_BATCH, bias + i);
            pC = rC + j;
            pL = C_buf;
            for (int m = 0; m < ROW_BATCH; ++m)
            {
                for (int n = 0; n < n_len; ++n)
                {
                    pC[n] = pL[n];
                }
                pL += COL_BATCH;
                pC += ldc;
            }
        }
    }
    int m_len = M % ROW_BATCH;
    if (m_len)
    {
        int i = M - M % ROW_BATCH;
        float* rC = C + i * ldc;
        float* pA = packA + i * kc;
        for (int j = 0; j < nc_floor_col; j += COL_BATCH)
        {
            float* pC = rC + j;
            float* pB = packB + j * kc;
            // printf("pC offset %d");
            inner_kernel_local(kc, pA, pB, pC, ldc);
            if (fuseBias || fuseRelu)
                activate<fuseBias, fuseRelu>(m_len, pC, ldc, bias + i);
        }
        if (n_len)
        {
            int j = nc_floor_col;
            float* pC = rC + j;
            float* pB = packB + j * kc;
            float* pL = C_buf;
            for (int m = 0; m < m_len; ++m)
            {
                for (int n = 0; n < n_len; ++n)
                {
                    pL[n] = pC[n];
                }
                pL += COL_BATCH;
                pC += ldc;
            }
            inner_kernel_local(kc, pA, pB, C_buf, COL_BATCH);
            if (fuseBias || fuseRelu)
                activate<fuseBias, fuseRelu>(m_len, C_buf, COL_BATCH, bias + i);
            pC = rC + j;
            pL = C_buf;
            for (int m = 0; m < m_len; ++m)
            {
                for (int n = 0; n < n_len; ++n)
                {
                    pC[n] = pL[n];
                }
                pL += COL_BATCH;
                pC += ldc;
            }
        }
    }
}

//Decide how many rows should be packed together.
template<int ROW_BATCH>
void packed_sgeconv_init(ConvParam *conv_param, int kc, float* packA, float* A)
{
    int M = conv_param->output_channels;
    int K = conv_param->kernel_h * conv_param->kernel_w * conv_param->input_channels;
    int lda = K;
    //int M_align = align_ceil(M, 6);
    for (int p = 0; p < K; p += kc)
    {

        //The last row batch may not have sufficient rows
        //Implicit padding so as to reduce code complexity for packed_sgemm
        //float* pPack = packA + (p / kc) * M_align * kc;
        float* pPack = packA + (p / kc) * M * kc;
        for (int i = 0; i < M; i += ROW_BATCH)
        {
            int k_len = kc;
            int j_len = ROW_BATCH;
            if (M - i < ROW_BATCH)
            {
                j_len = M - i;
            }
            float* pA = A + i * lda + p;
            if (K - p < kc)
                k_len = K - p;
            //Every ROW_BATCH rows are batched together.
            for (int k = 0; k < k_len; ++k)
            {
                for (int j = 0; j < j_len; ++j)
                {
                    pPack[j] = pA[j * lda];
                }
                pPack += j_len;
                pA++;
            }
        }
    }
}

// template void packed_sgeconv_init<6>(int M, int K, int kc, float* packedA, float* A, int lda);
template void packed_sgeconv_init<4>(ConvParam *conv_param, int kc, float* packA, float* A);

void pack_B_neon(int kc, int nc, float* packB, float* B, int ldb)
{
    const int COL_BATCH = 12;
    int nc_floor = nc - nc % COL_BATCH;
    for (int k = 0; k < kc; ++k)
    {
        float* pB = B + k * ldb;
        for (int j = 0; j < nc_floor; j += COL_BATCH)
        {
            float* pPack = packB + (j / COL_BATCH) * kc * COL_BATCH + k * COL_BATCH;
            //_mm256_store_ps(pPack, _mm256_loadu_ps(pB));
            //_mm256_store_ps(pPack + 8, _mm256_loadu_ps(pB + 8));
            vst1q_f32(pPack, vld1q_f32(pB));
            vst1q_f32(pPack + 4, vld1q_f32(pB + 4));
            vst1q_f32(pPack + 8, vld1q_f32(pB + 8));
            pB += COL_BATCH;
        }
        if (nc_floor < nc)
        {
            int j = nc_floor;
            int n_len = nc - nc_floor;
            float* pPack = packB + (j / COL_BATCH) * kc * COL_BATCH + k * COL_BATCH;
            for (int i = 0; i < n_len; ++i)
            {
                pPack[i] = pB[i];
            }
        }
    }
}

//In this version, we don't consider paddings.
template<int STRIDE>
void pack_B_im2col_neon(ConvParam *conv_param, int kc, int nc, int nt, float* packB, float *B, int ldb)
{
    // printf("pack param, kc %d nc %d nt %d\n", kc, nc, nt);
    const int COL_BATCH = 12;
    // const int STRIDE = conv_param->stride_w;
    int nc_floor = nc - nc % COL_BATCH;
    int kernel_elem_size = conv_param->kernel_w * conv_param->kernel_h;
    int num_pixels = conv_param->input_h * conv_param->input_w;
    // const __m256i v_idx = _mm256_set_epi32(14,12,10,8,6,4,2,0);
    //const __m256i v_idx = _mm256_set_epi32(7,6,5,4,3,2,1,0);

    int output_height = conv_param->output_h;
    int output_width = conv_param->output_w;

    int kernel_width = conv_param->kernel_w;
    int kernel_height = conv_param->kernel_h;

    for (int k = 0; k < kc; ++k)
    {
        float *pChannel = B + k * num_pixels; //Channel is k.
        int row_idx = k;
        for (int ki = 0; ki < kernel_elem_size; ++ki)
        {
            int kh = ki / kernel_width;
            int kw = ki % kernel_width;
            // for (int kw = 0; kw < kernel_width; ++kw)
            // {
            for (int j = 0; j < nc_floor; j += COL_BATCH) //nc is the #(output pixels) to be computed
            {
                // int kernel_elem_id = kh * conv_param->kernel_w + kw;

                float *pPack = packB + (j / COL_BATCH) * kc * COL_BATCH * kernel_elem_size + (row_idx * kernel_elem_size + ki) * COL_BATCH;
                // float *pPack = packB + (j / COL_BATCH) * kc * COL_BATCH * kernel_elem_size + (row_idx * kernel_elem_size + kernel_elem_id) * COL_BATCH;
                // printf("pPack offset %d\n", pPack - packB);
                int conv_row_id = (j + nt) / output_width; //conv_row_id should consider kernel elem id.
                int conv_col_id = (j + nt) % output_width;
                int img_col_idx = conv_col_id * conv_param->stride_w + kw;
                int img_row_idx = conv_row_id * conv_param->stride_h + kh;

                float* pB = pChannel + img_row_idx * conv_param->input_w + img_col_idx;
                if (output_width - conv_col_id >= COL_BATCH * STRIDE)
                {
                    // printf("16x output coord (%d, %d) img corrd (%d, %d)\n", conv_row_id, conv_col_id, img_row_idx, img_col_idx);
                    if (STRIDE == 1)
                    {
                        // _mm256_store_ps(pPack,     _mm256_loadu_ps(pChannel + img_row_idx * conv_param->input_w + img_col_idx));
                        // _mm256_store_ps(pPack + 8, _mm256_loadu_ps(pChannel + img_row_idx * conv_param->input_w + img_col_idx + 8));
                        // vst3q_f32(pPack, vld3q_f32(pChannel + img_row_idx * conv_param->input_w + img_col_idx));
                        float32x4_t vec1, vec2, vec3;
                        vec1 = vld1q_f32(pB);
                        vec2 = vld1q_f32(pB + 4);
                        vec3 = vld1q_f32(pB + 8);
                        vst1q_f32(pPack, vec1);
                        vst1q_f32(pPack + 4, vec2);
                        vst1q_f32(pPack + 8, vec3);
                    }
                    if (STRIDE == 2)
                    {
                        //_mm256_store_ps(pPack,     _mm256_i32gather_ps(pChannel + img_row_idx * conv_param->input_w + img_col_idx,     v_idx, 8));
                        //_mm256_store_ps(pPack + 8, _mm256_i32gather_ps(pChannel + img_row_idx * conv_param->input_w + img_col_idx + 16, v_idx, 8));
                        float32x4x2_t v_ld1, v_ld2, v_ld3;
                        v_ld1 = vld2q_f32(pB);
                        v_ld2 = vld2q_f32(pB + 8);
                        v_ld3 = vld2q_f32(pB + 16);
                        vst1q_f32(pPack,     v_ld1.val[0]);
                        vst1q_f32(pPack + 4, v_ld2.val[0]);
                        vst1q_f32(pPack + 8, v_ld3.val[0]);
                    }
                    if (STRIDE == 4)
                    {
                        //_mm256_store_ps(pPack,     _mm256_i32gather_ps(pChannel + img_row_idx * conv_param->input_w + img_col_idx,     v_idx, 8));
                        //_mm256_store_ps(pPack + 8, _mm256_i32gather_ps(pChannel + img_row_idx * conv_param->input_w + img_col_idx + 16, v_idx, 8));
                        float32x4x4_t v_ld1, v_ld2, v_ld3;
                        v_ld1 = vld4q_f32(pB);
                        v_ld2 = vld4q_f32(pB + 16);
                        v_ld3 = vld4q_f32(pB + 32);
                        vst1q_f32(pPack,     v_ld1.val[0]);
                        vst1q_f32(pPack + 4, v_ld2.val[0]);
                        vst1q_f32(pPack + 8, v_ld3.val[0]);
                    }
                }
                else
                {
                    int offset = 0;
                    if (output_width - conv_col_id >= 8 * STRIDE)
                    {
                        if (STRIDE == 1)
                            //_mm256_store_ps(pPack, _mm256_loadu_ps(pChannel + img_row_idx * conv_param->input_w + img_col_idx));
                            vst2q_f32(pPack, vld2q_f32(pChannel + img_row_idx * conv_param->input_w + img_col_idx));
                        if (STRIDE == 2)
                        {
                            float32x4x2_t v_ld1, v_ld2;
                            v_ld1 = vld2q_f32(pB);
                            v_ld2 = vld2q_f32(pB + 8);
                            vst1q_f32(pPack,     v_ld1.val[0]);
                            vst1q_f32(pPack + 4, v_ld2.val[0]);
                        }
                        if (STRIDE == 4)
                        {
                            float32x4x4_t v_ld1, v_ld2;
                            v_ld1 = vld4q_f32(pB);
                            v_ld2 = vld4q_f32(pB + 16);
                            vst1q_f32(pPack,     v_ld1.val[0]);
                            vst1q_f32(pPack + 4, v_ld2.val[0]);
                        }
                        //_mm256_store_ps(pPack, _mm256_i32gather_ps(pChannel + img_row_idx * conv_param->input_w + img_col_idx, v_idx, 8));
                        // printf("8x output coord (%d, %d) img corrd (%d, %d)\n", conv_row_id, conv_col_id, img_row_idx, img_col_idx);
                        // print_floats(pChannel + img_row_idx * conv_param->input_w + img_col_idx, 16);
                        // print_floats(pPack + img_row_idx * conv_param->input_w + img_col_idx, 8);
                        offset = 8;
                    }
                    else if (output_width - conv_col_id >= 4 * STRIDE)
                    {
                        if (STRIDE == 1)
                            //_mm256_store_ps(pPack, _mm256_loadu_ps(pChannel + img_row_idx * conv_param->input_w + img_col_idx));
                            vst1q_f32(pPack, vld1q_f32(pB));
                        if (STRIDE == 2)
                        {
                            float32x4x2_t v_ld1, v_ld2;
                            v_ld1 = vld2q_f32(pB);
                            vst1q_f32(pPack,     v_ld1.val[0]);
                        }
                        offset = 4;
                    }
                    // printf("offset %d\n", offset);
                    for (int i = offset; i < COL_BATCH; ++i)
                    {

                        conv_row_id = (j + i + nt) / output_width; //conv_row_id should consider kernel elem id.
                        conv_col_id = (j + i + nt) % output_width;
                        img_col_idx = conv_col_id * conv_param->stride_w + kw;
                        img_row_idx = conv_row_id * conv_param->stride_h + kh;
                        float *pB = pChannel + img_row_idx * conv_param->input_w + img_col_idx;
                        pPack[i] = pB[0];
                    }
                }
                if (nc_floor < nc)
                {
                    // int kernel_elem_id = kh * conv_param->kernel_w + kw;
                    int j = nc_floor;
                    int n_len = nc - nc_floor;
                    int col_idx = j + kw;
                    float *pPack = packB + (j / COL_BATCH) * kc * COL_BATCH * kernel_elem_size + (row_idx * kernel_elem_size + ki) * COL_BATCH;
                    for (int i = 0; i < n_len; ++i)
                    {
                        int conv_row_id = (j + i + nt) / output_width; //conv_row_id should consider kernel elem id.
                        int conv_col_id = (j + i + nt) % output_width; //Col id is always valid.
                        int img_col_idx = conv_col_id * conv_param->stride_w + kw;
                        int img_row_idx = conv_row_id * conv_param->stride_h + kh;
                        // printf("rem output coord (%d, %d) img corrd (%d, %d)\n", conv_row_id, conv_col_id, img_row_idx, img_col_idx);
                        float *pB = pChannel + img_row_idx * conv_param->input_w + img_col_idx;
                        pPack[i] = pB[0];
                        // printf("pB %5.3f\n", pB[0]);
                    }
                }
            }
        }
    }
}

void pack_B_im2col_neon_3x3s2(ConvParam *conv_param, int kc, int nc, int nt, float* packB, float *B, int ldb)
{
    const int COL_BATCH = 12;
    int nc_floor = nc - nc % COL_BATCH;
    int kernel_elem_size = conv_param->kernel_w * conv_param->kernel_h;
    int num_pixels = conv_param->input_h * conv_param->input_w;

    int output_height = conv_param->output_h;
    int output_width = conv_param->output_w;

    int kernel_width = conv_param->kernel_w;
    int kernel_height = conv_param->kernel_h;

    for (int k = 0; k < kc; ++k)
    {
        float *pChannel = B + k * num_pixels; //Channel is k.
        int row_idx = k;
        for (int j = 0; j < nc_floor; j += COL_BATCH) //nc is the #(output pixels) to be computed
        {
            for (int kh = 0; kh < conv_param->kernel_h; ++kh)
            {
                float *pPack = packB + (j / COL_BATCH) * kc * COL_BATCH * kernel_elem_size + (row_idx * kernel_elem_size + kh * 3) * COL_BATCH;
                int conv_row_id = (j + nt) / output_width; //conv_row_id should consider kernel elem id.
                int conv_col_id = (j + nt) % output_width;
                int img_col_idx = conv_col_id * conv_param->stride_w;
                int img_row_idx = conv_row_id * conv_param->stride_h + kh;
                float *pB = pChannel + img_row_idx * conv_param->input_w + img_col_idx;
                int pack_len = output_width - conv_col_id;
                if (pack_len >= COL_BATCH && conv_param->pad_left == 0)
                {
                    float32x4x2_t v_ld1, v_ld2, v_ld3;
                    v_ld1 = vld2q_f32(pB);
                    v_ld2 = vld2q_f32(pB + 8);
                    v_ld3 = vld2q_f32(pB + 16);
                    //kw = 0
                    vst1q_f32(pPack,     v_ld1.val[0]); // 0  2  4  6
                    vst1q_f32(pPack + 4, v_ld2.val[0]); // 8 10 12 14
                    vst1q_f32(pPack + 8, v_ld3.val[0]); //16 18 20 22
                    //kw = 1
                    vst1q_f32(pPack + 12, v_ld1.val[1]); // 1  3  5  7
                    v_ld1 = vld2q_f32(pB + 2);
                    vst1q_f32(pPack + 16, v_ld2.val[1]); // 9 11 13 15
                    v_ld2 = vld2q_f32(pB + 10);
                    vst1q_f32(pPack + 20, v_ld3.val[1]); //17 19 21 23
                    v_ld3 = vld2q_f32(pB + 18);
                    //kw = 2
                    vst1q_f32(pPack + 24, v_ld1.val[0]); // 2  4  6  8
                    vst1q_f32(pPack + 28, v_ld2.val[0]); //10 12 14 16
                    vst1q_f32(pPack + 32, v_ld3.val[0]); //18 20 22 24
                }
                else
                {
                    if (pack_len == 4 || pack_len == 8)
                    {
                        int conv_row_id = (j + nt + pack_len) / output_width; //conv_row_id should consider kernel elem id.
                        int conv_col_id = (j + nt + pack_len) % output_width;
                        int img_col_idx = conv_col_id * conv_param->stride_w;
                        int img_row_idx = conv_row_id * conv_param->stride_h + kh;
                        float *pB_next = pChannel + img_row_idx * conv_param->input_w + img_col_idx;
                        float32x4x2_t v_ld1, v_ld2, v_ld3;
                        if (pack_len == 8)
                        {
                            v_ld1 = vld2q_f32(pB);
                            v_ld2 = vld2q_f32(pB + 8);
                            v_ld3 = vld2q_f32(pB_next);
                        }
                        else
                        {
                            v_ld1 = vld2q_f32(pB);
                            v_ld2 = vld2q_f32(pB_next);
                            v_ld3 = vld2q_f32(pB_next + 8);
                        }
                        //kw = 0
                        vst1q_f32(pPack, v_ld1.val[0]);     // 0  2  4  6
                        vst1q_f32(pPack + 4, v_ld2.val[0]); // 8 10 12 14
                        vst1q_f32(pPack + 8, v_ld3.val[0]); //16 18 20 22
                        //kw = 1
                        vst1q_f32(pPack + 12, v_ld1.val[1]); // 1  3  5  7
                        vst1q_f32(pPack + 16, v_ld2.val[1]); // 9 11 13 15
                        vst1q_f32(pPack + 20, v_ld3.val[1]); //17 19 21 23
                        //kw = 2
                        if (pack_len == 8)
                        {
                            v_ld1 = vld2q_f32(pB + 2);
                            v_ld2 = vld2q_f32(pB + 10);
                            v_ld3 = vld2q_f32(pB_next + 2);
                        }
                        else
                        {
                            v_ld1 = vld2q_f32(pB + 2);
                            v_ld2 = vld2q_f32(pB_next + 2);
                            v_ld3 = vld2q_f32(pB_next + 10);
                        }
                        vst1q_f32(pPack + 24, v_ld1.val[0]); // 2  4  6  8
                        vst1q_f32(pPack + 28, v_ld2.val[0]); //10 12 14 16
                        vst1q_f32(pPack + 32, v_ld3.val[0]); //18 20 22 24
                    }
                    else
                    {
                        for (int kw = 0; kw < conv_param->kernel_w; ++kw)
                        {
                            for (int i = 0; i < COL_BATCH; ++i)
                            {
                                conv_row_id = (j + i + nt) / output_width; //conv_row_id should consider kernel elem id.
                                conv_col_id = (j + i + nt) % output_width;

                                img_col_idx = conv_col_id * conv_param->stride_w + kw;
                                img_row_idx = conv_row_id * conv_param->stride_h + kh;
                                pPack[kw * COL_BATCH + i] = *(pChannel + img_row_idx * conv_param->input_w + img_col_idx);
                            }
                        }
                    }
                }
                if (nc_floor < nc)
                {
                    int j = nc_floor;
                    int n_len = nc - nc_floor;

                    float *pPack = packB + (j / COL_BATCH) * kc * COL_BATCH * kernel_elem_size + (row_idx * kernel_elem_size + kh * 3) * COL_BATCH;
                    for (int kw = 0; kw < conv_param->kernel_w; ++kw)
                    {
                        for (int i = 0; i < n_len; ++i)
                        {
                            int conv_row_id = (j + i + nt) / output_width; //conv_row_id should consider kernel elem id.
                            int conv_col_id = (j + i + nt) % output_width; //Col id is always valid.
                            int img_row_idx = conv_row_id * conv_param->stride_h + kh;
                            int img_col_idx = conv_col_id * conv_param->stride_w + kw;
                            float *pB = pChannel + img_row_idx * conv_param->input_w + img_col_idx;
                            pPack[i] = pB[0];
                        }
                        pPack += COL_BATCH;
                    }
                }
            }
        }
    }
}

void pack_B_im2col_neon_5x5s4p2(ConvParam *conv_param, int kc, int nc, int nt, float* packB, float *B, int ldb)
{
    // conv_param->debug();
    const int STRIDE = 4;
    const int kernel_width = 5;
    const int kernel_height = 5;
    const int kernel_elem_size = 25;
    const int COL_BATCH = 12;
    int nc_floor = nc - nc % COL_BATCH;
    int right_border_idx = 5 + (conv_param->output_w - 1) * 4 - 1 - conv_param->pad_left;
    printf("right border idx %d\n", right_border_idx);

    int num_pixels = conv_param->input_h * conv_param->input_w;
    int output_h = conv_param->output_h;
    int output_w = conv_param->output_w;
    float32x4_t vZero = vdupq_n_f32(0.f);
    for (int j = 0; j < nc_floor; j += COL_BATCH) //nc is the #(output pixels in each image) to be computed
    {
        for (int k = 0; k < kc; ++k)
        {
            float *pChannel = B + k * num_pixels; //Channel is k.
            int row_idx = k;
            for (int kh = 0; kh < conv_param->kernel_h; ++kh)
            {
                float *pPack = packB + (j / COL_BATCH) * kc * COL_BATCH * kernel_elem_size + (row_idx * kernel_elem_size + kh * 5) * COL_BATCH;
                int conv_row_id = (j + nt) / output_w; //conv_row_id should consider kernel elem id.
                int conv_col_id = (j + nt) % output_w;
                int img_col_idx = conv_col_id * conv_param->stride_w - conv_param->pad_left;
                int img_row_idx = conv_row_id * conv_param->stride_h + kh - conv_param->pad_top;
                int img_col_right = conv_param->input_w - img_col_idx - COL_BATCH * conv_param->stride_w;
                float *pB = pChannel + img_row_idx * conv_param->input_w + img_col_idx;
                int pack_len = output_w - conv_col_id;
                // if(img_col_right <= 0 && pack_len >= COL_BATCH)
                //  printf("img right %d\n", img_col_right);
                if (pack_len >= COL_BATCH)
                {
                    if (img_row_idx < 0 || img_row_idx >= conv_param->input_h)
                    {
                        vst1q_f32(pPack, vZero);     // 0  4  8 12, conv1
                        vst1q_f32(pPack + 4, vZero); //16 20 24 28, conv2
                        vst1q_f32(pPack + 8, vZero); //32 36 40 44, conv3
                        //kw = 1
                        vst1q_f32(pPack + 12, vZero); // 1  5  9 13, conv1
                        vst1q_f32(pPack + 16, vZero); //17 21 25 29, conv2
                        vst1q_f32(pPack + 20, vZero); //33 37 41 45, conv3
                        //kw = 2
                        vst1q_f32(pPack + 24, vZero); // 2  6 10 14, conv1
                        vst1q_f32(pPack + 28, vZero); //18 22 26 30, conv2
                        vst1q_f32(pPack + 32, vZero); //34 38 42 46, conv3
                        //kw = 3
                        vst1q_f32(pPack + 36, vZero); // 3  7 11 15, conv1
                        vst1q_f32(pPack + 40, vZero); //19 23 27 31, conv2
                        vst1q_f32(pPack + 44, vZero); //35 39 43 47, conv3
                        //kw = 4
                        vst1q_f32(pPack + 48, vZero); // 4  8 12 16, conv1
                        vst1q_f32(pPack + 52, vZero); //20 24 28 32, conv2
                        vst1q_f32(pPack + 56, vZero); //36 40 44 48, conv3
                    }
                    else if (img_col_idx >= 0 && img_col_right >= 0 && img_col_idx <= conv_param->input_w)
                    {
                        float32x4x4_t vld1, vld2, vld3;
                        vld1 = vld4q_f32(pB);
                        vld2 = vld4q_f32(pB + 16);
                        vld3 = vld4q_f32(pB + 32);
                        //kw = 0
                        vst1q_f32(pPack, vld1.val[0]);   // 0  4  8 12, conv1
                        vst1q_f32(pPack + 4, vld2.val[0]); //16 20 24 28, conv2
                        vst1q_f32(pPack + 8, vld3.val[0]); //32 36 40 44, conv3
                        //kw = 1
                        vst1q_f32(pPack + 12, vld1.val[1]); // 1  5  9 13, conv1
                        vst1q_f32(pPack + 16, vld2.val[1]); //17 21 25 29, conv2
                        vst1q_f32(pPack + 20, vld3.val[1]); //33 37 41 45, conv3
                        //kw = 2
                        vst1q_f32(pPack + 24, vld1.val[2]); // 2  6 10 14, conv1
                        vst1q_f32(pPack + 28, vld2.val[2]); //18 22 26 30, conv2
                        vst1q_f32(pPack + 32, vld3.val[2]); //34 38 42 46, conv3
                        //kw = 3
                        vst1q_f32(pPack + 36, vld1.val[3]); // 3  7 11 15, conv1
                        vld1 = vld4q_f32(pB + 4);
                        vst1q_f32(pPack + 40, vld2.val[3]); //19 23 27 31, conv2
                        vld2 = vld4q_f32(pB + 20);
                        vst1q_f32(pPack + 44, vld3.val[3]); //35 39 43 47, conv3
                        vld3 = vld4q_f32(pB + 36);
                        //kw = 4
                        vst1q_f32(pPack + 48, vld1.val[0]); // 4  8 12 16, conv1
                        vst1q_f32(pPack + 52, vld2.val[0]); //20 24 28 32, conv2
                        vst1q_f32(pPack + 56, vld3.val[0]); //36 40 44 48, conv3
                    }
                    else if (img_col_idx == -2 && img_col_idx <= conv_param->input_w)
                    {
                        //pB has already been offset by paddings.
                        float32x4x4_t vld1, vld2, vld3;
                        vld1 = vld4q_f32(pB);
                        vld1.val[0] = vsetq_lane_f32(0.0f, vld1.val[0], 0);
                        vld1.val[1] = vsetq_lane_f32(0.0f, vld1.val[1], 0);
                        vld2 = vld4q_f32(pB + 16);
                        vld3 = vld4q_f32(pB + 32);
                        //kw = 0
                        vst1q_f32(pPack, vld1.val[0]);   // 0  4  8 12, conv1
                        vst1q_f32(pPack + 4, vld2.val[0]); //16 20 24 28, conv2
                        vst1q_f32(pPack + 8, vld3.val[0]); //32 36 40 44, conv3
                        //kw = 1
                        vst1q_f32(pPack + 12, vld1.val[1]); // 1  5  9 13, conv1
                        vst1q_f32(pPack + 16, vld2.val[1]); //17 21 25 29, conv2
                        vst1q_f32(pPack + 20, vld3.val[1]); //33 37 41 45, conv3
                        //kw = 2
                        vst1q_f32(pPack + 24, vld1.val[2]); // 2  6 10 14, conv1
                        vst1q_f32(pPack + 28, vld2.val[2]); //18 22 26 30, conv2
                        vst1q_f32(pPack + 32, vld3.val[2]); //34 38 42 46, conv3
                        //kw = 3
                        vst1q_f32(pPack + 36, vld1.val[3]); // 3  7 11 15, conv1
                        vld1 = vld4q_f32(pB + 4);
                        vst1q_f32(pPack + 40, vld2.val[3]); //19 23 27 31, conv2
                        vld2 = vld4q_f32(pB + 20);
                        vst1q_f32(pPack + 44, vld3.val[3]); //35 39 43 47, conv3
                        vld3 = vld4q_f32(pB + 36);
                        //kw = 4
                        vst1q_f32(pPack + 48, vld1.val[0]); // 4  8 12 16, conv1
                        vst1q_f32(pPack + 52, vld2.val[0]); //20 24 28 32, conv2
                        vst1q_f32(pPack + 56, vld3.val[0]); //36 40 44 48, conv3
                    }
                }
                else
                {
#if 1
                    //The two vectorized case, 4 + 8 and 8 + 4
                    if (pack_len == 8 || pack_len == 4)
                    {
                        // printf("pack_len %d\n", pack_len);
                        // float *pB_next = pB = pChannel + (img_row_idx + 1) * conv_param->input_w;//This is totally wrong
                        int conv_row_id = (j + nt + pack_len) / output_w; //conv_row_id should consider kernel elem id.
                        int conv_col_id = (j + nt + pack_len) % output_w;
                        int img_col_idx = conv_col_id * conv_param->stride_w - conv_param->pad_left;
                        int img_row_idx = conv_row_id * conv_param->stride_h + kh - conv_param->pad_top;
                        float *pB_next = pChannel + img_row_idx * conv_param->input_w + img_col_idx;
                        float32x4x4_t vld1, vld2, vld3;
                        if (pack_len == 8)
                        {
                            vld1 = vld4q_f32(pB);
                            vld2 = vld4q_f32(pB + 16);
                            vld3 = vld4q_f32(pB_next);
                            vld3.val[0] = vsetq_lane_f32(0.f, vld3.val[0], 0);
                            vld3.val[1] = vsetq_lane_f32(0.f, vld3.val[1], 0);
                        }
                        else
                        {
                            vld1 = vld4q_f32(pB);
                            vld2 = vld4q_f32(pB_next);
                            // vld1.val[0] = vsetq_lane_f32(0.f, vld2.val[0], 3);
                            vld3 = vld4q_f32(pB_next + 16);
                            // vld1.val[1] = vsetq_lane_f32(0.f, vld2.val[1], 3);
                        }
                        //kw = 0
                        vst1q_f32(pPack,      vld1.val[0]); // 0  4  8 12, conv1
                        vst1q_f32(pPack + 4,  vld2.val[0]); //16 20 24 28, conv2
                        vst1q_f32(pPack + 8,  vld3.val[0]); //32 36 40 44, conv3
                        //kw = 1
                        vst1q_f32(pPack + 12, vld1.val[1]); // 1  5  9 13, conv1
                        vst1q_f32(pPack + 16, vld2.val[1]); //17 21 25 29, conv2
                        vst1q_f32(pPack + 20, vld3.val[1]); //33 37 41 45, conv3
                        //kw = 2
                        vst1q_f32(pPack + 24, vld1.val[2]); // 2  6 10 14, conv1
                        vst1q_f32(pPack + 28, vld2.val[2]); //18 22 26 30, conv2
                        vst1q_f32(pPack + 32, vld3.val[2]); //34 38 42 46, conv3
                        //kw = 3
                        vst1q_f32(pPack + 36, vld1.val[3]); // 3  7 11 15, conv1
                        vst1q_f32(pPack + 40, vld2.val[3]); //19 23 27 31, conv2
                        vst1q_f32(pPack + 44, vld3.val[3]); //35 39 43 47, conv3
                        //kw = 4
                        if (pack_len == 8)
                        {
                            vld1 = vld4q_f32(pB + 4);
                            vld2 = vld4q_f32(pB + 20);
                            vld3 = vld4q_f32(pB_next + 4);
                            // vld2.val[0] = vsetq_lane_f32(0.f, vld2.val[0], 3);
                            // vld2.val[1] = vsetq_lane_f32(0.f, vld2.val[1], 3);
                        }
                        else
                        {
                            vld1 = vld4q_f32(pB + 4);
                            vld2 = vld4q_f32(pB_next + 4);
                            vld3 = vld4q_f32(pB_next + 20);
                            vld2.val[0] = vsetq_lane_f32(0.f, vld2.val[0], 0);
                            vld2.val[1] = vsetq_lane_f32(0.f, vld2.val[1], 0);
                        }
                        vst1q_f32(pPack + 48, vld1.val[0]); // 4  8 12 16, conv1
                        vst1q_f32(pPack + 52, vld2.val[0]); //20 24 28 32, conv2
                        vst1q_f32(pPack + 56, vld3.val[0]); //36 40 44 48, conv3
                    }
                    else
#endif
                    {
                        // printf("pack len %d id %d\n", pack_len, img_col_idx);
                        for (int kw = 0; kw < conv_param->kernel_w; ++kw)
                        {
                            for (int i = 0; i < COL_BATCH; ++i)
                            {
                                conv_row_id = (j + i + nt) / output_w; //conv_row_id should consider kernel elem id.
                                conv_col_id = (j + i + nt) % output_w;
                                img_col_idx = conv_col_id * conv_param->stride_w + kw - conv_param->pad_left;
                                img_row_idx = conv_row_id * conv_param->stride_h + kh - conv_param->pad_top;
                                // if(k == 0)
                                // {
                                //  printf("(%d, %d) (%d, %d)\n", conv_row_id, conv_col_id, img_row_idx, img_col_idx);
                                // }
                                if (img_col_idx < 0 || img_row_idx < 0 || img_col_idx >= conv_param->input_w || img_row_idx >= conv_param->input_h)
                                    pPack[i] = 0.f;
                                else
                                {
                                    pPack[i] = *(pChannel + img_row_idx * conv_param->input_w + img_col_idx);
                                }
                            }
                            pPack += COL_BATCH;
                        }
                    }
                }
                if (nc_floor < nc)
                {
                    printf("nc floor!\n");
                    int j = nc_floor;
                    int n_len = nc - nc_floor;
                    pPack = packB + (j / COL_BATCH) * kc * COL_BATCH * kernel_elem_size + (row_idx * kernel_elem_size + kh * 5) * COL_BATCH;
                    for (int kw = 0; kw < conv_param->kernel_w; ++kw)
                    {
                        int col_idx = j + kw;
                        for (int i = 0; i < n_len; ++i)
                        {
                            int conv_row_id = (j + i + nt) / output_w; //conv_row_id should consider kernel elem id.
                            int conv_col_id = (j + i + nt) % output_w; //Col id is always valid.
                            int img_col_idx = conv_col_id * conv_param->stride_w + kw;
                            int img_row_idx = conv_row_id * conv_param->stride_h + kh;
                            float *pB = pChannel + img_row_idx * conv_param->input_w + img_col_idx;
                            if (img_col_idx < 0 || img_col_idx < 0 || img_col_idx >= conv_param->input_w || img_row_idx >= conv_param->input_h)
                                pPack[i] = 0.f;
                            else
                                pPack[i] = pB[0];
                        }
                        pPack += COL_BATCH;
                    }
                }
            }
        }
    }
}


void pack_B_im2col_neon_5x5s4(ConvParam *conv_param, int kc, int nc, int nt, float* packB, float *B, int ldb)
{
    const int STRIDE = 4;
    const int kernel_width = 5;
    const int kernel_height = 5;
    const int kernel_elem_size = 25;
    const int COL_BATCH = 12;
    int nc_floor = nc - nc % COL_BATCH;

    int num_pixels = conv_param->input_h * conv_param->input_w;
    int output_height = conv_param->output_h;
    int output_width = conv_param->output_w;
    for (int j = 0; j < nc_floor; j += COL_BATCH) //nc is the #(output pixels in each image) to be computed
    {
        for (int k = 0; k < kc; ++k)
        {
            float *pChannel = B + k * num_pixels; //Channel is k.
            int row_idx = k;
            for (int kh = 0; kh < conv_param->kernel_h; ++kh)
            {
                float *pPack = packB + (j / COL_BATCH) * kc * COL_BATCH * kernel_elem_size + (row_idx * kernel_elem_size + kh * 5) * COL_BATCH;
                int conv_row_id = (j + nt) / output_width; //conv_row_id should consider kernel elem id.
                int conv_col_id = (j + nt) % output_width;
                int img_col_idx = conv_col_id * conv_param->stride_w;
                int img_row_idx = conv_row_id * conv_param->stride_h + kh;
                float *pB = pChannel + img_row_idx * conv_param->input_w + img_col_idx;
                int pack_len = output_width - conv_col_id;
                if (pack_len >= COL_BATCH)
                {
                    float32x4x4_t vld1, vld2, vld3;
                    vld1 = vld4q_f32(pB);
                    vld2 = vld4q_f32(pB + 16);
                    vld3 = vld4q_f32(pB + 32);
                    //kw = 0
                    vst1q_f32(pPack,     vld1.val[0]);// 0  4  8 12, conv1
                    vst1q_f32(pPack + 4, vld2.val[0]);//16 20 24 28, conv2
                    vst1q_f32(pPack + 8, vld3.val[0]);//32 36 40 44, conv3
                    //kw = 1
                    vst1q_f32(pPack + 12, vld1.val[1]);// 1  5  9 13, conv1
                    vst1q_f32(pPack + 16, vld2.val[1]);//17 21 25 29, conv2
                    vst1q_f32(pPack + 20, vld3.val[1]);//33 37 41 45, conv3
                    //kw = 2
                    vst1q_f32(pPack + 24, vld1.val[2]);// 2  6 10 14, conv1
                    vst1q_f32(pPack + 28, vld2.val[2]);//18 22 26 30, conv2
                    vst1q_f32(pPack + 32, vld3.val[2]);//34 38 42 46, conv3
                    //kw = 3
                    vst1q_f32(pPack + 36, vld1.val[3]);// 3  7 11 15, conv1
                    vld1 = vld4q_f32(pB + 4);
                    vst1q_f32(pPack + 40, vld2.val[3]);//19 23 27 31, conv2
                    vld2 = vld4q_f32(pB + 20);
                    vst1q_f32(pPack + 44, vld3.val[3]);//35 39 43 47, conv3
                    vld3 = vld4q_f32(pB + 36);
                    //kw = 4
                    vst1q_f32(pPack + 48, vld1.val[0]);// 4  8 12 16, conv1
                    vst1q_f32(pPack + 52, vld2.val[0]);//20 24 28 32, conv2
                    vst1q_f32(pPack + 56, vld3.val[0]);//36 40 44 48, conv3
                }
                else
                {
                    //The two vectorized case, 4 + 8 and 8 + 4
                    if (pack_len == 8 || pack_len == 4)
                    {
                        // printf("pack_len %d\n", pack_len);
                        // float *pB_next = pB = pChannel + (img_row_idx + 1) * conv_param->input_w;//This is totally wrong
                        int conv_row_id = (j + nt + pack_len) / output_width; //conv_row_id should consider kernel elem id.
                        int conv_col_id = (j + nt + pack_len) % output_width;
                        int img_col_idx = conv_col_id * conv_param->stride_w;
                        int img_row_idx = conv_row_id * conv_param->stride_h + kh;
                        float *pB_next = pChannel + img_row_idx * conv_param->input_w + img_col_idx;
                        float32x4x4_t vld1, vld2, vld3;
                        if (pack_len == 8)
                        {
                            vld1 = vld4q_f32(pB);
                            vld2 = vld4q_f32(pB + 16);
                            vld3 = vld4q_f32(pB_next);
                        }
                        else
                        {
                            vld1 = vld4q_f32(pB);
                            vld2 = vld4q_f32(pB_next);
                            vld3 = vld4q_f32(pB_next + 16);
                        }
                        //kw = 0
                        vst1q_f32(pPack,      vld1.val[0]); // 0  4  8 12, conv1
                        vst1q_f32(pPack + 4,  vld2.val[0]); //16 20 24 28, conv2
                        vst1q_f32(pPack + 8,  vld3.val[0]); //32 36 40 44, conv3
                        //kw = 1
                        vst1q_f32(pPack + 12, vld1.val[1]); // 1  5  9 13, conv1
                        vst1q_f32(pPack + 16, vld2.val[1]); //17 21 25 29, conv2
                        vst1q_f32(pPack + 20, vld3.val[1]); //33 37 41 45, conv3
                        //kw = 2
                        vst1q_f32(pPack + 24, vld1.val[2]); // 2  6 10 14, conv1
                        vst1q_f32(pPack + 28, vld2.val[2]); //18 22 26 30, conv2
                        vst1q_f32(pPack + 32, vld3.val[2]); //34 38 42 46, conv3
                        //kw = 3
                        vst1q_f32(pPack + 36, vld1.val[3]); // 3  7 11 15, conv1
                        vst1q_f32(pPack + 40, vld2.val[3]); //19 23 27 31, conv2
                        vst1q_f32(pPack + 44, vld3.val[3]); //35 39 43 47, conv3
                        //kw = 4
                        if (pack_len == 8)
                        {
                            vld1 = vld4q_f32(pB + 4);
                            vld2 = vld4q_f32(pB + 20);
                            vld3 = vld4q_f32(pB_next + 4);
                        }
                        else
                        {
                            vld1 = vld4q_f32(pB + 4);
                            vld2 = vld4q_f32(pB_next + 4);
                            vld3 = vld4q_f32(pB_next + 20);
                        }
                        vst1q_f32(pPack + 48, vld1.val[0]); // 4  8 12 16, conv1
                        vst1q_f32(pPack + 52, vld2.val[0]); //20 24 28 32, conv2
                        vst1q_f32(pPack + 56, vld3.val[0]); //36 40 44 48, conv3
                    }
                    else
                    {
                        for (int kw = 0; kw < conv_param->kernel_w; ++kw)
                        {
                            for (int i = 0; i < COL_BATCH; ++i)
                            {
                                int conv_row_id = (j + i + nt) / output_width; //conv_row_id should consider kernel elem id.
                                int conv_col_id = (j + i + nt) % output_width;
                                int img_col_idx = conv_col_id * conv_param->stride_w + kw;
                                int img_row_idx = conv_row_id * conv_param->stride_h + kh;
                                if (img_col_idx < 0 || img_col_idx < 0 || img_col_idx >= conv_param->input_w || img_row_idx >= conv_param->input_h)
                                    pPack[i] = 0.f;
                                else
                                    pPack[i] = *(pChannel + img_row_idx * conv_param->input_w + img_col_idx);
                            }
                            pPack += COL_BATCH;
                        }
                    }
                }
                if (nc_floor < nc)
                {
                    int j = nc_floor;
                    int n_len = nc - nc_floor;
                    pPack = packB + (j / COL_BATCH) * kc * COL_BATCH * kernel_elem_size + (row_idx * kernel_elem_size + kh * 5) * COL_BATCH;
                    for (int kw = 0; kw < conv_param->kernel_w; ++kw)
                    {
                        int col_idx = j + kw;
                        for (int i = 0; i < n_len; ++i)
                        {
                            int conv_row_id = (j + i + nt) / output_width; //conv_row_id should consider kernel elem id.
                            int conv_col_id = (j + i + nt) % output_width; //Col id is always valid.
                            int img_col_idx = conv_col_id * conv_param->stride_w + kw;
                            int img_row_idx = conv_row_id * conv_param->stride_h + kh;
                            float *pB = pChannel + img_row_idx * conv_param->input_w + img_col_idx;
                            if (img_col_idx < 0 || img_col_idx < 0 || img_col_idx >= conv_param->input_w || img_row_idx >= conv_param->input_h)
                                pPack[i] = 0.f;
                            else
                                pPack[i] = pB[0];
                        }
                        pPack += COL_BATCH;
                    }
                }
            }
        }
    }
}

void pack_B_im2col_scalar(ConvParam *conv_param, int kc, int nc, int nt, float* packB, float *B, int ldb)
{
    const int COL_BATCH = 12;
    int nc_floor = nc - nc % COL_BATCH;
    int output_height = conv_param->output_h;
    int output_width = conv_param->output_w;
    int kernel_height = conv_param->kernel_h;
    int kernel_width = conv_param->kernel_w;
    int kernel_elem_size = conv_param->kernel_w * conv_param->kernel_h;
    int num_pixels = conv_param->input_h * conv_param->input_w;
    int pack_stride = kc * kernel_elem_size * COL_BATCH;
    // int pack_stride = 1;
    // printf("pack stride %d\n", pack_stride);
    for (int k = 0; k < kc; ++k)
    {
        float *pChannel = B + k * num_pixels; //Channel is k.
        int row_idx = k;
        for (int ki = 0; ki < kernel_elem_size; ++ki)
        {
            int kh = ki / kernel_width;
            int kw = ki % kernel_width;
            float *pPack = packB + (row_idx * kernel_elem_size + ki) * COL_BATCH;
            for (int j = 0; j < nc_floor; j += COL_BATCH) //nc is the #(output pixels) to be computed
            {
                int conv_row_id = (j + nt) / output_width; //conv_row_id should consider kernel elem id.
                int conv_col_id = (j + nt) % output_width;
                int img_col_idx = conv_col_id * conv_param->stride_w + kw;
                int img_row_idx = conv_row_id * conv_param->stride_h + kh;

                for (int i = 0; i < COL_BATCH; ++i)
                {
                    conv_row_id = (j + i + nt) / output_width; //conv_row_id should consider kernel elem id.
                    conv_col_id = (j + i + nt) % output_width;
                    img_col_idx = conv_col_id * conv_param->stride_w + kw - conv_param->pad_left;
                    img_row_idx = conv_row_id * conv_param->stride_h + kh - conv_param->pad_top;

                    if (img_col_idx < 0 || img_row_idx < 0 || img_col_idx > conv_param->input_w || img_row_idx > conv_param->input_h)
                        pPack[i] = 0.f;
                    else
                    {
                        float *pB = pChannel + img_row_idx * conv_param->input_w + img_col_idx;
                        pPack[i] = *pB;
                    }
                }
                pPack += pack_stride;
            }
            if (nc_floor < nc)
            {
                int j = nc_floor;
                int n_len = nc - nc_floor;
                int col_idx = j + kw;
                for (int i = 0; i < n_len; ++i)
                {
                    int conv_row_id = (j + i + nt) / output_width; //conv_row_id should consider kernel elem id.
                    int conv_col_id = (j + i + nt) % output_width; //Col id is always valid.
                    int img_col_idx = conv_col_id * conv_param->stride_w + kw - conv_param->pad_left;
                    int img_row_idx = conv_row_id * conv_param->stride_h + kh - conv_param->pad_top;
                    if (img_col_idx < 0 || img_row_idx < 0 || img_col_idx > conv_param->input_w || img_row_idx > conv_param->input_h)
                        pPack[i] = 0.f;
                    else
                    {
                        float *pB = pChannel + img_row_idx * conv_param->input_w + img_col_idx;
                        pPack[i] = pB[0];
                    }
                }
            }

        }
    }
}

void pack_B_im2col_scalar_v2(ConvParam *conv_param, int kc, int nc, int nt, float* packB, float *B, int ldb)
{
    const int COL_BATCH = 12;
    int nc_floor = nc - nc % COL_BATCH;
    int output_height = conv_param->output_h;
    int output_width = conv_param->output_w;
    int kernel_height = conv_param->kernel_h;
    int kernel_width = conv_param->kernel_w;
    int kernel_elem_size = conv_param->kernel_w * conv_param->kernel_h;
    int num_pixels = conv_param->input_h * conv_param->input_w;
    for (int k = 0; k < kc; ++k)
    {
        float *pChannel = B + k * num_pixels; //Channel is k.
        int row_idx = k;
        for (int kh = 0; kh < kernel_height; ++kh)
        {
            float *pack_base = packB + (row_idx * kernel_elem_size + kh * conv_param->kernel_w) * COL_BATCH;
            for (int j = 0; j < nc_floor; j += COL_BATCH) //nc is the #(output pixels) to be computed
            {
                float *pPack = pack_base + j * kc * kernel_elem_size;
                for (int kw = 0; kw < kernel_width; ++kw)
                {
                    int conv_row_id = (j + nt) / output_width; //conv_row_id should consider kernel elem id.
                    int conv_col_id = (j + nt) % output_width;
                    int img_col_idx = conv_col_id * conv_param->stride_w + kw;
                    int img_row_idx = conv_row_id * conv_param->stride_h + kh;

                    for (int i = 0; i < COL_BATCH; ++i)
                    {
                        conv_row_id = (j + i + nt) / output_width; //conv_row_id should consider kernel elem id.
                        conv_col_id = (j + i + nt) % output_width;
                        img_col_idx = conv_col_id * conv_param->stride_w + kw - conv_param->pad_left;
                        img_row_idx = conv_row_id * conv_param->stride_h + kh - conv_param->pad_top;

                        if (img_col_idx < 0 || img_row_idx < 0 || img_col_idx > conv_param->input_w || img_row_idx > conv_param->input_h)
                            pPack[i] = 0.f;
                        else
                        {
                            float *pB = pChannel + img_row_idx * conv_param->input_w + img_col_idx;
                            pPack[i] = *pB;
                        }
                    }
                    pPack += COL_BATCH;
                }
            }
            if (nc_floor < nc)
            {
                int j = nc_floor;
                int n_len = nc - nc_floor;
                int col_idx = j;
                float *pPack = pack_base + j * kc * kernel_elem_size;
                for (int kw = 0; kw < kernel_width; ++kw)
                {
                    float *pPack = pack_base + kw * COL_BATCH + j * kc * kernel_elem_size;
                    for (int i = 0; i < n_len; ++i)
                    {
                        int conv_row_id = (j + i + nt) / output_width; //conv_row_id should consider kernel elem id.
                        int conv_col_id = (j + i + nt) % output_width; //Col id is always valid.
                        int img_col_idx = conv_col_id * conv_param->stride_w + kw - conv_param->pad_left;
                        int img_row_idx = conv_row_id * conv_param->stride_h + kh - conv_param->pad_top;
                        if (img_col_idx < 0 || img_row_idx < 0 || img_col_idx > conv_param->input_w || img_row_idx > conv_param->input_h)
                            pPack[i] = 0.f;
                        else
                        {
                            float *pB = pChannel + img_row_idx * conv_param->input_w + img_col_idx;
                            pPack[i] = pB[0];
                        }
                    }
                    pPack += COL_BATCH;
                }
            }
        }
    }
}
void pad_input_neon(ConvParam *conv_param, float* padded_input, float* input);

#include <stdlib.h>
template<bool fuseBias, bool fuseRelu>
void packed_sgeconv_im2col_activation(ConvParam *conv_param, float *packA, float *B, int ldb, float *C, int ldc, int nc, int kc, float* bias, int num_threads, float* pack_array)
{
    const int M = conv_param->output_channels;
    const int N = conv_param->output_h * conv_param->output_w;//pixel num
    const int K = conv_param->input_channels;
    const int kernel_elem_size = conv_param->kernel_h * conv_param->kernel_w;
    // printf("M %d N %d K %d\n", M, N, K);
    const int ROW_BATCH = 4;
    const int COL_BATCH = 12;

    InnerKernel inner_kernel_local = get_kernel_Nx12(M % ROW_BATCH);

    for (int i = 0; i < M; ++i)
    {
        memset(C + ldc * i, 0, sizeof(float) * N);
    }

    int M_align = align_ceil(M, ROW_BATCH);
    int N_align = align_ceil(N, COL_BATCH);

    int NBlocks = (N_align + nc - 1) / nc;
    int KBlocks = (K + kc - 1) / kc;
    //Our GEMM is implemented in GEPB fashion, as the operands are row-major
    int k_len = kc;
    int n_len = nc;

    // __attribute__((aligned(32))) float packB[kc * nc * kernel_elem_size];
    float *packB = pack_array;
    Timer tmr;
    double time_acc = 0.f;
    //kt is always the channel index.
    for (int kt = 0; kt < KBlocks; ++kt)
    {
        for (int nt = 0; nt < NBlocks; ++nt)
        {
            float* pA = packA + kt * kc * M * kernel_elem_size;
            float* pC = C + nt * nc;
            n_len = (nt == NBlocks - 1) ? (N - nt * nc) : nc;
            k_len = (kt == KBlocks - 1) ? (K - kt * kc) : kc;
            // printf("kt %d klen %d nt %d nlen %d\n", kt, k_len, nt, n_len);
            // printf("kt %d nt %d pack offset %d\n", kt, nt, kc * kt * conv_param->input_w * conv_param->input_h);
            // tmr.startBench();
            if (conv_param->stride_w == 1)
                pack_B_im2col_neon<1>(conv_param, k_len, n_len, nt * nc, packB, B + kc * kt * conv_param->input_w * conv_param->input_h, N);
            else if (conv_param->stride_w == 2 && conv_param->kernel_h == 3 && conv_param->kernel_w == 3)
                pack_B_im2col_neon_3x3s2(conv_param, k_len, n_len, nt * nc, packB, B + kc * kt * conv_param->input_w * conv_param->input_h, N);
            else if (conv_param->stride_w == 2)
                pack_B_im2col_neon<2>(conv_param, k_len, n_len, nt * nc, packB, B + kc * kt * conv_param->input_w * conv_param->input_h, N);
            else if (conv_param->stride_w == 4 && conv_param->kernel_h == 5 && conv_param->kernel_w == 5 && (conv_param->pad_left == 2) && (conv_param->pad_bottom == 2) && (conv_param->pad_right == 2) && (conv_param->pad_top == 2))
                pack_B_im2col_neon_5x5s4p2(conv_param, k_len, n_len, nt * nc, packB, B + kc * kt * conv_param->input_w * conv_param->input_h, N);
            else if (conv_param->stride_w == 4 && conv_param->kernel_h == 5 && conv_param->kernel_w == 5)
                pack_B_im2col_neon_5x5s4(conv_param, k_len, n_len, nt * nc, packB, B + kc * kt * conv_param->input_w * conv_param->input_h, N);
            else
                pack_B_im2col_scalar(conv_param, k_len, n_len, nt * nc, packB, B + kc * kt * conv_param->input_w * conv_param->input_h, N);
            // pack_B_im2col_scalar_v2(conv_param, k_len, n_len, nt * nc, packB, B + kc * kt * conv_param->input_w * conv_param->input_h, N);
            // print_floats(packB, (n_len * k_len * 25 + 11)/ 12, 12);
            // time_acc += tmr.endBench("Packing costs: ");
            // time_acc += tmr.endBench("Packing");
            // time_acc += tmr.endBench();

            // tmr.startBench();
            if (kt < KBlocks - 1)
            {
                compute_block_activation_reduce_ldst<false, false>(M, n_len, kc * conv_param->kernel_w * conv_param->kernel_h, pA, packB,  pC, ldc, bias, M, inner_kernel_local);
            }
            else
            {
                compute_block_activation_reduce_ldst<fuseBias, fuseRelu>(M, n_len, (K - kt * kc) * conv_param->kernel_w * conv_param->kernel_h, pA, packB, pC, ldc, bias, M, inner_kernel_local);
            }
            // tmr.endBench("Computing costs");
        }
    }
    // printf("packing spent %lf ms\n", time_acc);
}

template void packed_sgeconv_im2col_activation<false, false>(ConvParam *, float *, float *, int, float *, int, int, int, float*, int, float*);
template void packed_sgeconv_im2col_activation<false,  true>(ConvParam *, float *, float *, int, float *, int, int, int, float*, int, float*);
template void packed_sgeconv_im2col_activation<true,  false>(ConvParam *, float *, float *, int, float *, int, int, int, float*, int, float*);
template void packed_sgeconv_im2col_activation<true,   true>(ConvParam *, float *, float *, int, float *, int, int, int, float*, int, float*);

void pad_input_neon(ConvParam *conv_param, float* padded_input, float* input)
{
    float32x4_t vZero = vdupq_n_f32(0.f);
    const int channels = conv_param->input_channels;
    const int rows = conv_param->input_h;
    const int cols = conv_param->input_w;
    const int pdl = conv_param->pad_left;
    const int pdr = conv_param->pad_right;
    const int pdt = conv_param->pad_top;
    const int pdb = conv_param->pad_bottom;
    const int pad_cols = cols + pdl + pdr;
    const int pad_rows = rows + pdt + pdb;
    // const int pad_gap = conv_param->pad_left + conv_param->pad_right;
    // printf("channels %d", channels);
    // printf("rows %d cols %d\n", rows, cols);
    // printf("pad rows %d pad cols %d\n", pad_rows, pad_cols);
    float* pdst = padded_input;
    float* psrc = input;

    for (int c = 0; c < channels; ++c)
    {
        psrc = input + c * cols * rows;
        pdst = padded_input + c * pad_cols * pad_rows;
        for (int i = 0; i < conv_param->pad_top; ++i)
        {
            memset_floats_neon(pdst, pad_cols);
            // memset(pdst, 0, pad_cols * sizeof(float));
            pdst += pad_cols;
        }
        // pdst = padded_input + c * pad_cols * pad_rows + pad_cols;

        for (int i = 0; i < rows; ++i)
        {
            // printf("pdst %d psrc %d\n", pdst - padded_input, psrc - input);

            // printf("cols %d\n", cols);
            memcpy_floats_neon(pdst + pdl, psrc, cols);
            // memcpy(pdst + pdl, psrc, cols * sizeof(float));
            // memcpy(pdst + pdl, psrc, cols * sizeof(float));
            // memset(pdst + i * pad_cols, 0, sizeof(float) * pad_cols);
            pdst += pad_cols;
            psrc += cols;
        }
        for (int i = 0; i < conv_param->pad_bottom; ++i)
        {
            memset_floats_neon(pdst, pad_cols);
            // memset(pdst, 0, pad_cols * sizeof(float));
            pdst += pad_cols;
        }
    }
}
};