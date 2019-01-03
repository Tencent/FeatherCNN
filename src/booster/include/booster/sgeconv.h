#pragma once

#include <booster/booster.h>
/*
 * Performs single-float matrix multiply C = A * B in row-major fashion,
 * where C is MxN, A is MxK and B is KxN.
 * Allocation requirement: packA: M * K
 */

namespace booster
{
void pad_input_neon(booster::ConvParam *conv_param, float* padded_input, float* input);

template<int ROW_BATCH>
void packed_sgeconv_init(booster::ConvParam* conv_param, int kc, float* packA, float* A);

template<bool fuseBias, bool fuseRelu>
void packed_sgeconv_im2col_activation(booster::ConvParam* conv_param, float *packA, float *B, int ldb, float *C, int ldc, int nc, int kc, float* bias, int num_threads, float* pack_array);
};