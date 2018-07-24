#pragma once

/*
 * Performs single-float matrix multiply C = A * B in row-major fashion,
 * where C is MxN, A is MxK and B is KxN.
 * Allocation requirement: packA: M * K
 */

struct ConvParam{
    int output_channels;
    int input_channels;
    int input_h;
    int input_w;
    int kernel_h;
    int kernel_w;
    int output_h;
    int output_w;
    int stride_h;
    int stride_w;
    int pad_left;
    int pad_bottom;
    int pad_right;
    int pad_top;
    void AssignOutputDim()
    {
        output_h = (input_h + pad_top + pad_bottom - kernel_h) / stride_h + 1;
        output_w = (input_w + pad_left + pad_right - kernel_w) / stride_w + 1;
    }
    void AssignPaddedDim()
    {
        input_h = input_h + pad_left + pad_right;
        input_w = input_w + pad_top + pad_bottom;
        pad_left = 0;
        pad_bottom = 0;
        pad_right = 0;
        pad_top = 0;
    }
};

namespace sgeconv_dev{
void pad_input_neon(ConvParam *conv_param, float* padded_input, float* input);

template<int ROW_BATCH>
void packed_sgeconv_init(ConvParam* conv_param, int kc, float* packA, float* A);

template<bool fuseBias, bool fuseRelu>
void packed_sgeconv_im2col_activation(ConvParam* conv_param, float *packA, float *B, int ldb, float *C, int ldc, int nc, int kc, float* bias, int num_threads, float* pack_array);
};