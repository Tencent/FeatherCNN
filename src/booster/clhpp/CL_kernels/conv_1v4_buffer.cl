#include <common.h>

// N = 4, 8, or 16, which is the channel group size. 
__kernel void convolution(__global const DATA_TYPE* restrict in,     /* [ih, iw, ic] */
                          __global const DATA_TYPE* restrict weight, /* [oc/N, kh, kw, [ic, N, 1]] */
#ifdef BIAS
                          __global const DATA_TYPE* restrict bias,   /* [oc] */
#endif
                          __global DATA_TYPE* restrict out,          /* [oh, ow, oc] */
                          __private const int in_channels,           /* a multiple of N */
                          __private const int out_channels,          /* a multiple of N */
                          __private const int in_height,
                          __private const int in_width,
                          __private const int out_height,
                          __private const int out_width,
                          __private const int kernel_height,
                          __private const int kernel_width,
                          __private const int stride_height,
                          __private const int stride_width,
                          __private const int pad_top,
                          __private const int pad_left) {
  const int out_height_idx = get_global_id(0);
  const int out_width_idx = get_global_id(1) << 2;
  if (out_height_idx >= out_height || out_width_idx >= out_width) return;
  const int out_channel_idx = get_global_id(2) * N;

  const int in_height_beg = mad24(out_height_idx, stride_height, -pad_top);
  const int in_width_beg0 = mad24(out_width_idx, stride_width, -pad_left);
  const int in_width_beg1 = in_width_beg0 + stride_width;
  const int in_width_beg2 = in_width_beg1 + stride_width;
  const int in_width_beg3 = in_width_beg2 + stride_width;
  int kernel_val_idx = mul24(out_channel_idx, 
                             mul24(mul24(kernel_height, kernel_width), in_channels));

  DATA_TYPEN in_val0, in_val1, in_val2, in_val3, kernel_val;
#ifdef BIAS
  DATA_TYPEN out_val0 = VLOADN(0, &bias[out_channel_idx]);
  DATA_TYPEN out_val1 = out_val0;
  DATA_TYPEN out_val2 = out_val0;
  DATA_TYPEN out_val3 = out_val0;
#else
  DATA_TYPEN out_val0 = 0;
  DATA_TYPEN out_val1 = 0;
  DATA_TYPEN out_val2 = 0;
  DATA_TYPEN out_val3 = 0;
#endif
  for (int h = 0; h != kernel_height; ++h) {
    int in_height_idx = in_height_beg + h;
    if (in_height_idx < 0) in_height_idx = -in_height_idx;
    if (in_height_idx >= in_height) in_height_idx = in_height * 2 - 2 - in_height_idx;

    const int in_val_base_width_idx = mul24(mul24(in_height_idx, in_width), in_channels);
    for (int w = 0; w != kernel_width; ++w) {
      int in_width_idx0 = in_width_beg0 + w;
      if (in_width_idx0 < 0) in_width_idx0 = -in_width_idx0;
      if (in_width_idx0 >= in_width) in_width_idx0 = in_width * 2 - 2 - in_width_idx0;
      int in_width_idx1 = in_width_beg1 + w;
      if (in_width_idx1 < 0) in_width_idx1 = -in_width_idx1;
      if (in_width_idx1 >= in_width) in_width_idx1 = in_width * 2 - 2 - in_width_idx1;
      int in_width_idx2 = in_width_beg2 + w;
      if (in_width_idx2 < 0) in_width_idx2 = -in_width_idx2;
      if (in_width_idx2 >= in_width) in_width_idx2 = in_width * 2 - 2 - in_width_idx2;
      int in_width_idx3 = in_width_beg3 + w;
      if (in_width_idx3 < 0) in_width_idx3 = -in_width_idx3;
      if (in_width_idx3 >= in_width) in_width_idx3 = in_width * 2 - 2 - in_width_idx3;

      const int in_val_beg0 = mad24(in_width_idx0, in_channels, in_val_base_width_idx);
      const int in_val_beg1 = mad24(in_width_idx1, in_channels, in_val_base_width_idx);
      const int in_val_beg2 = mad24(in_width_idx2, in_channels, in_val_base_width_idx);
      const int in_val_beg3 = mad24(in_width_idx3, in_channels, in_val_base_width_idx);
      for (int c = 0; c < in_channels; c += N) {
        in_val0 = VLOADN(0, &in[in_val_beg0 + c]);
        in_val1 = VLOADN(0, &in[in_val_beg1 + c]);
        in_val2 = VLOADN(0, &in[in_val_beg2 + c]);
        in_val3 = VLOADN(0, &in[in_val_beg3 + c]);

#define LOAD_KERNEL_AND_CALC(i)                             \
        kernel_val = VLOADN(0, &weight[kernel_val_idx]);   \
        out_val0 = mad(in_val0.s##i, kernel_val, out_val0); \
        out_val1 = mad(in_val1.s##i, kernel_val, out_val1); \
        out_val2 = mad(in_val2.s##i, kernel_val, out_val2); \
        out_val3 = mad(in_val3.s##i, kernel_val, out_val3); \
        kernel_val_idx += N;

        LOAD_KERNEL_AND_CALC(0);
        LOAD_KERNEL_AND_CALC(1);
        LOAD_KERNEL_AND_CALC(2);
        LOAD_KERNEL_AND_CALC(3);
#if N == 8 || N == 16
        LOAD_KERNEL_AND_CALC(4);
        LOAD_KERNEL_AND_CALC(5);
        LOAD_KERNEL_AND_CALC(6);
        LOAD_KERNEL_AND_CALC(7);
#if N == 16
        LOAD_KERNEL_AND_CALC(8);
        LOAD_KERNEL_AND_CALC(9);
        LOAD_KERNEL_AND_CALC(a);
        LOAD_KERNEL_AND_CALC(b);
        LOAD_KERNEL_AND_CALC(c);
        LOAD_KERNEL_AND_CALC(d);
        LOAD_KERNEL_AND_CALC(e);
        LOAD_KERNEL_AND_CALC(f);
#endif
#endif

#undef LOAD_KERNEL_AND_CALC
      }
    }
  }

#if defined(USE_RELU)
  out_val0 = fmax(out_val0, (DATA_TYPE)0);
  out_val1 = fmax(out_val1, (DATA_TYPE)0);
  out_val2 = fmax(out_val2, (DATA_TYPE)0);
  out_val3 = fmax(out_val3, (DATA_TYPE)0);
#endif

  int out_val_idx = mad24(mad24(out_height_idx, out_width, out_width_idx), 
                          out_channels, 
                          out_channel_idx);
  VSTOREN(out_val0, 0, &out[out_val_idx]);
  out_val_idx += out_channels;
  if (out_width_idx + 1 < out_width) {
    VSTOREN(out_val1, 0, &out[out_val_idx]);
    out_val_idx += out_channels;
    if (out_width_idx + 2 < out_width) {
      VSTOREN(out_val2, 0, &out[out_val_idx]);
      out_val_idx += out_channels;
      if (out_width_idx + 3 < out_width) {
        VSTOREN(out_val3, 0, &out[out_val_idx]);
      }
    }
  }
}
