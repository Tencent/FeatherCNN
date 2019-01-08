#include <common.h>

// N = 4, 8, or 16, which is the channel block size. 
__kernel void conv(__global const DATA_TYPE* restrict in,     /* [ih, iw, ic] */
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
                   __private const int stride_width) {
  const int out_height_idx = get_global_id(0);
  const int out_width_idx = get_global_id(1) << 1;
  if (out_height_idx >= out_height || out_width_idx >= out_width) return;
  const int out_channel_block_idx = get_global_id(2);
  const int out_channel_idx = mul24(out_channel_block_idx, N);

  const int in_height_beg = mul24(out_height_idx, stride_height);
  const int in_height_end = in_height_beg + kernel_height;
  const int in_width_beg = mul24(out_width_idx, stride_width);
  const int in_width_end = in_width_beg + kernel_width;
  const int in_width_gap_size = mul24(in_width_beg + in_width - in_width_end, in_channels);
  const int in_width_stride = mul24(stride_width, in_channels);
  int in_val_idx = mul24(mad24(in_height_beg, in_width, in_width_beg), in_channels);

  const int kernel_size = mul24(mul24(kernel_height, kernel_width), in_channels);
  int kernel_val_idx = mul24(out_channel_idx, kernel_size);

  DATA_TYPEN in_val0, in_val1, kernel_val;
#ifdef BIAS
  DATA_TYPEN out_val0 = VLOADN(0, bias + out_channel_idx);
  DATA_TYPEN out_val1 = out_val0;
#else
  DATA_TYPEN out_val0 = 0;
  DATA_TYPEN out_val1 = 0;
#endif
  for (int in_height_idx = in_height_beg; in_height_idx != in_height_end; ++in_height_idx) {
    for (int in_width_idx = in_width_beg; in_width_idx != in_width_end; ++in_width_idx) {
      for (int in_channel_idx = 0; in_channel_idx != in_channels; in_channel_idx += N) {
        in_val0 = VLOADN(0, in + in_val_idx);
        in_val1 = VLOADN(0, in + in_val_idx + in_width_stride);
        in_val_idx += N;

#define LOAD_KERNEL_AND_CALC(i)                             \
        kernel_val = VLOADN(0, weight + kernel_val_idx);    \
        out_val0 = mad(in_val0.s##i, kernel_val, out_val0); \
        out_val1 = mad(in_val1.s##i, kernel_val, out_val1); \
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

    in_val_idx += in_width_gap_size;
  }

#if defined(USE_RELU)
  out_val0 = fmax(out_val0, (DATA_TYPE)0);
  out_val1 = fmax(out_val1, (DATA_TYPE)0);
#endif

  const int out_val_idx = mad24(mad24(out_height_idx, out_width, out_width_idx), 
                                out_channels, 
                                out_channel_idx);
  VSTOREN(out_val0, 0, out + out_val_idx);
  if (out_width_idx + 1 >= out_width) return;
  VSTOREN(out_val1, 0, out + out_val_idx + out_channels);
}

// N = 4, 8, or 16, which is the channel block size. 
__kernel void coalesced_conv(__global const DATA_TYPE* restrict in,     /* [ih, iw, ic] */
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
                             __private const int stride_width) {
  const int out_height_idx = get_global_id(0);
  const int out_width_idx = get_global_id(1);
  const int out_width_blocks = (out_width + 1) >> 1;
  if (out_height_idx >= out_height || out_width_idx >= out_width_blocks) return;
  const int out_channel_block_idx = get_global_id(2);
  const int out_channel_idx = mul24(out_channel_block_idx, N);

  const int in_height_beg = mul24(out_height_idx, stride_height);
  const int in_height_end = in_height_beg + kernel_height;
  const int in_width_beg = mul24(out_width_idx, stride_width);
  const int in_width_end = in_width_beg + kernel_width;
  const int in_width_gap_size = mul24(in_width_beg + in_width - in_width_end, in_channels);
  const int in_width_stride = mul24(mul24(stride_width, in_channels), out_width_blocks);
  int in_val_idx = mul24(mad24(in_height_beg, in_width, in_width_beg), in_channels);

  const int kernel_size = mul24(mul24(kernel_height, kernel_width), in_channels);
  int kernel_val_idx = mul24(out_channel_idx, kernel_size);

  DATA_TYPEN in_val0, in_val1, kernel_val;
#ifdef BIAS
  DATA_TYPEN out_val0 = VLOADN(0, bias + out_channel_idx);
  DATA_TYPEN out_val1 = out_val0;
#else
  DATA_TYPEN out_val0 = 0;
  DATA_TYPEN out_val1 = 0;
#endif
  for (int in_height_idx = in_height_beg; in_height_idx != in_height_end; ++in_height_idx) {
    for (int in_width_idx = in_width_beg; in_width_idx != in_width_end; ++in_width_idx) {
      for (int in_channel_idx = 0; in_channel_idx != in_channels; in_channel_idx += N) {
        in_val0 = VLOADN(0, in + in_val_idx);
        in_val1 = VLOADN(0, in + in_val_idx + in_width_stride);
        in_val_idx += N;

#define LOAD_KERNEL_AND_CALC(i)                             \
        kernel_val = VLOADN(0, weight + kernel_val_idx);    \
        out_val0 = mad(in_val0.s##i, kernel_val, out_val0); \
        out_val1 = mad(in_val1.s##i, kernel_val, out_val1); \
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

    in_val_idx += in_width_gap_size;
  }

#if defined(USE_RELU)
  out_val0 = fmax(out_val0, (DATA_TYPE)0);
  out_val1 = fmax(out_val1, (DATA_TYPE)0);
#endif

  const int out_width_stride = mul24(out_channels, out_width_blocks);
  const int out_val_idx = mad24(mad24(out_height_idx, out_width, out_width_idx), 
                                out_channels, 
                                out_channel_idx);
  VSTOREN(out_val0, 0, out + out_val_idx);
  if (out_width_idx + out_width_blocks >= out_width) return;
  VSTOREN(out_val1, 0, out + out_val_idx + out_width_stride);
}
