#include <common.h>

// N = 4, 8, or 16, which is the channel block size. 
__kernel void depthwise_conv(__global const DATA_TYPE* restrict in,     /* [ih, iw, c] */
                             __global const DATA_TYPE* restrict weight, /* [c/N, kh, kw, [N, 1]] */
#ifdef BIAS
                             __global const DATA_TYPE* restrict bias,   /* [c] */
#endif
                             __global DATA_TYPE* restrict out,          /* [oh, ow, c] */
                             __private const int channels,              /* a multiple of N */
                             __private const int in_height,
                             __private const int in_width,
                             __private const int out_height,
                             __private const int out_width,
                             __private const int kernel_height,
                             __private const int kernel_width,
                             __private const int stride_height,
                             __private const int stride_width) {
  const int out_height_idx = get_global_id(0);
  const int out_width_idx = get_global_id(1) << 2;
  if (out_height_idx >= out_height || out_width_idx >= out_width) return;
  const int channel_block_idx = get_global_id(2);
  const int channel_idx = mul24(channel_block_idx, N);

  const int in_height_beg = mul24(out_height_idx, stride_height);
  const int in_height_end = in_height_beg + kernel_height;
  const int in_width_beg = mul24(out_width_idx, stride_width);
  const int in_width_end = in_width_beg + kernel_width;
  const int in_width_gap_size = mul24(in_width_beg + in_width - in_width_end, channels);
  const int in_width_stride = mul24(stride_width, channels);
  int in_val_idx = mad24(mad24(in_height_beg, in_width, in_width_beg), 
                         channels,
                         channel_idx);

  const int kernel_size = mul24(kernel_height, kernel_width);
  int kernel_val_idx = mul24(channel_idx, kernel_size);

  DATA_TYPEN in_val0, in_val1, in_val2, in_val3, kernel_val;
#ifdef BIAS
  DATA_TYPEN out_val0 = VLOADN(0, bias + channel_idx);
  DATA_TYPEN out_val1 = out_val0;
  DATA_TYPEN out_val2 = out_val0;
  DATA_TYPEN out_val3 = out_val0;
#else
  DATA_TYPEN out_val0 = 0;
  DATA_TYPEN out_val1 = 0;
  DATA_TYPEN out_val2 = 0;
  DATA_TYPEN out_val3 = 0;
#endif
  for (int in_height_idx = in_height_beg; in_height_idx != in_height_end; ++in_height_idx) {
    for (int in_width_idx = in_width_beg; in_width_idx != in_width_end; ++in_width_idx) {
      in_val0 = VLOADN(0, in + in_val_idx);
      in_val1 = VLOADN(0, in + in_val_idx + in_width_stride);
      in_val2 = VLOADN(0, in + in_val_idx + (in_width_stride << 1));
      in_val3 = VLOADN(0, in + in_val_idx + in_width_stride + (in_width_stride << 1));
      in_val_idx += channels;

      kernel_val = VLOADN(0, weight + kernel_val_idx);
      kernel_val_idx += N;

      out_val0 = mad(in_val0, kernel_val, out_val0);
      out_val1 = mad(in_val1, kernel_val, out_val1);
      out_val2 = mad(in_val2, kernel_val, out_val2);
      out_val3 = mad(in_val3, kernel_val, out_val3);
    }

    in_val_idx += in_width_gap_size;
  }

#if defined(USE_RELU)
  out_val0 = fmax(out_val0, (DATA_TYPE)0);
  out_val1 = fmax(out_val1, (DATA_TYPE)0);
  out_val2 = fmax(out_val2, (DATA_TYPE)0);
  out_val3 = fmax(out_val3, (DATA_TYPE)0);
#endif

  int out_val_idx = mad24(mad24(out_height_idx, out_width, out_width_idx), 
                          channels, 
                          channel_idx);
  VSTOREN(out_val0, 0, out + out_val_idx);
  if (out_width_idx + 1 >= out_width) return;
  out_val_idx += channels;
  VSTOREN(out_val1, 0, out + out_val_idx);
  if (out_width_idx + 2 >= out_width) return;
  out_val_idx += channels;
  VSTOREN(out_val2, 0, out + out_val_idx);
  if (out_width_idx + 3 >= out_width) return;
  out_val_idx += channels;
  VSTOREN(out_val3, 0, out + out_val_idx);
}

// N = 4, 8, or 16, which is the channel block size. 
__kernel void coalesced_depthwise_conv(__global const DATA_TYPE* restrict in,     /* [ih, iw, c] */
                                       __global const DATA_TYPE* restrict weight, /* [c/N, kh, kw, [N, 1]] */
#ifdef BIAS
                                       __global const DATA_TYPE* restrict bias,   /* [c] */
#endif
                                       __global DATA_TYPE* restrict out,          /* [oh, ow, c] */
                                       __private const int channels,              /* a multiple of N */
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
  const int out_width_blocks = (out_width + 3) >> 2;
  if (out_height_idx >= out_height || out_width_idx >= out_width_blocks) return;
  const int channel_block_idx = get_global_id(2);
  const int channel_idx = mul24(channel_block_idx, N);

  const int in_height_beg = mul24(out_height_idx, stride_height);
  const int in_height_end = in_height_beg + kernel_height;
  const int in_width_beg = mul24(out_width_idx, stride_width);
  const int in_width_end = in_width_beg + kernel_width;
  const int in_width_gap_size = mul24(in_width_beg + in_width - in_width_end, channels);
  const int in_width_stride = mul24(mul24(stride_width, channels), out_width_blocks);
  int in_val_idx = mad24(mad24(in_height_beg, in_width, in_width_beg), 
                         channels,
                         channel_idx);

  const int kernel_size = mul24(kernel_height, kernel_width);
  int kernel_val_idx = mul24(channel_idx, kernel_size);

  DATA_TYPEN in_val0, in_val1, in_val2, in_val3, kernel_val;
#ifdef BIAS
  DATA_TYPEN out_val0 = VLOADN(0, bias + channel_idx);
  DATA_TYPEN out_val1 = out_val0;
  DATA_TYPEN out_val2 = out_val0;
  DATA_TYPEN out_val3 = out_val0;
#else
  DATA_TYPEN out_val0 = 0;
  DATA_TYPEN out_val1 = 0;
  DATA_TYPEN out_val2 = 0;
  DATA_TYPEN out_val3 = 0;
#endif
  for (int in_height_idx = in_height_beg; in_height_idx != in_height_end; ++in_height_idx) {
    for (int in_width_idx = in_width_beg; in_width_idx != in_width_end; ++in_width_idx) {
      in_val0 = VLOADN(0, in + in_val_idx);
      in_val1 = VLOADN(0, in + in_val_idx + in_width_stride);
      in_val2 = VLOADN(0, in + in_val_idx + (in_width_stride << 1));
      in_val3 = VLOADN(0, in + in_val_idx + in_width_stride + (in_width_stride << 1));
      in_val_idx += channels;

      kernel_val = VLOADN(0, weight + kernel_val_idx);
      kernel_val_idx += N;

      out_val0 = mad(in_val0, kernel_val, out_val0);
      out_val1 = mad(in_val1, kernel_val, out_val1);
      out_val2 = mad(in_val2, kernel_val, out_val2);
      out_val3 = mad(in_val3, kernel_val, out_val3);
    }

    in_val_idx += in_width_gap_size;
  }

#if defined(USE_RELU)
  out_val0 = fmax(out_val0, (DATA_TYPE)0);
  out_val1 = fmax(out_val1, (DATA_TYPE)0);
  out_val2 = fmax(out_val2, (DATA_TYPE)0);
  out_val3 = fmax(out_val3, (DATA_TYPE)0);
#endif

  const int out_width_stride = mul24(channels, out_width_blocks);
  int out_val_idx = mad24(mad24(out_height_idx, out_width, out_width_idx), 
                          channels, 
                          channel_idx);
  VSTOREN(out_val0, 0, out + out_val_idx);

  const int out_width_idx1 = out_width_idx + out_width_blocks;
  if (out_width_idx1 >= out_width) return;
  out_val_idx += out_width_stride;
  VSTOREN(out_val1, 0, out + out_val_idx);

  const int out_width_idx2 = out_width_idx1 + out_width_blocks;
  if (out_width_idx2 >= out_width) return;
  out_val_idx += out_width_stride;
  VSTOREN(out_val2, 0, out + out_val_idx);
  
  const int out_width_idx3 = out_width_idx2 + out_width_blocks;
  if (out_width_idx3 >= out_width) return;
  out_val_idx += out_width_stride;
  VSTOREN(out_val3, 0, out + out_val_idx);
}
