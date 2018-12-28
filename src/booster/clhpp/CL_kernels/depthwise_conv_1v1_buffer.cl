#include <common.h>

// N = 4, 8, or 16, which is the channel group size. 
__kernel void convolution_depthwise(__global const DATA_TYPE* restrict in,     /* [ih, iw, c] */
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
                                    __private const int stride_width,
                                    __private const int pad_top,
                                    __private const int pad_left) {
  const int out_height_idx = get_global_id(0);
  const int out_width_idx = get_global_id(1);
  if (out_height_idx >= out_height || out_width_idx >= out_width) return;
  const int channel_group_idx = get_global_id(2);
  const int channel_idx = mul24(channel_group_idx, N);

  int in_height_beg = mad24(out_height_idx, stride_height, -pad_top);
  int in_height_end = in_height_beg + kernel_height;
  const int kernel_height_beg_gap = select(0, -in_height_beg, in_height_beg < 0);
  in_height_beg = max(0, in_height_beg);
  in_height_end = min(in_height_end, in_height);
  int in_width_beg = mad24(out_width_idx, stride_width, -pad_left);
  int in_width_end = in_width_beg + kernel_width;
  const int kernel_width_beg_gap = select(0, -in_width_beg, in_width_beg < 0);
  const int kernel_width_end_gap = select(0, in_width_end - in_width, in_width_end > in_width);
  in_width_beg = max(0, in_width_beg);
  in_width_end = min(in_width_end, in_width);
  const int in_width_gap_size = mul24(in_width_beg + in_width - in_width_end, channels);
  int in_val_idx = mad24(mad24(in_height_beg, in_width, in_width_beg), 
                         channels,
                         channel_idx);

  const int kernel_width_size = N;
  const int kernel_height_size = mul24(kernel_width, kernel_width_size);
  const int kernel_height_beg_gap_size = mul24(kernel_height_beg_gap, kernel_height_size);
  const int kernel_width_beg_gap_size = mul24(kernel_width_beg_gap, kernel_width_size);
  const int kernel_width_end_gap_size = mul24(kernel_width_end_gap, kernel_width_size);
  const int kernel_width_gap_size = kernel_width_beg_gap_size + kernel_width_end_gap_size;
  int kernel_val_idx = mad24(channel_group_idx, 
                             mul24(kernel_height, kernel_height_size),
                             kernel_height_beg_gap_size) + kernel_width_beg_gap_size;

  DATA_TYPEN in_val, kernel_val;
#ifdef BIAS
  DATA_TYPEN out_val = VLOADN(0, &bias[channel_idx]);
#else
  DATA_TYPEN out_val = 0;
#endif
  for (int in_height_idx = in_height_beg; in_height_idx != in_height_end; ++in_height_idx) {
    for (int in_width_idx = in_width_beg; in_width_idx != in_width_end; ++in_width_idx) {
      in_val = VLOADN(0, &in[in_val_idx]);
      kernel_val = VLOADN(0, &weight[kernel_val_idx]);
      out_val = mad(in_val, kernel_val, out_val);

      in_val_idx += channels;
      kernel_val_idx += N;
    }

    in_val_idx += in_width_gap_size;
    kernel_val_idx += kernel_width_gap_size;
  }

#if defined(USE_RELU)
  out_val = fmax(out_val, (DATA_TYPEN)0);
#endif

  const int out_val_idx = mad24(mad24(out_height_idx, out_width, out_width_idx), 
                                channels, 
                                channel_idx);
  VSTOREN(out_val, 0, &out[out_val_idx]);
}
