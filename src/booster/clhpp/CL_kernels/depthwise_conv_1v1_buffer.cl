#include <common.h>

// N = 4, 8, or 16, which is the channel group size. 
__kernel void convolution_depthwise(__global const DATA_TYPE* restrict input,   /* [ih, iw, c] */
                                    __global const DATA_TYPE* restrict weights, /* [c/N, kh, kw, [N, 1]] */
#ifdef BIAS
                                    __global const DATA_TYPE* restrict bias,    /* [c] */
#endif
                                    __global DATA_TYPE* restrict output,        /* [oh, ow, c] */
                                    __private const int channels,               /* a multiple of 4 */
                                    __private const int input_height,
                                    __private const int input_width,
                                    __private const int output_height,
                                    __private const int output_width,
                                    __private const int kernel_height,
                                    __private const int kernel_width,
                                    __private const int stride_height,
                                    __private const int stride_width,
                                    __private const int padding_top,
                                    __private const int padding_left) {
  const int out_height_idx = get_global_id(0);
  const int out_width_idx = get_global_id(1);
  if (out_height_idx >= output_height || out_width_idx >= output_width) return;
  const int out_channel_idx = get_global_id(2) * N;

  const int in_height_beg = mad24(out_height_idx, stride_height, -padding_top);
  const int in_height_end = in_height_beg + kernel_height;
  const int in_width_beg = mad24(out_width_idx, stride_width, -padding_left);
  const int in_width_end = in_width_beg + kernel_width;
  const int kernel_height_size = kernel_width * N;
  int kernel_val_idx = mul24(out_channel_idx, mul24(kernel_height, kernel_width));

  DATA_TYPEN in_val, kernel_val;
#ifdef BIAS
  DATA_TYPEN out_val = VLOADN(0, &bias[out_channel_idx]);
#else
  DATA_TYPEN out_val = 0;
#endif
  for (int in_height_idx = in_height_beg; in_height_idx != in_height_end; ++in_height_idx) {
    if (in_height_idx < 0 || in_height_idx >= input_height) {
      kernel_val_idx += kernel_height_size;
      continue;
    }

    int in_val_idx = mad24(mad24(in_height_idx, input_width, in_width_beg), channels, out_channel_idx);
    for (int in_width_idx = in_width_beg; in_width_idx != in_width_end;
         ++in_width_idx, in_val_idx += channels, kernel_val_idx += N) {
      if (in_width_idx < 0 || in_width_idx >= input_width) continue;

      in_val = VLOADN(0, &input[in_val_idx]);
      kernel_val = VLOADN(0, &weights[kernel_val_idx]);
      out_val += in_val * kernel_val;
    }
  }

#if defined(USE_RELU)
  out_val = fmax(out_val, (DATA_TYPEN)0);
#endif

  int out_val_idx = mad24(mad24(out_height_idx, output_width, out_width_idx), channels, out_channel_idx);
  VSTOREN(out_val, 0, &output[out_val_idx]);
}
