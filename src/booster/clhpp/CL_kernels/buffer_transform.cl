#include <common.h>

// N = 4, 8, or 16, which is the channel group size. 
__kernel void pad_input(__global const DATA_TYPE* restrict in, /* [ih, iw, c] */
                        __global DATA_TYPE* restrict out,      /* [oh, ow, c] */
                        __private const int channels,          /* a multiple of N */
                        __private const int in_height,
                        __private const int in_width,
                        __private const int out_height,
                        __private const int out_width,
                        __private const int pad_top,
                        __private const int pad_left) {
  const int out_height_idx = get_global_id(0);
  const int out_width_idx = get_global_id(1);
  if (out_height_idx >= out_height || out_width_idx >= out_width) return;
  const int channel_group_idx = get_global_id(2);
  const int channel_idx = mul24(channel_group_idx, N);

  const int in_height_idx = out_height_idx - pad_top;
  const int in_width_idx = out_width_idx - pad_left;
  DATA_TYPEN out_val = 0;
  if (0 <= in_height_idx && in_height_idx < in_height &&
      0 <= in_width_idx && in_width_idx < in_width) {
    const int in_val_idx = mad24(mad24(in_height_idx, in_width, in_width_idx), 
                                 channels,
                                 channel_idx);
    out_val = VLOADN(0, &in[in_val_idx]);
  }

  const int out_val_idx = mad24(mad24(out_height_idx, out_width, out_width_idx),
                                channels,
                                channel_idx);
  VSTOREN(out_val, 0, &out[out_val_idx]);
}