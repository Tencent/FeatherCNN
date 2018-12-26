#include <common.h>

#define HALF_MAX 0x1.ffcp15h

// N = 4, 8, or 16, which is the channel group size.
__kernel void pooling(__global const DATA_TYPE* restrict in, /* [ih, iw, c] */
                      __global DATA_TYPE* restrict out,      /* [oh, ow, c] */
                      __private const int channels,          /* a multiple of 4 */
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
  in_height_beg = max(0, in_height_beg);
  in_height_end = min(in_height_end, in_height);
  int in_width_beg = mad24(out_width_idx, stride_width, -pad_left);
  int in_width_end = in_width_beg + kernel_width;
  in_width_beg = max(0, in_width_beg);
  in_width_end = min(in_width_end, in_width);
  const int in_width_gap_size = mul24(in_width_beg + in_width - in_width_end, channels);
  int in_val_idx = mad24(mad24(in_height_beg, in_width, in_width_beg), 
                         channels,
                         channel_idx);

#ifdef AVE_POOLING
  DATA_TYPEN out_val = 0;
#else
  DATA_TYPEN out_val = (DATA_TYPEN)(MIN_VAL);
#endif
  for (int in_height_idx = in_height_beg; in_height_idx != in_height_end; ++in_height_idx) {
    for (int in_width_idx = in_width_beg; in_width_idx != in_width_end; ++in_width_idx) {
#ifdef AVE_POOLING
      out_val += VLOADN(0, &in[in_val_idx]);
#else
      out_val = fmax(out_val, VLOADN(0, &in[in_val_idx]));
#endif
      in_val_idx += channels;
    }
    
    in_val_idx += in_width_gap_size;
  }

#ifdef AVE_POOLING
  out_val /= mul24(in_height_end - in_height_beg, in_width_end - in_width_beg);
#endif

  const int out_val_idx = mad24(mad24(out_height_idx, out_width, out_width_idx), 
                                channels, 
                                channel_idx);
  VSTOREN(out_val, 0, &out[out_val_idx]);
}
