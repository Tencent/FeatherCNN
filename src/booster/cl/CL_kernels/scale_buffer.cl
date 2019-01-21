#include <common.h>

// N = 4, 8, or 16, which is the channel block size.
__kernel void scale(__global const DATA_TYPE* restrict in, /* [ih, iw, c] */
                        __global const DATA_TYPE* restrict scale,
#ifdef BIAS
                        __global const DATA_TYPE* restrict bias,   /* [oc] */
#endif
                        __private const int out_height,
                        __private const int out_width,
                        __private const int out_channels,
                        __global DATA_TYPE* restrict out) {
  const int out_height_idx = get_global_id(0);
  const int out_width_idx = get_global_id(1);
  if (out_height_idx >= out_height || out_width_idx >= out_width) return;
  const int channel_block_idx = get_global_id(2);
  const int channel_idx = mul24(channel_block_idx, N);

  DATA_TYPEN scale_val = VLOADN(0, scale + channel_idx);
  const int val_idx = mad24(mad24(out_height_idx, out_width, out_width_idx), out_channels, channel_idx);
  DATA_TYPEN in_val = VLOADN(0, in + val_idx);

#ifdef BIAS
  DATA_TYPEN out_val = VLOADN(0, bias + channel_idx);
#else
  DATA_TYPEN out_val = 0;
#endif
  out_val = mad(in_val, scale_val, out_val);

#if defined(USE_RELU)
  out_val = fmax(out_val, (DATA_TYPE)0);
#endif

  const int out_val_idx = mad24(mad24(out_height_idx, out_width,
                    out_width_idx), out_channels, channel_idx);
  VSTOREN(out_val, 0, out + out_val_idx);

}
