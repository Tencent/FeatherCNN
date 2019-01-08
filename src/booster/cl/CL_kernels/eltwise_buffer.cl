#include <common.h>

// N = 4, 8, or 16, which is the channel block size.
__kernel void eltwise(__global const DATA_TYPE* restrict in0, /* [h, w, c] */
                      __global const DATA_TYPE* restrict in1, /* [h, w, c] */
                      __global DATA_TYPE* restrict out,       /* [h, w, c] */
                      __private const int height,
                      __private const int width,
                      __private const int channels) {         /* a multiple of N */
  const int height_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  if (height_idx >= height || width_idx >= width) return;
  const int out_channel_block_idx = get_global_id(2);
  const int out_channel_idx = mul24(out_channel_block_idx, N);

  const int val_idx = mad24(mad24(height_idx, width, width_idx), 
                            channels, 
                            out_channel_idx);
  DATA_TYPEN in_val0 = VLOADN(0, in0 + val_idx);
  DATA_TYPEN in_val1 = VLOADN(0, in1 + val_idx);
  VSTOREN(in_val0 + in_val1, 0, out + val_idx);
}