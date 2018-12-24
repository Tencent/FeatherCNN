#include <common.h>

// N = 4, 8, or 16, which is the channel group size.
__kernel void eltwise(__global const DATA_TYPE* restrict input0, /* [h, w, c] */
                      __global const DATA_TYPE* restrict input1, /* [h, w, c] */
                      __global DATA_TYPE* restrict output,       /* [h, w, c] */
                      __private const int height,
                      __private const int width,
                      __private const int channels) {            /* a multiple of 4 */
  const int height_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  if (height_idx >= height || width_idx >= width) return;
  const int out_channel_group_idx = get_global_id(2);
  const int out_channel_idx = mul24(out_channel_group_idx, N);

  const int val_idx = mad24(mad24(height_idx, width, width_idx), channels, out_channel_idx);
  DATA_TYPEN in_val0 = VLOADN(0, &input0[val_idx]);
  DATA_TYPEN in_val1 = VLOADN(0, &input1[val_idx]);
  VSTOREN(in_val0 + in_val1, 0, &output[val_idx]);
}