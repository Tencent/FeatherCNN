#include <common.h>

// N = 4, 8, or 16, which is the channel group size.
__kernel void eltwise(__global const DATA_TYPE* restrict input0, /* [c/N, h, w, N] */
                      __global const DATA_TYPE* restrict input1, /* [c/N, h, w, N] */
                      __global DATA_TYPE* restrict output,       /* [c/N, h, w, N] */
                      __private const int height,
                      __private const int width,
                      __private const int channels) {            /* a multiple of N */
  const int height_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  if (height_idx >= height || width_idx >= width) return;
  const int channel_group_idx = get_global_id(2);

  const int val_idx = mul24(N, mad24(mad24(channel_group_idx, 
                                           height, 
                                           height_idx),
                                     width, 
                                     width_idx));
  DATA_TYPEN in_val0 = VLOADN(0, &input0[val_idx]);
  DATA_TYPEN in_val1 = VLOADN(0, &input1[val_idx]);
  VSTOREN(in_val0 + in_val1, 0, &output[val_idx]);
}
