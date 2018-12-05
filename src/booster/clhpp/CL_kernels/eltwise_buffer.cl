#include <common.h>

__kernel void eltwise(__global const DATA_TYPE* restrict input0,
                      __global const DATA_TYPE* restrict input1,
                      __global DATA_TYPE* restrict output,
                      __private const int height,
                      __private const int width,
                      __private const int channels) {
  const int height_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  if (height_idx >= height || width_idx >= width) return;
  const int channel_idx = get_global_id(2) * CHANNEL_GROUP_SIZE;

  int val_idx = mad24(mad24(height_idx, width, width_idx), channels, channel_idx);
  DATA_TYPEN in_val0 = VLOADN(0, &input0[val_idx]);
  DATA_TYPEN in_val1 = VLOADN(0, &input1[val_idx]);
  VSTOREN(in_val0 + in_val1, 0, &output[val_idx]);
}