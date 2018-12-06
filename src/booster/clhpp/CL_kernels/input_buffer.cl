#include <common.h>

__kernel void init1O4(__global const float* restrict input, /* [c, h, w] */
                      __global DATA_TYPE* restrict output,  /* [h, w, 4] */
                      __private const int height,
                      __private const int width) {
  int height_idx = get_global_id(0);
  int width_idx = get_global_id(1);
  if (height_idx >= height || width_idx >= width) return;

#if INPUT_CHANNELS == 1
  const int x_idx = mad24(height_idx, width, width_idx);
  float4 out_val_float4 = (float4)(input[x_idx], 0.0f, 0.0f, 0.0f);
#elif INPUT_CHANNELS == 2
  const int in_channel_size = mul24(height, width);
  const int x_idx = mad24(height_idx, width, width_idx);
  const int y_idx = x_idx + in_channel_size;
  float4 out_val_float4 = (float4)(input[x_idx], input[y_idx], 0.0f, 0.0f);
#elif INPUT_CHANNELS == 3
  const int in_channel_size = mul24(height, width);
  const int x_idx = mad24(height_idx, width, width_idx);
  const int y_idx = x_idx + in_channel_size;
  const int z_idx = y_idx + in_channel_size;
  float4 out_val_float4 = (float4)(input[x_idx], input[y_idx], input[z_idx], 0.0f);
#elif INPUT_CHANNELS == 4
  const int in_channel_size = mul24(height, width);
  const int x_idx = mad24(height_idx, width, width_idx);
  const int y_idx = x_idx + in_channel_size;
  const int z_idx = y_idx + in_channel_size;
  const int w_idx = z_idx + in_channel_size;
  float4 out_val_float4 = (float4)(input[x_idx], input[y_idx], input[z_idx], input[w_idx]);
#else
  return;
#endif

  const int out_val_idx = x_idx << 2;
  DATA_TYPE4 out_val = CONVERT4(out_val_float4);
  vstore4(out_val, 0, &output[out_val_idx]);
}
