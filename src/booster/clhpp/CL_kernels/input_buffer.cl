#include <common.h>

__kernel void init1O4(__global const float* restrict input, /* [c, h, w] */
                      __global DATA_TYPE* restrict output,  /* [h, w, (c+3)/4 * 4] */
                      __private const int height,
                      __private const int width) {
  const int height_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  if (height_idx >= height || width_idx >= width) return;

#if INPUT_CHANNELS <= 4

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
#else /* INPUT_CHANNELS == 4 */
  const int in_channel_size = mul24(height, width);
  const int x_idx = mad24(height_idx, width, width_idx);
  const int y_idx = x_idx + in_channel_size;
  const int z_idx = y_idx + in_channel_size;
  const int w_idx = z_idx + in_channel_size;
  float4 out_val_float4 = (float4)(input[x_idx], input[y_idx], input[z_idx], input[w_idx]);
#endif

  DATA_TYPE4 out_val = CONVERT4(out_val_float4);
  const int out_val_idx = x_idx << 2;
  vstore4(out_val, 0, &output[out_val_idx]);

#else /* INPUT_CHANNELS > 4 */

  const int channel_idx = get_global_id(2) << 2;
  const int in_channel_size = mul24(height, width);
  int idx = mad24(channel_idx, 
                  in_channel_size, 
                  mad24(height_idx, width, width_idx));
  float4 out_val_float4 = 0;
  switch (INPUT_CHANNELS - channel_idx) {
    case 1:
      out_val_float4.x = input[idx];

      break;
    case 2:
      out_val_float4.x = input[idx];
      idx += in_channel_size;
      out_val_float4.y = input[idx];

      break;
    case 3:
      out_val_float4.x = input[idx];
      idx += in_channel_size;
      out_val_float4.y = input[idx];
      idx += in_channel_size;
      out_val_float4.z = input[idx];

      break;
    default:
      out_val_float4.x = input[idx];
      idx += in_channel_size;
      out_val_float4.y = input[idx];
      idx += in_channel_size;
      out_val_float4.z = input[idx];
      idx += in_channel_size;
      out_val_float4.w = input[idx];

      break;
  }
  DATA_TYPE4 out_val = CONVERT4(out_val_float4);
  const int out_val_idx = mad24(mad24(height_idx, width, width_idx), 
                                ((INPUT_CHANNELS + 3) >> 2) << 2, 
                                channel_idx);
  vstore4(out_val, 0, &output[out_val_idx]);

#endif
}
