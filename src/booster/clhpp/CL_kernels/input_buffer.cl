#include <common.h>

// N = 4, 8, or 16, which is the channel group size. 
__kernel void chw_to_hwc(__global const IN_DATA_TYPE* restrict input, /* [c, h, w] */
                         __global DATA_TYPE* restrict output,         /* [h, w, (c+N-1)/N * N] */
                         __private const int height,
                         __private const int width) {
  const int height_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  if (height_idx >= height || width_idx >= width) return;

#if INPUT_CHANNELS <= 4 /* common case */

  int idx = mad24(height_idx, width, width_idx);
  const int out_val_idx = mul24(idx, N);

  IN_DATA_TYPEN val = 0;
  val.s0 = input[idx];
#if INPUT_CHANNELS >= 2
  const int in_channel_size = mul24(height, width);
  idx += in_channel_size;
  val.s1 = input[idx];
#if INPUT_CHANNELS >= 3
  idx += in_channel_size;
  val.s2 = input[idx];
#if INPUT_CHANNELS == 4
  idx += in_channel_size;
  val.s3 = input[idx];
#endif /* INPUT_CHANNELS == 4 */
#endif /* INPUT_CHANNELS >= 3 */
#endif /* INPUT_CHANNELS >= 2 */
  VSTOREN(CONVERTN(val), 0, &output[out_val_idx]);

#else /* INPUT_CHANNELS > 4 */

  const int channel_idx = mul24(get_global_id(2), N);
  const int in_channel_size = mul24(height, width);
  const int height_width_idx = mad24(height_idx, width, width_idx);
  int idx = mad24(channel_idx, in_channel_size, height_width_idx);
  const int output_channels = mul24((INPUT_CHANNELS+N-1)/N, N);
  const int out_val_idx = mad24(height_width_idx, output_channels, channel_idx);

  IN_DATA_TYPEN val = 0;
  switch (INPUT_CHANNELS - channel_idx) {
    case 1:
      val.s0 = input[idx];

      break;
    case 2:
      val.s0 = input[idx];
      idx += in_channel_size;
      val.s1 = input[idx];

      break;
    case 3:
      val.s0 = input[idx];
      idx += in_channel_size;
      val.s1 = input[idx];
      idx += in_channel_size;
      val.s2 = input[idx];

      break;
    default:
      val.s0 = input[idx];
      idx += in_channel_size;
      val.s1 = input[idx];
      idx += in_channel_size;
      val.s2 = input[idx];
      idx += in_channel_size;
      val.s3 = input[idx];

      break;
  }
  VSTOREN(CONVERTN(val), 0, &output[out_val_idx]);

#endif
}
