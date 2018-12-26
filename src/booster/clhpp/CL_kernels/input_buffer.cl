#include <common.h>

// N = 4, 8, or 16, which is the channel group size. 
__kernel void chw_to_hwc(__global const IN_DATA_TYPE* restrict in, /* [c, h, w] */
                         __global DATA_TYPE* restrict out,         /* [h, w, (c+N-1)/N * N] */
                         __private const int height,
                         __private const int width) {
  const int height_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  if (height_idx >= height || width_idx >= width) return;

#if IN_CHANNELS <= 4 /* common case */

  int idx = mad24(height_idx, width, width_idx);
  const int out_val_idx = mul24(idx, N);

  IN_DATA_TYPEN val = 0;
  val.s0 = in[idx];
#if IN_CHANNELS >= 2
  const int height_x_width = mul24(height, width);
  idx += height_x_width;
  val.s1 = in[idx];
#if IN_CHANNELS >= 3
  idx += height_x_width;
  val.s2 = in[idx];
#if IN_CHANNELS == 4
  idx += height_x_width;
  val.s3 = in[idx];
#endif /* IN_CHANNELS == 4 */
#endif /* IN_CHANNELS >= 3 */
#endif /* IN_CHANNELS >= 2 */
  VSTOREN(CONVERTN(val), 0, &out[out_val_idx]);

#else /* IN_CHANNELS > 4 */

  const int channel_group_idx = get_global_id(2);
  const int in_channel_beg = mul24(channel_group_idx, N);
  const int channels_within_group = min(N, IN_CHANNELS - in_channel_beg);
  const int in_channel_end = in_channel_beg + channels_within_group;
  const int height_x_width = mul24(height, width);  
  const int height_width_idx = mad24(height_idx, width, width_idx);
  int idx = mad24(in_channel_end, height_x_width, height_width_idx);
  IN_DATA_TYPEN val = 0;
#define LOAD_INPUT(i)    \
  idx -= height_x_width; \
  val.s##i = in[idx];

  // load from back to front.
  switch (channels_within_group) {
#if N == 16
    case 16:
      LOAD_INPUT(f);
    case 15:
      LOAD_INPUT(e);
    case 14:
      LOAD_INPUT(d);
    case 13:
      LOAD_INPUT(c);
    case 12:
      LOAD_INPUT(b);
    case 11:
      LOAD_INPUT(a);
    case 10:
      LOAD_INPUT(9);
    case 9:
      LOAD_INPUT(8);
#endif
#if N == 8 || N == 16
    case 8:
      LOAD_INPUT(7);
    case 7:
      LOAD_INPUT(6);
    case 6:
      LOAD_INPUT(5);
    case 5:
      LOAD_INPUT(4);
#endif
    case 4:
      LOAD_INPUT(3);
    case 3:
      LOAD_INPUT(2);
    case 2:
      LOAD_INPUT(1);
    case 1:
      LOAD_INPUT(0);
  }
  const int out_channels = mul24((IN_CHANNELS+N-1)/N, N);
  const int out_val_idx = mad24(height_width_idx, out_channels, in_channel_beg);
  VSTOREN(CONVERTN(val), 0, &out[out_val_idx]);

#endif
}
