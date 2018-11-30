#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define HALF_MAX 0x1.ffcp15h
#define MIN_VAL -HALF_MAX
__kernel void pooling(__global const half* restrict input,
                      __global half* restrict output,
                      __private const int output_channels,
                      __private const int input_height,
                      __private const int input_width,
                      __private const int output_height,
                      __private const int output_width,
                      __private const int kernel_height,
                      __private const int kernel_width,
                      __private const int stride_height,
                      __private const int stride_width,
                      __private const int padding_top,
                      __private const int padding_left) {
  const int out_height_idx = get_global_id(0);
  const int out_width_idx = get_global_id(1);
  if (out_height_idx >= output_height || out_width_idx >= output_width) return;
  const int out_channel_idx = get_global_id(2) << 3;

  int in_height_beg = mad24(out_height_idx, stride_height, -padding_top);
  int in_height_end = in_height_beg + kernel_height;
  int in_width_beg = mad24(out_width_idx, stride_width, -padding_left);
  int in_width_end = in_width_beg + kernel_width;
  in_height_beg = max(0, in_height_beg);
  in_height_end = min(in_height_end, input_height);
  in_width_beg = max(0, in_width_beg);
  in_width_end = min(in_width_end, input_width);

#ifdef AVE_POOLING
  half8 out_val = (half8)0;
#else
  half8 out_val = (half8)(MIN_VAL);
#endif

  for (int in_height_idx = in_height_beg; in_height_idx != in_height_end; ++in_height_idx) {
    const int in_val_base_idx = mad24(in_height_idx, mul24(input_width, output_channels), out_channel_idx);
    const int in_val_beg = mad24(in_width_beg, output_channels, in_val_base_idx);
    const int in_val_end = mad24(in_width_end, output_channels, in_val_base_idx);
    for (int in_val_idx = in_val_beg; in_val_idx != in_val_end; in_val_idx += output_channels) {
#ifdef AVE_POOLING
      out_val += vload8(0, &input[in_val_idx]);
#else
      out_val = fmax(out_val, vload8(0, &input[in_val_idx]));
#endif
    }
  }

#ifdef AVE_POOLING
  out_val /= mul24((in_height_end - in_height_beg), (in_width_end - in_width_beg));
#endif

  int out_val_idx = mad24(mad24(out_height_idx, output_width, out_width_idx), output_channels, out_channel_idx);
  vstore8(out_val, 0, &output[out_val_idx]);
}
