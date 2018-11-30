#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void convolution_depthwise(__global const half* restrict input,
                                    __global const half* restrict weights,
                                    __global const half* restrict bias,
                                    __global half* restrict output,
                                    __private const int input_channels,
                                    __private const int input_height,
                                    __private const int input_width,
                                    __private const int output_height,
                                    __private const int output_width,
                                    __private const int kernel_height,
                                    __private const int kernel_width,
                                    __private const int stride_height,
                                    __private const int stride_width,
                                    __private const int padding_top,
                                    __private const int padding_left,
                                    __private const int use_relu) {
  const int out_height_idx = get_global_id(0);
  const int out_width_idx = get_global_id(1);
  if (out_height_idx >= output_height || out_width_idx >= output_width) return;
  const int out_channel_idx = get_global_id(2) << 3;

  const int in_height_beg = mad24(out_height_idx, stride_height, -padding_top);
  const int in_height_end = in_height_beg + kernel_height;
  const int in_width_beg = mad24(out_width_idx, stride_width, -padding_left);
  const int in_width_end = in_width_beg + kernel_width;
  const int kernel_height_size = mul24(kernel_width, input_channels);
  int kernel_val_idx = out_channel_idx;

  half8 in_val, kernel_val;
  half8 out_val = vload8(0, &bias[out_channel_idx]);
  for (int in_height_idx = in_height_beg; in_height_idx != in_height_end; ++in_height_idx) {
    if (in_height_idx < 0 || in_height_idx >= input_height) {
      kernel_val_idx += kernel_height_size;
      continue;
    }

    int in_val_idx = mad24(mad24(in_height_idx, input_width, in_width_beg), input_channels, out_channel_idx);
    for (int in_width_idx = in_width_beg; in_width_idx != in_width_end;
         ++in_width_idx, in_val_idx += input_channels, kernel_val_idx += input_channels) {
      if (in_width_idx < 0 || in_width_idx >= input_width) continue;

      in_val = vload8(0, &input[in_val_idx]);
      kernel_val = vload8(0, &weights[kernel_val_idx]);
      out_val += in_val * kernel_val;
    }
  }

  if (use_relu) {
    out_val = fmax(out_val, (half8)0);
  }

  int out_val_idx = mad24(mad24(out_height_idx, output_width, out_width_idx), input_channels, out_channel_idx);
  vstore8(out_val, 0, &output[out_val_idx]);
}
