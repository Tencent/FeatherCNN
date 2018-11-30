#define CHANNEL_GROUP_SIZE 4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define MIN_VAL -100000.0f
__kernel void pooling(__global const half* restrict input,
                      __global half* restrict output,
                      __private const int input_width,
                      __private const int output_height,
                      __private const int output_width,
                      __private const int output_channels,
                      __private const int kernel_height,
                      __private const int kernel_width,
                      __private const int stride_height,
                      __private const int stride_width) {
  const int out_height_idx = get_global_id(0);
  const int out_width_idx = get_global_id(1);
  if (out_height_idx >= output_height || out_width_idx >= output_width) return;
  const int out_channel_idx = get_global_id(2) * CHANNEL_GROUP_SIZE;

  const int in_height_idx = out_height_idx * stride_height;
  const int in_width_idx = out_width_idx * stride_width;
  half4 out_val = (half4)(MIN_VAL);
  for (int h = in_height_idx; h < input_height; ++h) {
    for (int w = in_width_idx; w < input_width; ++w) {
      int in_val_idx = (h * input_width + w)
                          * output_channels + out_channel_idx;
      half4 in_val = vload4(0, &input[in_val_idx]);
      out_val = fmax(out_val, in_val);
    }
  }

  int out_val_idx = (out_height_idx * output_width + out_width_idx) * output_channels + out_channel_idx;
  vstore4(out_val, 0, &output[out_val_idx]);
}
