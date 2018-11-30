#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define MIN_VAL -100000.0f
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
                      __private const int padding_left,
                      __private const int grain) {
  const int out_height_idx = get_global_id(0);
  const int out_width_idx = get_global_id(1);
  if (out_height_idx >= output_height || out_width_idx >= output_width) return;
  const int out_channel_idx = get_global_id(2) << 2;

  int in_height_start = mad24(out_height_idx, stride_height, -padding_top);
  int in_height_end = in_height_start + kernel_height;
  int in_width_start = mad24(out_width_idx, stride_width, -padding_left);
  int in_width_end = in_width_start + kernel_width;
  in_height_start = max(0, in_height_start);
  in_height_end = min(in_height_end, input_height);
  in_width_start = max(0, in_width_start);
  in_width_end = min(in_width_end, input_width);

  half4 out_val = (half4)(MIN_VAL);
  half4 in_val, out_val;
  for (int in_height_idx = in_height_start; in_height_idx != in_height_end; ++in_height_idx) {
    const int in_val_base_idx = mad24(in_height_idx, mul24(input_width, output_channels), out_channel_idx);
    for (int in_width_idx = in_width_start; in_width_idx != in_width_end; ++in_width_idx) {
      int in_val_idx = mad24(in_width_idx, output_channels, in_val_base_idx);

#define READ_INPUT(i)                          \
      in_val = vload##i(0, &input[in_val_idx]); \
      out_val = fmax(out_val, in_val);

      READ_INPUT(grain);

#undef READ_INPUT

      //half4 in_val = vload4(0, &input[in_val_idx]);
      //out_val = fmax(out_val, in_val);
    }
  }

  int out_val_idx = (out_height_idx * output_width + out_width_idx) * output_channels + out_channel_idx;
  vstore4(out_val, 0, &output[out_val_idx]);
}
