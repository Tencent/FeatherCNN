#define IN_CHANNEL_GROUP_SIZE 16
#define OUT_CHANNEL_GROUP_SIZE 16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void convolution(__global const half* restrict input,
                          __global const half* restrict weights,
                          __global const half* restrict bias,
                          __global half* restrict output,
                          __private const int input_channels,
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
                          __private const int use_relu) {
  const int out_height_idx = get_global_id(0);
  const int out_width_idx = get_global_id(1);
  if (out_height_idx >= output_height || out_width_idx >= output_width) return;
  const int out_channel_idx = get_global_id(2) * OUT_CHANNEL_GROUP_SIZE;

  const int kernel_size = kernel_height * kernel_width * input_channels;
  const int kernel_offset = kernel_size * out_channel_idx;
  int kernel_val_idx = kernel_offset;

  const int in_height_base_idx = out_height_idx * stride_height - padding_top;
  const int in_width_base_idx = out_width_idx * stride_width - padding_left;
  half16 out_val = vload16(0, &bias[out_channel_idx]);
  for (int h = 0; h != kernel_height; ++h) {
    int in_height_idx = in_height_base_idx + h;
    if (in_height_idx < 0 || in_height_idx >= input_height) {
      kernel_val_idx += OUT_CHANNEL_GROUP_SIZE * input_channels * kernel_width;
      continue;
    }

    const int in_val_base_width_idx = in_height_idx * input_width * input_channels;
    for (int w = 0; w != kernel_width; ++w) {
      int in_width_idx = in_width_base_idx + w;
      if (in_width_idx < 0 || in_width_idx >= input_width) {
        kernel_val_idx += OUT_CHANNEL_GROUP_SIZE * input_channels;
        continue;
      }

      const int in_val_base_idx = in_val_base_width_idx + in_width_idx * input_channels;
      for (int c = 0; c < input_channels; c += IN_CHANNEL_GROUP_SIZE) {
        half16 in_val = vload16(0, &input[in_val_base_idx + c]);
        half16 kernel_val = vload16(0, &weights[kernel_val_idx]);
        out_val.s0 += dot(kernel_val.s0123, in_val.s0123);
        out_val.s0 += dot(kernel_val.s4567, in_val.s4567);
        out_val.s0 += dot(kernel_val.s89ab, in_val.s89ab);
        out_val.s0 += dot(kernel_val.scdef, in_val.scdef);

        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
        kernel_val = vload16(0, &weights[kernel_val_idx]);
        out_val.s1 += dot(kernel_val.s0123, in_val.s0123);
        out_val.s1 += dot(kernel_val.s4567, in_val.s4567);
        out_val.s1 += dot(kernel_val.s89ab, in_val.s89ab);
        out_val.s1 += dot(kernel_val.scdef, in_val.scdef);

        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
        kernel_val = vload16(0, &weights[kernel_val_idx]);
        out_val.s2 += dot(kernel_val.s0123, in_val.s0123);
        out_val.s2 += dot(kernel_val.s4567, in_val.s4567);
        out_val.s2 += dot(kernel_val.s89ab, in_val.s89ab);
        out_val.s2 += dot(kernel_val.scdef, in_val.scdef);

        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
        kernel_val = vload16(0, &weights[kernel_val_idx]);
        out_val.s3 += dot(kernel_val.s0123, in_val.s0123);
        out_val.s3 += dot(kernel_val.s4567, in_val.s4567);
        out_val.s3 += dot(kernel_val.s89ab, in_val.s89ab);
        out_val.s3 += dot(kernel_val.scdef, in_val.scdef);

        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
        kernel_val = vload16(0, &weights[kernel_val_idx]);
        out_val.s4 += dot(kernel_val.s0123, in_val.s0123);
        out_val.s4 += dot(kernel_val.s4567, in_val.s4567);
        out_val.s4 += dot(kernel_val.s89ab, in_val.s89ab);
        out_val.s4 += dot(kernel_val.scdef, in_val.scdef);

        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
        kernel_val = vload16(0, &weights[kernel_val_idx]);
        out_val.s5 += dot(kernel_val.s0123, in_val.s0123);
        out_val.s5 += dot(kernel_val.s4567, in_val.s4567);
        out_val.s5 += dot(kernel_val.s89ab, in_val.s89ab);
        out_val.s5 += dot(kernel_val.scdef, in_val.scdef);

        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
        kernel_val = vload16(0, &weights[kernel_val_idx]);
        out_val.s6 += dot(kernel_val.s0123, in_val.s0123);
        out_val.s6 += dot(kernel_val.s4567, in_val.s4567);
        out_val.s6 += dot(kernel_val.s89ab, in_val.s89ab);
        out_val.s6 += dot(kernel_val.scdef, in_val.scdef);

        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
        kernel_val = vload16(0, &weights[kernel_val_idx]);
        out_val.s7 += dot(kernel_val.s0123, in_val.s0123);
        out_val.s7 += dot(kernel_val.s4567, in_val.s4567);
        out_val.s7 += dot(kernel_val.s89ab, in_val.s89ab);
        out_val.s7 += dot(kernel_val.scdef, in_val.scdef);

        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
        kernel_val = vload16(0, &weights[kernel_val_idx]);
        out_val.s8 += dot(kernel_val.s0123, in_val.s0123);
        out_val.s8 += dot(kernel_val.s4567, in_val.s4567);
        out_val.s8 += dot(kernel_val.s89ab, in_val.s89ab);
        out_val.s8 += dot(kernel_val.scdef, in_val.scdef);


        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
        kernel_val = vload16(0, &weights[kernel_val_idx]);
        out_val.s9 += dot(kernel_val.s0123, in_val.s0123);
        out_val.s9 += dot(kernel_val.s4567, in_val.s4567);
        out_val.s9 += dot(kernel_val.s89ab, in_val.s89ab);
        out_val.s9 += dot(kernel_val.scdef, in_val.scdef);

        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
        kernel_val = vload16(0, &weights[kernel_val_idx]);
        out_val.sa += dot(kernel_val.s0123, in_val.s0123);
        out_val.sa += dot(kernel_val.s4567, in_val.s4567);
        out_val.sa += dot(kernel_val.s89ab, in_val.s89ab);
        out_val.sa += dot(kernel_val.scdef, in_val.scdef);



        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
        kernel_val = vload16(0, &weights[kernel_val_idx]);
        out_val.sb += dot(kernel_val.s0123, in_val.s0123);
        out_val.sb += dot(kernel_val.s4567, in_val.s4567);
        out_val.sb += dot(kernel_val.s89ab, in_val.s89ab);
        out_val.sb += dot(kernel_val.scdef, in_val.scdef);

        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
        kernel_val = vload16(0, &weights[kernel_val_idx]);
        out_val.sc += dot(kernel_val.s0123, in_val.s0123);
        out_val.sc += dot(kernel_val.s4567, in_val.s4567);
        out_val.sc += dot(kernel_val.s89ab, in_val.s89ab);
        out_val.sc += dot(kernel_val.scdef, in_val.scdef);

        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
        kernel_val = vload16(0, &weights[kernel_val_idx]);
        out_val.sd += dot(kernel_val.s0123, in_val.s0123);
        out_val.sd += dot(kernel_val.s4567, in_val.s4567);
        out_val.sd += dot(kernel_val.s89ab, in_val.s89ab);
        out_val.sd += dot(kernel_val.scdef, in_val.scdef);

        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
        kernel_val = vload16(0, &weights[kernel_val_idx]);
        out_val.se += dot(kernel_val.s0123, in_val.s0123);
        out_val.se += dot(kernel_val.s4567, in_val.s4567);
        out_val.se += dot(kernel_val.s89ab, in_val.s89ab);
        out_val.se += dot(kernel_val.scdef, in_val.scdef);

        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
        kernel_val = vload16(0, &weights[kernel_val_idx]);
        out_val.sf += dot(kernel_val.s0123, in_val.s0123);
        out_val.sf += dot(kernel_val.s4567, in_val.s4567);
        out_val.sf += dot(kernel_val.s89ab, in_val.s89ab);
        out_val.sf += dot(kernel_val.scdef, in_val.scdef);

        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
      }
    }
  }

  half16 zero = (half16)0;
  if (use_relu) {
    out_val = fmax(out_val, zero);
  }

  int out_val_idx = (out_height_idx * output_width + out_width_idx) * output_channels + out_channel_idx;
  vstore16(out_val, 0, &output[out_val_idx]);
}
