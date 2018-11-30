#define IN_CHANNEL_GROUP_SIZE 16
#define OUT_CHANNEL_GROUP_SIZE 16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void inner_product(__global const half* restrict input,
                            __global const half* restrict weights,
                            __global const half* restrict bias,
                            __global half* restrict output,
                            __private const int input_channels,
                            __private const int output_channels,
                            __private const int input_height,
                            __private const int input_width,
                            __private const int use_relu) {
  const int out_channel_idx = get_global_id(2) * OUT_CHANNEL_GROUP_SIZE;
  const int kernel_size = input_height * input_width * input_channels;
  const int kernel_offset = kernel_size * out_channel_idx;
  int kernel_val_idx = kernel_offset;
  half16 out_val = vload16(0, &bias[out_channel_idx]);
  for (int h = 0; h != input_height; ++h) {
    const int val_base_width_idx = h * input_width * input_channels;
    for (int w = 0; w != input_width; ++w) {
      const int val_base_idx = val_base_width_idx + w * input_channels;
#pragma unroll
      for (int c = 0; c < input_channels; c += IN_CHANNEL_GROUP_SIZE) {
        half16 in_val = vload16(0, &input[val_base_idx + c]);
        
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

  vstore16(out_val, 0, &output[out_channel_idx]);
}
