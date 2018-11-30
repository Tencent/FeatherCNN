#define IN_CHANNEL_GROUP_SIZE 4
#define OUT_CHANNEL_GROUP_SIZE 4
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
  half4 out_val = vload4(0, &bias[out_channel_idx]);
  for (int h = 0; h != input_height; ++h) {
    const int val_base_width_idx = h * input_width * input_channels;
    for (int w = 0; w != input_width; ++w) {
      const int val_base_idx = val_base_width_idx + w * input_channels;
#pragma unroll
      for (int c = 0; c < input_channels; c += IN_CHANNEL_GROUP_SIZE) {
        half4 in_val = vload4(0, &input[val_base_idx + c]);

        half4 kernel_val = vload4(0, &weights[kernel_val_idx]);
        out_val.x += dot(kernel_val, in_val);

        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
        kernel_val = vload4(0, &weights[kernel_val_idx]);
        out_val.y += dot(kernel_val, in_val);

        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
        kernel_val = vload4(0, &weights[kernel_val_idx]);
        out_val.z += dot(kernel_val, in_val);

        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
        kernel_val = vload4(0, &weights[kernel_val_idx]);
        out_val.w += dot(kernel_val, in_val);

        kernel_val_idx += IN_CHANNEL_GROUP_SIZE;
      }
    }
  }

  half4 zero = (half4)0;
  if (use_relu) {
    out_val = fmax(out_val, zero);
  }

  vstore4(out_val, 0, &output[out_channel_idx]);
}
