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
  const int out_channel_idx = get_global_id(2) << 3;

  int in_val_idx = 0;
  int kernel_val_idx = mul24(out_channel_idx, mul24(mul24(input_height, input_width), input_channels));

  half8 in_val, kernel_val;
  half8 out_val = vload8(0, &bias[out_channel_idx]);
  for (int in_height_idx = 0; in_height_idx != input_height; ++in_height_idx) {
    for (int in_width_idx = 0; in_width_idx != input_width; ++in_width_idx) {
#pragma unroll
      for (int in_channel_idx = 0; in_channel_idx < input_channels; in_channel_idx += 8) {
        in_val = vload8(0, &input[in_val_idx]);
        in_val_idx += 8;
        
#define LOAD_KERNEL_AND_CALC(k, i)                          \
        kernel_val = vload##k(0, &weights[kernel_val_idx]); \
        out_val = mad(in_val.s##i, kernel_val, out_val);    \
        kernel_val_idx += k;

        LOAD_KERNEL_AND_CALC(8, 0);
        LOAD_KERNEL_AND_CALC(8, 1);
        LOAD_KERNEL_AND_CALC(8, 2);
        LOAD_KERNEL_AND_CALC(8, 3);
        LOAD_KERNEL_AND_CALC(8, 4);
        LOAD_KERNEL_AND_CALC(8, 5);
        LOAD_KERNEL_AND_CALC(8, 6);
        LOAD_KERNEL_AND_CALC(8, 7);

#undef LOAD_KERNEL_AND_CALC
      }
    }
  }

  if (use_relu) {
    out_val = fmax(out_val, (half8)0);
  }

  vstore8(out_val, 0, &output[out_channel_idx]);
}
