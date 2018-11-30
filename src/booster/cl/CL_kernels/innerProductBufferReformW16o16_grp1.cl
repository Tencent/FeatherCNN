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
  const int out_channel_idx = get_global_id(2) << 4;

  int in_val_idx = 0;
  int kernel_val_idx = mul24(out_channel_idx, mul24(mul24(input_height, input_width), input_channels));

  half16 in_val, kernel_val;
  half16 out_val = vload16(0, &bias[out_channel_idx]);
  for (int in_height_idx = 0; in_height_idx != input_height; ++in_height_idx) {
    for (int in_width_idx = 0; in_width_idx != input_width; ++in_width_idx) {
#pragma unroll
      for (int in_channel_idx = 0; in_channel_idx < input_channels; in_channel_idx += 16) {
        in_val = vload16(0, &input[in_val_idx]);
        in_val_idx += 16;
        
#define LOAD_KERNEL_AND_CALC(k, i)                          \
        kernel_val = vload##k(0, &weights[kernel_val_idx]); \
        out_val = mad(in_val.s##i, kernel_val, out_val);    \
        kernel_val_idx += k;

        LOAD_KERNEL_AND_CALC(16, 0);
        LOAD_KERNEL_AND_CALC(16, 1);
        LOAD_KERNEL_AND_CALC(16, 2);
        LOAD_KERNEL_AND_CALC(16, 3);
        LOAD_KERNEL_AND_CALC(16, 4);
        LOAD_KERNEL_AND_CALC(16, 5);
        LOAD_KERNEL_AND_CALC(16, 6);
        LOAD_KERNEL_AND_CALC(16, 7);
        LOAD_KERNEL_AND_CALC(16, 8);
        LOAD_KERNEL_AND_CALC(16, 9);
        LOAD_KERNEL_AND_CALC(16, a);
        LOAD_KERNEL_AND_CALC(16, b);
        LOAD_KERNEL_AND_CALC(16, c);
        LOAD_KERNEL_AND_CALC(16, d);
        LOAD_KERNEL_AND_CALC(16, e);
        LOAD_KERNEL_AND_CALC(16, f);

#undef LOAD_KERNEL_AND_CALC
      }
    }
  }

  if (use_relu) {
    out_val = fmax(out_val, (half16)0);
  }

  vstore16(out_val, 0, &output[out_channel_idx]);
}
