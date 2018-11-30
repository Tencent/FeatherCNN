#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void convolution(__global const half* restrict input,   /* [h, w, c] */
                          __global const half* restrict weights, /* [cout/16, h, w, [cin, 16, 1]] */
                          __global const half* restrict bias,    /* cout */
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
  const int out_width_idx = get_global_id(1) << 1;
  if (out_height_idx >= output_height || out_width_idx >= output_width) return;
  const int out_channel_idx = get_global_id(2) << 4;

  const int in_height_beg = mad24(out_height_idx, stride_height, -padding_top);
  const int in_width_beg0 = mad24(out_width_idx, stride_width, -padding_left);
  const int in_width_beg1 = in_width_beg0 + stride_width;
  int kernel_val_idx = mul24(out_channel_idx, mul24(mul24(kernel_height, kernel_width), input_channels));

  half16 in_val0, in_val1, kernel_val;
  half16 out_val0 = vload16(0, &bias[out_channel_idx]);
  half16 out_val1 = out_val0;
  for (int h = 0; h != kernel_height; ++h) {
    int in_height_idx = in_height_beg + h;
    if (in_height_idx < 0) in_height_idx = -in_height_idx;
    if (in_height_idx >= input_height) in_height_idx = input_height * 2 - 2 - in_height_idx;

    const int in_val_base_width_idx = mul24(mul24(in_height_idx, input_width), input_channels);
    for (int w = 0; w != kernel_width; ++w) {
      int in_width_idx0 = in_width_beg0 + w;
      if (in_width_idx0 < 0) in_width_idx0 = -in_width_idx0;
      if (in_width_idx0 >= input_width) in_width_idx0 = input_width * 2 - 2 - in_width_idx0;
      int in_width_idx1 = in_width_beg1 + w;
      if (in_width_idx1 < 0) in_width_idx1 = -in_width_idx1;
      if (in_width_idx1 >= input_width) in_width_idx1 = input_width * 2 - 2 - in_width_idx1;

      const int in_val_beg0 = mad24(in_width_idx0, input_channels, in_val_base_width_idx);
      const int in_val_beg1 = mad24(in_width_idx1, input_channels, in_val_base_width_idx);
      for (int c = 0; c < input_channels; c += 16) {
        in_val0 = vload16(0, &input[in_val_beg0 + c]);
        in_val1 = vload16(0, &input[in_val_beg1 + c]);

#define LOAD_KERNEL_AND_CALC(k, i)                          \
        kernel_val = vload##k(0, &weights[kernel_val_idx]); \
        out_val0 = mad(in_val0.s##i, kernel_val, out_val0); \
        out_val1 = mad(in_val1.s##i, kernel_val, out_val1); \
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
    out_val0 = fmax(out_val0, (half16)0);
    out_val1 = fmax(out_val1, (half16)0);
  }

  int out_val_idx = mad24(mad24(out_height_idx, output_width, out_width_idx), output_channels, out_channel_idx);
  vstore16(out_val0, 0, &output[out_val_idx]);
  if (out_width_idx + 1 < output_width) {
    vstore16(out_val1, 0, &output[out_val_idx + output_channels]);
  }
}
