#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void init1O4(__global const half* restrict input,
                      __global half* restrict output,
                      __private const int height,
                      __private const int width,
                      __private const int input_channels) {
  int height_idx = get_global_id(0);
  int width_idx = get_global_id(1);
  if (height_idx >= height || width_idx >= width) return;

  const int in_val_idx = (height_idx * width + width_idx) * input_channels;
  const int out_val_idx = (height_idx * width + width_idx) * 4;
  half4 in_val = vload4(0, &input[in_val_idx]);
  in_val.w = 0.0f;
  vstore4(in_val, 0, &output[out_val_idx]);
}
