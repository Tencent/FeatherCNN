#define CHANNEL_GROUP_SIZE 4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void eltwise(__global const half* restrict input0,
                      __global const half* restrict input1,
                      __global half* restrict output,
                      __private const int height,
                      __private const int width,
                      __private const int channels) {
  const int height_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  if (height_idx >= height || width_idx >= width) return;
  const int channel_idx = get_global_id(2) * CHANNEL_GROUP_SIZE;

  int val_idx = (height_idx * width + width_idx) * channels + channel_idx;
  half4 in_val0 = vload4(0, &input0[val_idx]);
  half4 in_val1 = vload4(0, &input1[val_idx]);
  vstore4(in_val0 + in_val1, 0, &output[val_idx]);
}