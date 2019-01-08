const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void uint8_hwc_to_hwc(__read_only image2d_t input,
                        __global half* restrict output,
                        __private const int height,
                        __private const int width) {
  int height_idx = get_global_id(0);
  int width_idx = get_global_id(1);
  if (height_idx >= height || width_idx >= width) return;

  int2 coordinate = (int2)(width_idx, height_idx);
  half4 in_val = read_imageh(input, sampler, coordinate);
  int in_val_idx = (height_idx * width + width_idx) * 4;
  in_val *= (half4){ 255.0f, 255.0f, 255.0f, 0.0f };
  in_val -= (half4){ 104.0f, 117.0f, 123.0f, 0.0f };
  vstore4(in_val, 0, &output[in_val_idx]);
}
