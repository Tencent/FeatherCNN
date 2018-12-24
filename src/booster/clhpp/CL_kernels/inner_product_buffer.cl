#include <common.h>

// N = 4, 8, or 16, which is the channel group size.
__kernel void inner_product(__global const DATA_TYPE* restrict input,   /* [h, w, ic] */
                            __global const DATA_TYPE* restrict weights, /* [oc/N, h, w, [ic, N, 1]] */
#ifdef BIAS
                            __global const DATA_TYPE* restrict bias,    /* [oc] */
#endif
                            __global DATA_TYPE* restrict output,        /* [oc] */
                            __private const int input_channels,         /* a multiple of 4 */
                            __private const int output_channels,        /* a multiple of 4 */
                            __private const int input_height,
                            __private const int input_width) {
  const int out_channel_group_idx = get_global_id(2);
  const int out_channel_idx = mul24(out_channel_group_idx, N);

  int in_val_idx = 0;
  int kernel_val_idx = mul24(out_channel_idx, 
                             mul24(mul24(input_height, input_width), input_channels));

  DATA_TYPEN in_val, kernel_val;
#ifdef BIAS
  DATA_TYPEN out_val = VLOADN(0, &bias[out_channel_idx]);
#else
  DATA_TYPEN out_val = 0;
#endif
  for (int in_height_idx = 0; in_height_idx != input_height; ++in_height_idx) {
    for (int in_width_idx = 0; in_width_idx != input_width; ++in_width_idx) {
#pragma unroll
      for (int in_channel_idx = 0; in_channel_idx != input_channels; in_channel_idx += N) {
        in_val = VLOADN(0, &input[in_val_idx]);
        in_val_idx += N;
        
#define LOAD_KERNEL_AND_CALC(i)                           \
        kernel_val = VLOADN(0, &weights[kernel_val_idx]); \
        out_val = mad(in_val.s##i, kernel_val, out_val);  \
        kernel_val_idx += N;

        LOAD_KERNEL_AND_CALC(0);
        LOAD_KERNEL_AND_CALC(1);
        LOAD_KERNEL_AND_CALC(2);
        LOAD_KERNEL_AND_CALC(3);
#if N == 8 || N == 16
        LOAD_KERNEL_AND_CALC(4);
        LOAD_KERNEL_AND_CALC(5);
        LOAD_KERNEL_AND_CALC(6);
        LOAD_KERNEL_AND_CALC(7);
#if N == 16
        LOAD_KERNEL_AND_CALC(8);
        LOAD_KERNEL_AND_CALC(9);
        LOAD_KERNEL_AND_CALC(a);
        LOAD_KERNEL_AND_CALC(b);
        LOAD_KERNEL_AND_CALC(c);
        LOAD_KERNEL_AND_CALC(d);
        LOAD_KERNEL_AND_CALC(e);
        LOAD_KERNEL_AND_CALC(f);
#endif
#endif

#undef LOAD_KERNEL_AND_CALC
      }
    }
  }

#if defined(USE_RELU)
  out_val = fmax(out_val, (DATA_TYPE)0);
#endif

  VSTOREN(out_val, 0, &output[out_channel_idx]);
}
