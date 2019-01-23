#include <common.h>

DATA_TYPEN cell_merge(DATA_TYPEN left,
                         DATA_TYPEN right,
                         const int mod,
                         const bool reversed) {
  if (!reversed) {
    switch (mod) {
#if N == 4
      case 1:return (DATA_TYPEN)(left.x, right.x, right.y, right.z);
      case 2:return (DATA_TYPEN)(left.x, left.y, right.x, right.y);
      case 3:return (DATA_TYPEN)(left.x, left.y, left.z, right.x);
      default:return (DATA_TYPEN) 0;
#endif
#if N == 8
      case 1:return (DATA_TYPEN)(left.s0, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6);
      case 2:return (DATA_TYPEN)(left.s0, left.s1, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5);
      case 3:return (DATA_TYPEN)(left.s0, left.s1, left.s2, right.s0, right.s1, right.s2, right.s3, right.s4);
      case 4:return (DATA_TYPEN)(left.s0, left.s1, left.s2, left.s3, right.s0, right.s1, right.s2, right.s3);
      case 5:return (DATA_TYPEN)(left.s0, left.s1, left.s2, left.s3, left.s4, right.s0, right.s1, right.s2);
      case 6:return (DATA_TYPEN)(left.s0, left.s1, left.s2, left.s3, left.s4, left.s5, right.s0, right.s1);
      case 7:return (DATA_TYPEN)(left.s0, left.s1, left.s2, left.s3, left.s4, left.s5, left.s6, right.s0);
      default:return (DATA_TYPEN) 0;
#endif
#if N == 16
      case 1:return (DATA_TYPEN)(left.s0, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6, right.s7, right.s8, right.s9, right.sa, right.sb, right.sc, right.sd, right.se);
      case 2:return (DATA_TYPEN)(left.s0, left.s1, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6, right.s7, right.s8, right.s9, right.sa, right.sb, right.sc, right.sd);
      case 3:return (DATA_TYPEN)(left.s0, left.s1, left.s2, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6, right.s7, right.s8, right.s9, right.sa, right.sb, right.sc);
      case 4:return (DATA_TYPEN)(left.s0, left.s1, left.s2, left.s3, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6, right.s7, right.s8, right.s9, right.sa, right.sb);
      case 5:return (DATA_TYPEN)(left.s0, left.s1, left.s2, left.s3, left.s4, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6, right.s7, right.s8, right.s9, right.sa);
      case 6:return (DATA_TYPEN)(left.s0, left.s1, left.s2, left.s3, left.s4, left.s5, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6, right.s7, right.s8, right.s9);
      case 7:return (DATA_TYPEN)(left.s0, left.s1, left.s2, left.s3, left.s4, left.s5, left.s6, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6, right.s7, right.s8);
      case 8:return (DATA_TYPEN)(left.s0, left.s1, left.s2, left.s3, left.s4, left.s5, left.s6, left.s7, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6, right.s7);
      case 9:return (DATA_TYPEN)(left.s0, left.s1, left.s2, left.s3, left.s4, left.s5, left.s6, left.s7, left.s8, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6);
      case 10:return (DATA_TYPEN)(left.s0, left.s1, left.s2, left.s3, left.s4, left.s5, left.s6, left.s7, left.s8, left.s9, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5);
      case 11:return (DATA_TYPEN)(left.s0, left.s1, left.s2, left.s3, left.s4, left.s5, left.s6, left.s7, left.s8, left.s9, left.sa, right.s0, right.s1, right.s2, right.s3, right.s4);
      case 12:return (DATA_TYPEN)(left.s0, left.s1, left.s2, left.s3, left.s4, left.s5, left.s6, left.s7, left.s8, left.s9, left.sa, left.sb, right.s0, right.s1, right.s2, right.s3);
      case 13:return (DATA_TYPEN)(left.s0, left.s1, left.s2, left.s3, left.s4, left.s5, left.s6, left.s7, left.s8, left.s9, left.sa, left.sb, left.sc, right.s0, right.s1, right.s2);
      case 14:return (DATA_TYPEN)(left.s0, left.s1, left.s2, left.s3, left.s4, left.s5, left.s6, left.s7, left.s8, left.s9, left.sa, left.sb, left.sc, left.sd, right.s0, right.s1);
      case 15:return (DATA_TYPEN)(left.s0, left.s1, left.s2, left.s3, left.s4, left.s5, left.s6, left.s7, left.s8, left.s9, left.sa, left.sb, left.sc, left.sd, left.se, right.s0);
      default:return (DATA_TYPEN) 0;
#endif
    }
  } else {
    switch (mod) {
#if N == 4
      case 1:return (DATA_TYPEN)(left.w, right.x, right.y, right.z);
      case 2:return (DATA_TYPEN)(left.z, left.w, right.x, right.y);
      case 3:return (DATA_TYPEN)(left.y, left.z, left.w, right.x);
      default:return (DATA_TYPEN) 0;
#endif
#if N == 8
      case 1:return (DATA_TYPEN)(left.s7, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6);
      case 2:return (DATA_TYPEN)(left.s6, left.s7, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5);
      case 3:return (DATA_TYPEN)(left.s5, left.s6, left.s7, right.s0, right.s1, right.s2, right.s3, right.s4);
      case 4:return (DATA_TYPEN)(left.s4, left.s5, left.s6, left.s7, right.s0, right.s1, right.s2, right.s3);
      case 5:return (DATA_TYPEN)(left.s3, left.s4, left.s5, left.s6, left.s7, right.s0, right.s1, right.s2);
      case 6:return (DATA_TYPEN)(left.s2, left.s3, left.s4, left.s5, left.s6, left.s7, right.s0, right.s1);
      case 7:return (DATA_TYPEN)(left.s1, left.s2, left.s3, left.s4, left.s5, left.s6, left.s7, right.s0);
      default:return (DATA_TYPEN) 0;
#endif
#if N == 16
      case 1:return (DATA_TYPEN)(left.sf, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6, right.s7, right.s8, right.s9, right.sa, right.sb, right.sc, right.sd, right.se);
      case 2:return (DATA_TYPEN)(left.se, left.sf, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6, right.s7, right.s8, right.s9, right.sa, right.sb, right.sc, right.sd);
      case 3:return (DATA_TYPEN)(left.sd, left.se, left.sf, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6, right.s7, right.s8, right.s9, right.sa, right.sb, right.sc);
      case 4:return (DATA_TYPEN)(left.sc, left.sd, left.se, left.sf, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6, right.s7, right.s8, right.s9, right.sa, right.sb);
      case 5:return (DATA_TYPEN)(left.sb, left.sc, left.sd, left.se, left.sf, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6, right.s7, right.s8, right.s9, right.sa);
      case 6:return (DATA_TYPEN)(left.sa, left.sb, left.sc, left.sd, left.se, left.sf, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6, right.s7, right.s8, right.s9);
      case 7:return (DATA_TYPEN)(left.s9, left.sa, left.sb, left.sc, left.sd, left.se, left.sf, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6, right.s7, right.s8);
      case 8:return (DATA_TYPEN)(left.s8, left.s9, left.sa, left.sb, left.sc, left.sd, left.se, left.sf, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6, right.s7);
      case 9:return (DATA_TYPEN)(left.s7, left.s8, left.s9, left.sa, left.sb, left.sc, left.sd, left.se, left.sf, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5, right.s6);
      case 10:return (DATA_TYPEN)(left.s6, left.s7, left.s8, left.s9, left.sa, left.sb, left.sc, left.sd, left.se, left.sf, right.s0, right.s1, right.s2, right.s3, right.s4, right.s5);
      case 11:return (DATA_TYPEN)(left.s5, left.s6, left.s7, left.s8, left.s9, left.sa, left.sb, left.sc, left.sd, left.se, left.sf, right.s0, right.s1, right.s2, right.s3, right.s4);
      case 12:return (DATA_TYPEN)(left.s4, left.s5, left.s6, left.s7, left.s8, left.s9, left.sa, left.sb, left.sc, left.sd, left.se, left.sf, right.s0, right.s1, right.s2, right.s3);
      case 13:return (DATA_TYPEN)(left.s3, left.s4, left.s5, left.s6, left.s7, left.s8, left.s9, left.sa, left.sb, left.sc, left.sd, left.se, left.sf, right.s0, right.s1, right.s2);
      case 14:return (DATA_TYPEN)(left.s2, left.s3, left.s4, left.s5, left.s6, left.s7, left.s8, left.s9, left.sa, left.sb, left.sc, left.sd, left.se, left.sf, right.s0, right.s1);
      case 15:return (DATA_TYPEN)(left.s1, left.s2, left.s3, left.s4, left.s5, left.s6, left.s7, left.s8, left.s9, left.sa, left.sb, left.sc, left.sd, left.se, left.sf, right.s0);
      default:return (DATA_TYPEN) 0;
#endif
    }
  }
}

// N = 4, 8, or 16, which is the channel block size.
__kernel void concat(__global const DATA_TYPE* restrict in0, /* [ih, iw, c] */
                        __global const DATA_TYPE* restrict in1,
                        __global DATA_TYPE* restrict out,
                        __private const int out_height,
                        __private const int out_width,
                        __private const int out_channels,
                        __private const int in0_channels) {
  const int out_height_idx = get_global_id(0);
  const int out_width_idx = get_global_id(1);
  if (out_height_idx >= out_height || out_width_idx >= out_width) return;
  const int channel_block_idx = get_global_id(2);
  const int channel_idx = mul24(channel_block_idx, N);


  DATA_TYPEN out_val = 0;
#if defined(DIVISIBLE)
  if (channel_idx < in0_channels) {
    const int val_idx = mad24(mad24(out_height_idx, out_width, out_width_idx), in0_channels, channel_idx);
    out_val = VLOADN(0, in0 + val_idx);
  } else {
    const int val_idx = mad24(mad24(out_height_idx, out_width, out_width_idx), out_channels-in0_channels, channel_idx-in0_channels);
    out_val = VLOADN(0, in1 + val_idx);
  }
#else
  if (channel_idx + N < in0_channels) {
    const int val_idx = mad24(mad24(out_height_idx, out_width, out_width_idx), in0_channels, channel_idx);
    out_val = VLOADN(0, in0 + val_idx);

  } else if (channel_idx >= in0_channels) {
    int val_idx = mad24(mad24(out_height_idx, out_width, out_width_idx), out_channels-in0_channels, channel_idx-in0_channels);
    DATA_TYPEN out_val0 = VLOADN(0, in1 + val_idx);
    val_idx += N;
    DATA_TYPEn out_val1 = VLOADN(0, in1 + val_idx);

    out_val = cell_merge(out_val0, out_val1, in0_channels % N, true);

  } else {
    const int val_idx0 = mad24(mad24(out_height_idx, out_width, out_width_idx), in0_channels, in0_channels);
    DATA_TYPEN out_val0 = VLOADN(0, in0 + val_idx);
    DATA_TYPEn out_val1 = VLOADN(0, in1);

    out_val = cell_merge(out_val0, out_val1, in0_channels % N, false);
  }

#endif

#if defined(USE_RELU)
  out_val = fmax(out_val, (DATA_TYPE)0);
#endif

  const int out_val_idx = mad24(mad24(out_height_idx, out_width, out_width_idx), out_channels, channel_idx);

  VSTOREN(out_val, 0, out + out_val_idx);





}
