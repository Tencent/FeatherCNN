__kernel void winograd_input_trans_2x2(
                          __global const half* restrict input,   /* [h, w, c] */
                          __global half* restrict input_trans,
                          __private const int input_channels,
                          __private const int input_height,
                          __private const int input_width,
                          __private const int blk_row_size,
                          __private const int blk_col_size) {

    const int blk_row_idx = get_global_id(0);
    const int blk_col_idx = get_global_id(1);
    if (blk_row_idx >= blk_row_size || blk_col_idx >= blk_col_size) return;
    const int channel_idx = get_global_id(2) << 2;

    half4 d00, d01, d02, d03;
    half4 d10, d11, d12, d13;
    half4 d20, d21, d22, d23;
    half4 d30, d31, d32, d33;

    half4 w00, w01, w02, w03;
    half4 w10, w11, w12, w13;
    half4 w20, w21, w22, w23;
    half4 w30, w31, w32, w33;

    int val_idx = ((2 * blk_row_idx) * input_width + (2 * blk_col_idx)) * input_channels + channel_idx;
    const int width_blk_size = input_width * input_channels;

    d00 = vload4(0, input[val_idx]);
    val_idx += input_channels;
    d01 = vload4(0, input[val_idx]);
    val_idx += input_channels;
    d02 = vload4(0, input[val_idx]);
    val_idx += input_channels;
    d03 = vload4(0, input[val_idx]);
    val_idx += input_channels - 3 * input_channels; // back to original idx and move a width block

    d10 = vload4(0, input[val_idx]);
    val_idx += input_channels;
    d11 = vload4(0, input[val_idx]);
    val_idx += input_channels;
    d12 = vload4(0, input[val_idx]);
    val_idx += input_channels;
    d13 = vload4(0, input[val_idx]);
    val_idx += input_channels - 3 * input_channels; // back to original idx and move a width block

    d20 = vload4(0, input[val_idx]);
    val_idx += input_channels;
    d21 = vload4(0, input[val_idx]);
    val_idx += input_channels;
    d22 = vload4(0, input[val_idx]);
    val_idx += input_channels;
    d23 = vload4(0, input[val_idx]);
    val_idx += input_channels - 3 * input_channels; // back to original idx and move a width block

    d30 = vload4(0, input[val_idx]);
    val_idx += input_channels;
    d31 = vload4(0, input[val_idx]);
    val_idx += input_channels;
    d32 = vload4(0, input[val_idx]);
    val_idx += input_channels;
    d33 = vload4(0, input[val_idx]);

    w00 = d00 - d20, w01 = d01 - d21, w02 = d02 - d22, w03 = d03 - d23;
    w10 = d10 + d20, w11 = d11 + d21, w12 = d12 + d22, w13 = d13 + d23;
    w20 = d20 - d10, w21 = d21 - d11, w22 = d22 - d12, w23 = d23 - d13;
    w30 = d30 - d10, w31 = d31 - d11, w32 = d32 - d12, w33 = d33 - d13;

    d00 = w00 - w02, d01 = w10 - w12, d02 = w20 - w22, d03 = w30 - w32;
    d10 = w01 + w02, d11 = w11 + w12, d12 = w21 + w22, d13 = w31 + w32;
    d20 = w02 - w01, d21 = w12 - w11, d22 = w22 - w21, d23 = w32 - w31;
    d30 = w01 - w03, d31 = w11 - w13, d32 = w21 - w23, d33 = w31 - w33;


}
