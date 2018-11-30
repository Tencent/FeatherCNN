const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
#define channelsInGroup 4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void concat_1_4O4(__global half* restrict inputMatrix,
                      __global half* restrict outputMatrix,
                      const int outputHeight,
                      const int outputWidth,
                      const int storeChannel,
                      const int outputChannel
                     ){

    int outputRowIndex = get_global_id(0);
    int outputColumnIndex = get_global_id(1);
    int outputChannelIndex = get_global_id(2);

    if(outputRowIndex >= outputHeight || outputColumnIndex >= outputWidth){
        return;
    }
    half originValue = (half)(0.0f);
    int channelSize = outputHeight * outputWidth;
    int channelGroupSize = 4 * outputHeight * outputWidth;
    int c_i = outputChannelIndex % 4;
    int c_o = (outputChannelIndex + storeChannel) % 4;

    if(outputChannelIndex < outputChannel)
    {
        int origin = outputChannelIndex/4 * channelGroupSize + outputRowIndex * outputWidth * 4 + outputColumnIndex * 4 + c_i;
        originValue = inputMatrix[origin];
    }
    int output = (storeChannel + outputChannelIndex)/4 * channelGroupSize + outputRowIndex * outputWidth * 4 + outputColumnIndex * 4 + c_o;
    outputMatrix[output] = originValue;
}
