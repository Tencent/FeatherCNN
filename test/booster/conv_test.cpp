#include <booster/booster.h>
#include <booster/helper.h>

#include "utils.h"
// #define TEST_SGECONV
int test_general_conv_kernels(int output_channels, int input_channels, int input_h, int input_w, int kernel_h, int kernel_w, int stride, int pad)
{
    booster::ConvParam conv_param;
    // conv_param
    conv_param.output_channels = output_channels;
    conv_param.input_channels = input_channels;
    conv_param.input_h = input_h;
    conv_param.input_w = input_w;
    conv_param.kernel_h = kernel_h;
    conv_param.kernel_w = kernel_w;
    conv_param.stride_h = stride;
    conv_param.stride_w = stride;
    conv_param.pad_left = pad;
    conv_param.pad_bottom = pad;
    conv_param.pad_right = pad;
    conv_param.pad_top = pad;
    conv_param.group = 1;
    conv_param.bias_term = true;
    conv_param.AssignOutputDim();
    conv_param.activation = booster::None;
    conv_param.LogParams("TEST");

    float* kernel_data = (float*) malloc(sizeof(float) * conv_param.kernel_h * conv_param.kernel_w * conv_param.input_channels * conv_param.output_channels);
    rand_fill<float>(kernel_data, conv_param.kernel_h * conv_param.kernel_w * conv_param.input_channels * conv_param.output_channels);
    // float* naive_processed_kernel = (float*) malloc(sizeof(float) * conv_param.kernel_h * conv_param.kernel_w * conv_param.input_channels * conv_param.output_channels);
    // float* im2col_processed_kernel = (float*) malloc(sizeof(float) * conv_param.kernel_h * conv_param.kernel_w * conv_param.input_channels * conv_param.output_channels);
    
    float* input_data = (float*) malloc(sizeof(float) * conv_param.input_channels * conv_param.input_h * conv_param.input_w);
    rand_fill<float>(input_data, conv_param.input_channels * conv_param.input_h * conv_param.input_w);
    float* output_data_1 = (float*) malloc(sizeof(float) * conv_param.output_channels * conv_param.output_h * conv_param.output_w);
    float* output_data_2 = (float*) malloc(sizeof(float) * conv_param.output_channels * conv_param.output_h * conv_param.output_w);
    float* output_data_3 = (float*) malloc(sizeof(float) * conv_param.output_channels * conv_param.output_h * conv_param.output_w);

    float* bias_data = (float*) malloc(sizeof(float) * conv_param.output_channels);

    rand_fill<float>(bias_data, conv_param.output_channels);
    //Initialize
    printf("Initializing naive...\n");
    booster::ConvBooster naive_booster;
    naive_booster.ForceSelectAlgo(booster::NAIVE);
    int buffer_size = 0;
    int processed_kernel_size = 0;
    naive_booster.GetBufferSize(&conv_param, &buffer_size, &processed_kernel_size);
    float* naive_processed_kernel = (float*) malloc(sizeof(float) * processed_kernel_size);
    float* naive_buffer = (float*) malloc(sizeof(float) * buffer_size);
    naive_booster.Init(&conv_param, naive_processed_kernel, kernel_data);
    
    printf("Initializing im2col...\n");
    booster::ConvBooster im2col_booster;
    im2col_booster.ForceSelectAlgo(booster::IM2COL);
    im2col_booster.GetBufferSize(&conv_param, &buffer_size, &processed_kernel_size);
    float* im2col_processed_kernel = (float*) malloc(sizeof(float) * processed_kernel_size);
    float* im2col_buffer = (float*) malloc(sizeof(float) * buffer_size);
    im2col_booster.Init(&conv_param, im2col_processed_kernel, kernel_data);

    
#ifdef TEST_SGECONV
    printf("Initializing sgeconv...\n");
    booster::ConvBooster sgeconv_booster;
    sgeconv_booster.ForceSelectAlgo(booster::SGECONV);
    sgeconv_booster.GetBufferSize(&conv_param, &buffer_size, &processed_kernel_size);
    float* sgeconv_processed_kernel = (float*) malloc(sizeof(float) * processed_kernel_size);
    float* sgeconv_buffer = (float*) malloc(sizeof(float) * buffer_size);
    sgeconv_booster.Init(&conv_param, sgeconv_processed_kernel, kernel_data);
#endif
    
    //Forward
    printf("Forward naive...\n");
    naive_booster.Forward(&conv_param, output_data_1, input_data, naive_processed_kernel, naive_buffer, bias_data);
    printf("Forward im2col...\n");
    im2col_booster.Forward(&conv_param, output_data_2, input_data, im2col_processed_kernel, im2col_buffer, bias_data); 

#ifdef TEST_SGECONV
    printf("Forward sgeconv...\n");
    sgeconv_booster.Forward(&conv_param, output_data_3, input_data, sgeconv_processed_kernel, sgeconv_buffer, bias_data); 
#endif

    //Check results
    diff(output_data_1, output_data_2, conv_param.output_channels * conv_param.output_w * conv_param.output_h);
#ifdef TEST_SGECONV
    diff(output_data_1, output_data_3, conv_param.output_channels * conv_param.output_w * conv_param.output_h);
#endif
    //Cleanup
    free(naive_processed_kernel);
    free(naive_buffer);
    free(im2col_processed_kernel);
    free(im2col_buffer);
#ifdef TEST_SGECONV
    free(sgeconv_processed_kernel);
    free(sgeconv_buffer);
#endif
    free(kernel_data);
    free(input_data);
    free(output_data_1);
    free(output_data_2);
    free(output_data_3);
    free(bias_data);
    
    return 0;
}

int main()
{
    test_general_conv_kernels(64, 32, 224, 224, 5, 5, 4, 1);
    return 0;
}
