#include <booster/booster.h>
#include <booster/helper.h>

#include "utils.h"
// #define TEST_SGECONV
#define TEST_WINOGRADF63FUSED
int test_general_conv_kernels(int output_channels, int input_channels, int input_h, int input_w, int kernel_h, int kernel_w, int stride, int pad, int nloops)
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
    // conv_param.bias_term = true;
    conv_param.bias_term = false;
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
    float* output_data_4 = (float*) malloc(sizeof(float) * conv_param.output_channels * conv_param.output_h * conv_param.output_w);

    float* bias_data = (float*) malloc(sizeof(float) * conv_param.output_channels);

    rand_fill<float>(bias_data, conv_param.output_channels);
    //Initialize
    int buffer_size = 0;
    int processed_kernel_size = 0;
#ifdef TEST_NAIVE
    printf("Initializing naive...\n");
    booster::ConvBooster naive_booster;
    naive_booster.ForceSelectAlgo(booster::NAIVE);
    naive_booster.GetBufferSize(&conv_param, &buffer_size, &processed_kernel_size);
    float* naive_processed_kernel = (float*) malloc(sizeof(float) * processed_kernel_size);
    float* naive_buffer = (float*) malloc(sizeof(float) * buffer_size);
    naive_booster.Init(&conv_param, naive_processed_kernel, kernel_data);
#endif

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

#ifdef TEST_WINOGRADF63FUSED
    printf("Initializing fused Winograd F(6x6, 3x3)...\n");
    booster::ConvBooster winogradf63fused_booster;
    winogradf63fused_booster.ForceSelectAlgo(booster::WINOGRADF63FUSED);
    winogradf63fused_booster.GetBufferSize(&conv_param, &buffer_size, &processed_kernel_size);
    float* winogradf63fused_processed_kernel = (float*) malloc(sizeof(float) * processed_kernel_size);
    float* winogradf63fused_buffer = (float*) malloc(sizeof(float) * buffer_size);
    winogradf63fused_booster.Init(&conv_param, winogradf63fused_processed_kernel, kernel_data);
#endif

    //Forward
#ifdef TEST_NAIVE
    printf("Forward naive...\n");
    naive_booster.Forward(&conv_param, output_data_1, input_data, naive_processed_kernel, naive_buffer, bias_data);
#endif

    printf("Forward im2col...\n");
    Timer tmr;
    tmr.startBench();
    for (int i = 0; i < nloops; ++i)
        im2col_booster.Forward(&conv_param, output_data_2, input_data, im2col_processed_kernel, im2col_buffer, bias_data); 
    double im2col_time_ms = tmr.endBench() / nloops;
    double im2col_gflops = conv_param.GetFLOPS() / im2col_time_ms / 1.0e6;
    printf("IM2COL spent %lfms at %5.3lfGFLOPS.\n", im2col_time_ms, im2col_gflops);

#ifdef TEST_SGECONV
    printf("Forward sgeconv...\n");
    sgeconv_booster.Forward(&conv_param, output_data_3, input_data, sgeconv_processed_kernel, sgeconv_buffer, bias_data); 
#endif

#ifdef TEST_WINOGRADF63FUSED
    printf("Forward fused Winograd F(6x6, 3x3)...\n");
    tmr.startBench();
    for (int i = 0; i < nloops; ++i)
        winogradf63fused_booster.Forward(&conv_param, output_data_4, input_data, winogradf63fused_processed_kernel, winogradf63fused_buffer, bias_data); 
    double winogradf63fused_time_ms = tmr.endBench() / nloops;
    double winogradf63fused_gflops = conv_param.GetFLOPS() / winogradf63fused_time_ms / 1.0e6;
    printf("WINOGRADF63FUSED spent %lfms at %5.3lfGFLOPS.\n", winogradf63fused_time_ms, winogradf63fused_gflops);
#endif

    // Check results
#ifdef TEST_NAIVE
    diff(output_data_1, output_data_2, conv_param.output_channels * conv_param.output_w * conv_param.output_h);
#endif
#ifdef TEST_SGECONV
    diff(output_data_2, output_data_3, conv_param.output_channels * conv_param.output_w * conv_param.output_h);
#endif
#ifdef TEST_WINOGRADF63FUSED
    diff(output_data_2, output_data_4, conv_param.output_channels , conv_param.output_w * conv_param.output_h);
#endif
    //Cleanup
#ifdef TEST_NAIVE
    free(naive_processed_kernel);
    free(naive_buffer);
#endif
    free(im2col_processed_kernel);
    free(im2col_buffer);
#ifdef TEST_SGECONV
    free(sgeconv_processed_kernel);
    free(sgeconv_buffer);
#endif
#ifdef TEST_WINOGRADF63FUSED
    free(winogradf63fused_processed_kernel);
    free(winogradf63fused_buffer);
#endif
    free(kernel_data);
    free(input_data);
    free(output_data_1);
    free(output_data_2);
    free(output_data_3);
    free(output_data_4);
    free(bias_data);
    
    return 0;
}

int main()
{
    test_general_conv_kernels(1024, 1280, 18, 18, 3, 3, 1, 1, 1);
    test_general_conv_kernels(1024, 1280, 18, 18, 3, 3, 1, 1, 20);
    test_general_conv_kernels(64, 64, 224, 224, 3, 3, 1, 1, 20);
    
    return 0;
}
