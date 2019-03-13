#include <booster/booster.h>
#include <booster/helper.h>
#include "utils.h"
// #define RUN_NAIVE_REF
// #define TEST_SGECONV
#define TEST_WINOGRADF63FUSED
#define TEST_MKLDNN


void *aligned_malloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#elif defined(_SX)
    return malloc(size);
#else
    void *p;
    return !posix_memalign(&p, alignment, size) ? p : NULL;
#endif
}

#ifdef _WIN32
void _free(void *ptr) {
    _aligned_free(ptr);
}
#else
void _free(void *ptr) {
    free(ptr);
}
#endif

int test_feather_booster(booster::ConvParam* param, booster::ConvAlgo conv_algo, int nloops)
{
    booster::ConvBooster conv_booster;
    conv_booster.ForceSelectAlgo(conv_algo);
    printf("Initializing %s...\n", conv_booster.GetAlgoName().c_str());
    int buffer_size = 0;
    int processed_kernel_size = 0;
    conv_booster.GetBufferSize(param, &buffer_size, &processed_kernel_size);
    float* conv_processed_kernel = (float*) aligned_malloc(sizeof(float) * processed_kernel_size, 64);
    float* conv_buffer = (float*) aligned_malloc(sizeof(float) * buffer_size, 64);
    param->processed_kernel_fp32 = conv_processed_kernel;
    param->common_buffer_fp32 = conv_buffer;
    conv_booster.Init(param);
    printf("Forward %s...\n", conv_booster.GetAlgoName().c_str());
    Timer tmr;
    Timer inner_tmr;
    tmr.startBench();
    for (int i = 0; i < nloops; ++i)
    {
        inner_tmr.startBench();
        conv_booster.Forward(param); 
        inner_tmr.endBench((conv_booster.GetAlgoName() + " forward:").c_str());
    }
    double time_ms = tmr.endBench() / nloops;
    double gflops = param->GetFLOPS() / time_ms / 1.0e6;
    printf("Booster %s spent %lfms at %5.3lfGFLOPS.\n", conv_booster.GetAlgoName().c_str(), time_ms, gflops);
    free(conv_processed_kernel);
    free(conv_buffer);
    return 0;
}


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
    conv_param.bias_term = true;
    conv_param.AssignOutputDim();
    conv_param.activation = booster::None;
    conv_param.LogParams("TEST");

    ThreadPool *thread_pool = new ThreadPool(4);
    conv_param.thpool = thread_pool;

    float* kernel_data = (float*) malloc(sizeof(float) * conv_param.kernel_h * conv_param.kernel_w * conv_param.input_channels * conv_param.output_channels);
    rand_fill<float>(kernel_data, conv_param.kernel_h * conv_param.kernel_w * conv_param.input_channels * conv_param.output_channels);
    
    float* input_data = (float*) malloc(sizeof(float) * conv_param.input_channels * conv_param.input_h * conv_param.input_w);
    rand_fill<float>(input_data, conv_param.input_channels * conv_param.input_h * conv_param.input_w);
    float* output_data_ref = (float*) malloc(sizeof(float) * conv_param.output_channels * conv_param.output_h * conv_param.output_w);
    float* output_data = (float*) malloc(sizeof(float) * conv_param.output_channels * conv_param.output_h * conv_param.output_w);
    
    float* bias_data = (float*) malloc(sizeof(float) * conv_param.output_channels);
    rand_fill<float>(bias_data, conv_param.output_channels);
    conv_param.input_fp32 = input_data;
    conv_param.bias_fp32 = bias_data;
    conv_param.kernel_fp32 = kernel_data;
    bool im2col_is_ref = true;
#ifdef RUN_NAIVE_REF
    im2col_is_ref = false;
    conv_param.output_fp32 = output_data_ref;
    test_feather_booster(&conv_param, booster::NAIVE, 1);
#endif
    if (booster::CheckMethodCompat(&conv_param, booster::IM2COL))
    {
        if (im2col_is_ref)
        {
            conv_param.output_fp32 = output_data_ref;
            test_feather_booster(&conv_param, booster::IM2COL, 1);
        }
        else
        {
            conv_param.output_fp32 = output_data;
            test_feather_booster(&conv_param, booster::IM2COL, nloops);
            diff(output_data_ref, output_data, conv_param.output_channels * conv_param.output_w * conv_param.output_h);
        }
    }

#ifdef TEST_WINOGRADF63FUSED
    if (booster::CheckMethodCompat(&conv_param, booster::WINOGRADF63FUSED))
    {
        conv_param.output_fp32 = output_data;
        test_feather_booster(&conv_param, booster::WINOGRADF63FUSED, nloops);
        diff(output_data_ref, output_data, conv_param.output_channels * conv_param.output_w * conv_param.output_h);
    }
#endif
#ifdef TEST_MKLDNN
    if (booster::CheckMethodCompat(&conv_param, booster::MKLDNN))
    {
        conv_param.output_fp32 = output_data;
        test_feather_booster(&conv_param, booster::MKLDNN, nloops);
        diff(output_data_ref, output_data, conv_param.output_channels * conv_param.output_w * conv_param.output_h);
    }
#endif

    //Cleanup
    delete thread_pool;
    free(kernel_data);
    free(input_data);
    free(output_data_ref);
    free(output_data);
    free(bias_data);
    
    return 0;
}

int main()
{
    // test_general_conv_kernels(4, 4, 12, 12, 3, 3, 1, 1, 1);
    // test_general_conv_kernels(16, 32, 8, 8, 3, 3, 1, 1, 1);
     test_general_conv_kernels(1024, 1280, 18, 18, 3, 3, 1, 1, 100);
    // test_general_conv_kernels(128, 128, 18, 18, 3, 3, 1, 1, 1);
    test_general_conv_kernels(64, 64, 224, 224, 3, 3, 1, 1, 100);
    test_general_conv_kernels(32, 3, 576, 576, 3, 3, 1, 1, 30);
    test_general_conv_kernels(256, 512, 36, 36, 1, 1, 1, 1, 30);
    // test_general_conv_kernels(128, 128, 112, 112, 3, 3, 1, 1, 30);
    
    return 0;
}
