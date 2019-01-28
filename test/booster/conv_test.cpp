#include <booster/booster.h>
#include <booster/helper.h>

#include "utils.h"
// #define TEST_SGECONV
#define TEST_WINOGRADF63FUSED
// #define MKLDNN_STANDALONE_TEST
#define TEST_MKLDNN


#ifdef MKLDNN_STANDALONE_TEST
#include <mkldnn.hpp>
#include <mm_malloc.h>

void test_mkldnn_split(booster::ConvParam *conv_param, float* output, float* input, float* kernel, const int nloop)
{
    float* dummy_bias = (float*) _mm_malloc(sizeof(float) * conv_param->output_channels, 32);
    memset(dummy_bias, 0, sizeof(float) * conv_param->output_channels);
    
    auto cpu_engine = mkldnn::engine(mkldnn::engine::cpu, 0);

    mkldnn::memory::dims conv_src_tz = {1, conv_param->input_channels, conv_param->input_h, conv_param->input_w};
    mkldnn::memory::dims conv_weights_tz = {conv_param->output_channels, conv_param->input_channels, conv_param->kernel_h, conv_param->kernel_w};
    mkldnn::memory::dims conv_dst_tz = {1, conv_param->output_channels, conv_param->output_h, conv_param->output_w};
    mkldnn::memory::dims conv_bias_tz = {conv_param->output_channels};
    mkldnn::memory::dims conv_strides = {1, 1};
    mkldnn::memory::dims conv_padding = {conv_param->pad_left, conv_param->pad_top};

    auto conv_user_src_memory = mkldnn::memory({{{conv_src_tz},
                                                 mkldnn::memory::data_type::f32,
                                                 mkldnn::memory::format::nchw},
                                                cpu_engine},
                                               input);
    auto conv_user_dst_memory = mkldnn::memory({{{conv_dst_tz},
                                                 mkldnn::memory::data_type::f32,
                                                 mkldnn::memory::format::nchw},
                                                cpu_engine},
                                                output);
    auto conv_user_weights_memory = mkldnn::memory({{{conv_weights_tz},
                                                     mkldnn::memory::data_type::f32,
                                                     mkldnn::memory::format::oihw},
                                                    cpu_engine},
                                                   kernel);
    auto conv_user_bias_memory = mkldnn::memory({{{conv_bias_tz},
                                                     mkldnn::memory::data_type::f32,
                                                     mkldnn::memory::format::x},
                                                    cpu_engine},
                                                   dummy_bias);

    auto conv_src_md = mkldnn::memory::desc({conv_src_tz},
                                            mkldnn::memory::data_type::f32,
                                            mkldnn::memory::format::any);
    auto conv_bias_md = mkldnn::memory::desc({conv_bias_tz},
                                            mkldnn::memory::data_type::f32,
                                            mkldnn::memory::format::any);

    auto conv_weights_md = mkldnn::memory::desc({conv_weights_tz},
                                                mkldnn::memory::data_type::f32,
                                                mkldnn::memory::format::any);

    auto conv_dst_md = mkldnn::memory::desc({conv_dst_tz},
                                            mkldnn::memory::data_type::f32,
                                            mkldnn::memory::format::any);

    auto conv_desc = mkldnn::convolution_forward::desc(mkldnn::prop_kind::forward,
                                                       mkldnn::convolution_direct,
                                                       conv_src_md,
                                                       conv_weights_md,
                                                       conv_bias_md,
                                                       conv_dst_md,
                                                       conv_strides,
                                                       conv_padding,
                                                       conv_padding,
                                                       mkldnn::padding_kind::zero);
    //Bind Engine
    auto conv_prim_desc = mkldnn::convolution_forward::primitive_desc(conv_desc, cpu_engine);
    
    std::vector<mkldnn::primitive> input_ops;
    std::vector<mkldnn::primitive> forward_ops;
    std::vector<mkldnn::primitive> output_ops;
    auto conv_src_memory = conv_user_src_memory;
    if (mkldnn::memory::primitive_desc(conv_prim_desc.src_primitive_desc()) !=
        conv_user_src_memory.get_primitive_desc())
    {
        conv_src_memory = mkldnn::memory(conv_prim_desc.src_primitive_desc());
        input_ops.push_back(mkldnn::reorder(conv_user_src_memory, conv_src_memory));
    }

    auto conv_weights_memory = conv_user_weights_memory;
    if (mkldnn::memory::primitive_desc(conv_prim_desc.weights_primitive_desc()) !=
        conv_user_weights_memory.get_primitive_desc())
    {
        conv_weights_memory =
            mkldnn::memory(conv_prim_desc.weights_primitive_desc());

        input_ops.push_back(mkldnn::reorder(conv_user_weights_memory,
                                      conv_weights_memory));
    }
    auto conv_dst_memory = conv_user_dst_memory;
    if (mkldnn::memory::primitive_desc(conv_prim_desc.dst_primitive_desc()) !=
        conv_user_dst_memory.get_primitive_desc())
    {
        conv_dst_memory =
            mkldnn::memory(conv_prim_desc.dst_primitive_desc());
        output_ops.push_back(mkldnn::reorder(conv_dst_memory,
                                             conv_user_dst_memory));
    }

    forward_ops.push_back(mkldnn::convolution_forward(conv_prim_desc, conv_src_memory,
                                              conv_weights_memory, conv_user_bias_memory, conv_dst_memory));
    // forward_ops.push_back(mkldnn::reorder(conv_user_dst_memory,
    //                                          conv_dst_memory));

    mkldnn::stream(mkldnn::stream::kind::eager).submit(input_ops).wait();
    Timer timer;
    timer.startBench();
    Timer mkldnn_tmr;
    for(int i = 0; i < nloop; ++i)
    {
        mkldnn_tmr.startBench();
        mkldnn::stream(mkldnn::stream::kind::eager).submit(forward_ops).wait();
        mkldnn_tmr.endBench("MKLDNN latency: ");
    }
    timer.endBench("MKLDNN Avg Latency: ", (double)nloop);

    mkldnn::stream(mkldnn::stream::kind::eager).submit(output_ops).wait();
}

void test_mkldnn_pipeline(booster::ConvParam *conv_param, float* output, float* input, float* kernel, const int nloop)
{
    float* dummy_bias = (float*) _mm_malloc(sizeof(float) * conv_param->output_channels, 32);
    memset(dummy_bias, 0, sizeof(float) * conv_param->output_channels);
    
    auto cpu_engine = mkldnn::engine(mkldnn::engine::cpu, 0);

    mkldnn::memory::dims conv_src_tz = {1, conv_param->input_channels, conv_param->input_h, conv_param->input_w};
    mkldnn::memory::dims conv_weights_tz = {conv_param->output_channels, conv_param->input_channels, conv_param->kernel_h, conv_param->kernel_w};
    mkldnn::memory::dims conv_dst_tz = {1, conv_param->output_channels, conv_param->output_h, conv_param->output_w};
    mkldnn::memory::dims conv_bias_tz = {conv_param->output_channels};
    mkldnn::memory::dims conv_strides = {1, 1};
    mkldnn::memory::dims conv_padding = {conv_param->pad_left, conv_param->pad_top};

    auto conv_user_src_memory = mkldnn::memory({{{conv_src_tz},
                                                 mkldnn::memory::data_type::f32,
                                                 mkldnn::memory::format::nchw},
                                                cpu_engine},
                                               input);
    auto conv_user_dst_memory = mkldnn::memory({{{conv_dst_tz},
                                                 mkldnn::memory::data_type::f32,
                                                 mkldnn::memory::format::nchw},
                                                cpu_engine},
                                                output);
    auto conv_user_weights_memory = mkldnn::memory({{{conv_weights_tz},
                                                     mkldnn::memory::data_type::f32,
                                                     mkldnn::memory::format::oihw},
                                                    cpu_engine},
                                                   kernel);
    auto conv_user_bias_memory = mkldnn::memory({{{conv_bias_tz},
                                                     mkldnn::memory::data_type::f32,
                                                     mkldnn::memory::format::x},
                                                    cpu_engine},
                                                   dummy_bias);

    auto conv_src_md = mkldnn::memory::desc({conv_src_tz},
                                            mkldnn::memory::data_type::f32,
                                            mkldnn::memory::format::any);
    auto conv_bias_md = mkldnn::memory::desc({conv_bias_tz},
                                            mkldnn::memory::data_type::f32,
                                            mkldnn::memory::format::any);

    auto conv_weights_md = mkldnn::memory::desc({conv_weights_tz},
                                                mkldnn::memory::data_type::f32,
                                                mkldnn::memory::format::any);

    auto conv_dst_md = mkldnn::memory::desc({conv_dst_tz},
                                            mkldnn::memory::data_type::f32,
                                            mkldnn::memory::format::any);

    auto conv_desc = mkldnn::convolution_forward::desc(mkldnn::prop_kind::forward,
                                                       mkldnn::convolution_direct,
                                                       conv_src_md,
                                                       conv_weights_md,
                                                       conv_bias_md,
                                                       conv_dst_md,
                                                       conv_strides,
                                                       conv_padding,
                                                       conv_padding,
                                                       mkldnn::padding_kind::zero);
    //Bind Engine
    auto conv_prim_desc = mkldnn::convolution_forward::primitive_desc(conv_desc, cpu_engine);
    
    std::vector<mkldnn::primitive> input_ops;
    std::vector<mkldnn::primitive> forward_ops;
    std::vector<mkldnn::primitive> output_ops;
    auto conv_src_memory = conv_user_src_memory;
    if (mkldnn::memory::primitive_desc(conv_prim_desc.src_primitive_desc()) !=
        conv_user_src_memory.get_primitive_desc())
    {
        conv_src_memory = mkldnn::memory(conv_prim_desc.src_primitive_desc());
        forward_ops.push_back(mkldnn::reorder(conv_user_src_memory, conv_src_memory));
    }

    auto conv_weights_memory = conv_user_weights_memory;
    if (mkldnn::memory::primitive_desc(conv_prim_desc.weights_primitive_desc()) !=
        conv_user_weights_memory.get_primitive_desc())
    {
        conv_weights_memory =
            mkldnn::memory(conv_prim_desc.weights_primitive_desc());

        input_ops.push_back(mkldnn::reorder(conv_user_weights_memory,
                                      conv_weights_memory));
    }

    auto conv_dst_memory = conv_user_dst_memory;

    if (mkldnn::memory::primitive_desc(conv_prim_desc.dst_primitive_desc()) !=
        conv_user_dst_memory.get_primitive_desc())
    {
        conv_dst_memory =
            mkldnn::memory(conv_prim_desc.dst_primitive_desc());
        
    }

    forward_ops.push_back(mkldnn::convolution_forward(conv_prim_desc, conv_src_memory,
                                              conv_weights_memory, conv_user_bias_memory, conv_dst_memory));
    
    if (mkldnn::memory::primitive_desc(conv_prim_desc.dst_primitive_desc()) !=
        conv_user_dst_memory.get_primitive_desc())
        forward_ops.push_back(mkldnn::reorder(conv_dst_memory,
                                             conv_user_dst_memory));

    // forward_ops.push_back(mkldnn::reorder(conv_user_dst_memory,
    //                                          conv_dst_memory));
    mkldnn::stream(mkldnn::stream::kind::eager).submit(input_ops).wait();
    Timer timer;
    timer.startBench();
    Timer mkldnn_tmr;
    for(int i = 0; i < nloop; ++i)
    {
        mkldnn_tmr.startBench();
        mkldnn::stream(mkldnn::stream::kind::eager).submit(forward_ops).wait();
        mkldnn_tmr.endBench("MKLDNN latency: ");
    }
    timer.endBench("MKLDNN Avg Latency: ", (double)nloop);
    // mkldnn::stream(mkldnn::stream::kind::eager).submit(output_ops).wait();
}
#endif


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
    float* output_data_4 = (float*) malloc(sizeof(float) * conv_param.output_channels * conv_param.output_h * conv_param.output_w);
    float* output_data_5 = (float*) malloc(sizeof(float) * conv_param.output_channels * conv_param.output_h * conv_param.output_w);

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
#ifdef TEST_MKLDNN
    printf("Initializing MKLDNN...\n");
    booster::ConvBooster mkldnn_booster;
    conv_param.input_fp32 = input_data;
    conv_param.output_fp32 = output_data_5;
    conv_param.bias_fp32 = bias_data;
    mkldnn_booster.ForceSelectAlgo(booster::ConvAlgo::MKLDNN);
    mkldnn_booster.Init(&conv_param, NULL, kernel_data);
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
    // print_floats(output_data_2, conv_param.output_channels * conv_param.output_h, conv_param.output_w);
#ifdef TEST_SGECONV
    printf("Forward sgeconv...\n");
    sgeconv_booster.Forward(&conv_param, output_data_3, input_data, sgeconv_processed_kernel, sgeconv_buffer, bias_data); 
#endif

#ifdef TEST_WINOGRADF63FUSED
    printf("Forward fused Winograd F(6x6, 3x3)...\n");
    tmr.startBench();
    Timer inner_tmr;
    for (int i = 0; i < nloops; ++i)
    {
        inner_tmr.startBench();
        winogradf63fused_booster.Forward(&conv_param, output_data_4, input_data, winogradf63fused_processed_kernel, winogradf63fused_buffer, bias_data); 
        inner_tmr.endBench("WINOGRADF63FUSED spent");
    }
    double winogradf63fused_time_ms = tmr.endBench() / nloops;
    double winogradf63fused_gflops = conv_param.GetFLOPS() / winogradf63fused_time_ms / 1.0e6;
    printf("WINOGRADF63FUSED spent %lfms at %5.3lfGFLOPS.\n", winogradf63fused_time_ms, winogradf63fused_gflops);
    diff(output_data_2, output_data_4, conv_param.output_channels * conv_param.output_w * conv_param.output_h);
#endif

    printf("Forward MKLDNN...\n");
    tmr.startBench();
    for (int i = 0; i < nloops; ++i)
    {
        inner_tmr.startBench();
        mkldnn_booster.Forward(&conv_param, NULL, NULL, NULL, NULL, NULL);
        inner_tmr.endBench("MKLDNN spent");
    }
    double mkldnn_time_ms = tmr.endBench() / nloops;
    double mkldnn_gflops = conv_param.GetFLOPS() / mkldnn_time_ms / 1.0e6;
    printf("MKLDNN spent %lfms at %5.3lfGFLOPS.\n", mkldnn_time_ms, mkldnn_gflops);
    diff(output_data_2, output_data_5, conv_param.output_channels * conv_param.output_w * conv_param.output_h);

    // Check results
#ifdef TEST_NAIVE
    diff(output_data_1, output_data_2, conv_param.output_channels * conv_param.output_w * conv_param.output_h);
#endif
#ifdef TEST_SGECONV
    diff(output_data_2, output_data_3, conv_param.output_channels * conv_param.output_w * conv_param.output_h);
#endif
#ifdef TEST_WINOGRADF63FUSED
    
#endif
#ifdef MKLDNN_STANDALONE_TEST
    // test_mkldnn_pipeline(&conv_param, output_data_5, input_data, kernel_data, nloops);
    diff(output_data_2, output_data_5, conv_param.output_channels * conv_param.output_w * conv_param.output_h);
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
    free(output_data_5);
    free(bias_data);
    
    return 0;
}

int main()
{
    // test_general_conv_kernels(4, 4, 12, 12, 3, 3, 1, 1, 1);
    // test_general_conv_kernels(4, 8, 8, 8, 3, 3, 1, 1, 1);
    test_general_conv_kernels(1024, 1280, 18, 18, 3, 3, 1, 1, 30);
    // test_general_conv_kernels(192, 1280, 18, 18, 3, 3, 1, 1, 1);
    test_general_conv_kernels(64, 64, 224, 224, 3, 3, 1, 1, 30);
    // test_general_conv_kernels(3, 3, 24, 24, 3, 3, 1, 1, 30);
    // test_general_conv_kernels(64, 64, 224, 224, 3, 3, 1, 1, 1);
    // test_general_conv_kernels(128, 128, 112, 112, 3, 3, 1, 1, 1);
    
    return 0;
}
