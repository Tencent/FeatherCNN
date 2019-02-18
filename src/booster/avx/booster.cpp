//Tencent is pleased to support the open source community by making FeatherCNN available.

//Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.

//Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
//in compliance with the License. You may obtain a copy of the License at
//
//https://opensource.org/licenses/BSD-3-Clause
//
//Unless required by applicable law or agreed to in writing, software distributed
//under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//CONDITIONS OF ANY KIND, either express or implied. See the License for the
//specific language governing permissions and limitations under the License.

#include <booster/booster.h>
#include <booster/depthwise.h>
#include <booster/generic_kernels.h>
#include <booster/sgemm.h>
#include <booster/sgeconv.h>
#include <booster/helper.h>
#include <booster/winograd_kernels.h>

#include <string.h>

namespace booster
{
//NAIVE Methods
int NAIVE_GetBufferSize(ConvParam *param, int* buffer_size, int* processed_kernel_size)
{
    *buffer_size = param->input_channels * param->output_h * param->output_w * param->kernel_h * param->kernel_w;
    *processed_kernel_size = param->input_channels * param->output_channels * param->kernel_h * param->kernel_w;
    return 0;
}

int NAIVE_Init(ConvParam *param)
{
    memcpy(param->processed_kernel_fp32, param->kernel_fp32, sizeof(float) * param->output_channels * param->input_channels * param->kernel_h * param->kernel_w);
    return 0;
}

int NAIVE_Forward(ConvParam *param)
{
    const int M = param->output_channels;
    const int N = param->output_h * param->output_w;
    const int K = param->input_channels * param->kernel_h * param->kernel_w;
    im2col(param, param->common_buffer_fp32, param->input_fp32);
    naive_sgemm(M, N, K, param->processed_kernel_fp32, param->common_buffer_fp32, param->output_fp32);
    if (param->bias_term)
    {
        size_t out_stride = param->output_w * param->output_h;
        for (int i = 0; i < param->output_channels; ++i)
        {
            float bias = param->bias_fp32[i];
            for (int j = 0; j < out_stride; ++j)
            {
                param->output_fp32[out_stride * i + j] = param->output_fp32[out_stride * i + j] + bias;
            }
        }
    }
    return 0;
}

//IM2COL Methods
int IM2COL_GetBufferSize(ConvParam *param, int* buffer_size, int* processed_kernel_size)
{
    const int nc = 160;
    const int kc = 320;
    *buffer_size = (kc + 6) * nc + param->input_channels * param->output_h * param->output_w * param->kernel_h * param->kernel_w;
    *processed_kernel_size = param->input_channels * param->output_channels * param->kernel_h * param->kernel_w;
    return 0;
}

int IM2COL_Init(ConvParam *param)
{
    const int nc = 160;
    const int kc = 320;
    const int M = param->output_channels;
    const int K = param->input_channels * param->kernel_h * param->kernel_w;
    packed_sgemm_init<6>(M, K, kc, param->processed_kernel_fp32, param->kernel_fp32, K);
    return 0;
}

int IM2COL_Forward(ConvParam *param)
{
    const int nc = 160;
    const int kc = 320;
    const int offset = (kc + 6) * nc;
    const int M = param->output_channels;
    const int N = param->output_h * param->output_w;
    const int K = param->input_channels * param->kernel_h * param->kernel_w;
    float* im2col_buf = param->common_buffer_fp32 + offset;
    im2col(param, im2col_buf, param->input_fp32);
    if ((!param->bias_term) && (param->activation == None))
        packed_sgemm_activation<false, false>(M, N, K, param->processed_kernel_fp32, im2col_buf, N, param->output_fp32, N, nc, kc, param->bias_fp32, 1, param->common_buffer_fp32);
    else if ((param->bias_term) && (param->activation == None))
        packed_sgemm_activation<true,  false>(M, N, K, param->processed_kernel_fp32, im2col_buf, N, param->output_fp32, N, nc, kc, param->bias_fp32, 1, param->common_buffer_fp32);
    else if ((!param->bias_term) && (param->activation == ReLU))
        packed_sgemm_activation<false,  true>(M, N, K, param->processed_kernel_fp32, im2col_buf, N, param->output_fp32, N, nc, kc, param->bias_fp32, 1, param->common_buffer_fp32);
    else if ((param->bias_term) && (param->activation == ReLU))
        packed_sgemm_activation<true,   true>(M, N, K, param->processed_kernel_fp32, im2col_buf, N, param->output_fp32, N, nc, kc, param->bias_fp32, 1, param->common_buffer_fp32);
    return 0;
}

//SGECONV Methods
int SGECONV_GetBufferSize(ConvParam *param, int* buffer_size, int* processed_kernel_size)
{
    return 0;
}

int SGECONV_Init(ConvParam *param)
{
    return 0;
}

int SGECONV_Forward(ConvParam *param)
{
    return 0;
}

//DEPTHWISE Methods
int DEPTHWISE_GetBufferSize(ConvParam *param, int* buffer_size, int* processed_kernel_size)
{
	ConvParam padded_param = *param;
	padded_param.AssignPaddedDim();
	*buffer_size = param->input_channels * padded_param.input_h * padded_param.input_w;
    *processed_kernel_size = param->group * param->kernel_h * param->kernel_w;
    return 0;
}

int DEPTHWISE_Init(ConvParam *param)
{
    memcpy(param->processed_kernel_fp32, param->kernel_fp32, sizeof(float) * param->group * param->kernel_h * param->kernel_w);
    return 0;
}

int DEPTHWISE_Forward(ConvParam *param)
{
    void (*dwConv)(float *, float *, int, int, int, int, int, float *, int, int, int, int, float *);
    if (param->bias_term && (param->activation == ReLU))
        dwConv = dwConv_template<true, true>;
    else if (param->bias_term && !(param->activation == ReLU))
        dwConv = dwConv_template<true, false>;
    else if (!param->bias_term && (param->activation == ReLU))
        dwConv = dwConv_template<false, true>;
    else if (!param->bias_term && !(param->activation == ReLU))
        dwConv = dwConv_template<false, false>;

    if (param->pad_left > 0 || param->pad_right > 0 || param->pad_top > 0 || param->pad_bottom > 0)
    {
		ConvParam padded_param = *param;
		padded_param.AssignPaddedDim();
        pad_input(param->common_buffer_fp32, param->input_fp32, param->input_channels, param->input_w, param->input_h, param->pad_left,
                  param->pad_top, param->pad_right, param->pad_bottom);
		dwConv(param->output_fp32, param->common_buffer_fp32, param->input_channels, padded_param.input_w, padded_param.input_h, param->stride_w, param->stride_h, param->processed_kernel_fp32, param->kernel_w, param->kernel_h, param->group, 1, param->bias_fp32);
    }
    else
        dwConv(param->output_fp32, param->input_fp32, param->input_channels, param->input_w, param->input_h, param->stride_w, param->stride_h, param->processed_kernel_fp32, param->kernel_w, param->kernel_h, param->group, 1, param->bias_fp32);
    return 0;
}
//WINOGRADF23 Methods
int WINOGRADF23_GetBufferSize(ConvParam *param, int* buffer_size, int* processed_kernel_size)
{
    return 0;
}

int WINOGRADF23_Init(ConvParam *param)
{
    return 0;
}

int WINOGRADF23_Forward(ConvParam *param)
{
    return 0;
}

//WINOGRADF63 Methods
int WINOGRADF63_GetBufferSize(ConvParam *param, int* buffer_size, int* processed_kernel_size)
{
    int num_threads = 1;
    ConvParam padded_param = *param;
    padded_param.AssignPaddedDim();
    size_t packArraySize = getPackArraySize_F6x6_3x3(padded_param.input_channels, num_threads);
    int nRowBlocks = (padded_param.input_w + 3) / 6;
    int nColBlocks = (padded_param.input_h + 3) / 6;
    int nBlocks = nRowBlocks * nColBlocks;

    size_t winograd_mem_size = 0;
    winograd_mem_size += 64 * nBlocks * padded_param.input_channels;    //VT
    winograd_mem_size += 64 * nBlocks * padded_param.output_channels;   //WT
    winograd_mem_size += packArraySize;                    //WT
    winograd_mem_size += padded_param.input_w * padded_param.input_h * padded_param.input_channels; //Padded Input

    *buffer_size = winograd_mem_size;
    *processed_kernel_size = 64 * padded_param.input_channels * padded_param.output_channels;
    return 0;
}

int WINOGRADF63_Init(ConvParam *param)
{
    transformKernel_F6x6_3x3(param->processed_kernel_fp32, param->kernel_fp32, param->input_channels, param->output_channels);
    return 0;
}

int WINOGRADF63_Forward(ConvParam *param)
{
    const size_t inputw = param->input_w + param->pad_left + param->pad_right;
    const size_t inputh = param->input_h + param->pad_top + param->pad_bottom;
    const int nRowBlocks = (inputw + 3) / 6;
    const int nColBlocks = (inputh + 3) / 6;
    const int nBlocks = nRowBlocks * nColBlocks;

    //Get addresses
    float *VT = param->common_buffer_fp32;
    float *WT = VT + 64 * nBlocks * param->input_channels;                      //Offset by sizeof VT
    float *padded_input = WT + 64 * nBlocks * param->output_channels;           //Offset by sizeof WT
    float *pack_array = padded_input + inputw * inputh * param->input_channels; //Offset by sizeof WT
    pad_input(padded_input, param->input_fp32, param->input_channels, param->input_w, param->input_h, param->pad_left, param->pad_top, param->pad_right, param->pad_bottom);
    WinogradOutType out_type;
    if ((!param->bias_term) && (param->activation == None))
        out_type = Nothing;
    else if ((param->bias_term) && (param->activation == None))
        out_type = Bias;
    else if ((!param->bias_term) && (param->activation == ReLU))
        out_type = Relu;
    else if ((param->bias_term) && (param->activation == ReLU))
        out_type = BiasReLU;
    winogradNonFusedTransform_F6x6_3x3(param->output_fp32, param->output_channels, WT, VT, param->processed_kernel_fp32, padded_input, param->input_channels, inputh, inputw, out_type, param->bias_fp32, pack_array, 1);
    return 0;
}

//WINOGRADF63FUSED Methods
int WINOGRADF63FUSED_GetBufferSize(ConvParam *param, int* buffer_size, int* processed_kernel_size)
{
    int num_threads = 1;
    const int cache_block = 48;
    size_t winograd_mem_size = 0;
    winograd_mem_size += cache_block * 64 * param->input_channels * 4; //fused VT
    winograd_mem_size += cache_block * 64 * param->output_channels * 4; //fused WT

    *buffer_size = winograd_mem_size;
    *processed_kernel_size = 64 * param->input_channels * param->output_channels;
    return 0;
}

int WINOGRADF63FUSED_Init(ConvParam *param)
{
    Winograd_F63_Fused::transformKernel_F6x6_3x3(param->processed_kernel_fp32, param->kernel_fp32, param->input_channels, param->output_channels);
    return 0;
}

int WINOGRADF63FUSED_Forward(ConvParam *param)
{
    // printf("Fused Winograd Forward\n");
    // param->LogParams("FORWARD_TEST");
    if (!param->bias_term && param->activation == None)
    {
        Winograd_F63_Fused::WinogradF63Fused<false, false>(param, param->output_fp32, param->input_fp32, param->processed_kernel_fp32, param->bias_fp32, param->common_buffer_fp32);
    }
    else if (!param->bias_term && param->activation == ReLU)
    {
        Winograd_F63_Fused::WinogradF63Fused<true, false>(param, param->output_fp32, param->input_fp32, param->processed_kernel_fp32, param->bias_fp32, param->common_buffer_fp32);
    }
    else if (param->bias_term && param->activation == None)
    {
        Winograd_F63_Fused::WinogradF63Fused<false, true>(param, param->output_fp32, param->input_fp32, param->processed_kernel_fp32, param->bias_fp32, param->common_buffer_fp32);
    }
    else if (param->bias_term && param->activation == ReLU)
    {
        Winograd_F63_Fused::WinogradF63Fused<true, true>(param, param->output_fp32, param->input_fp32, param->processed_kernel_fp32, param->bias_fp32, param->common_buffer_fp32);
    }
    return 0;
}

int DUMMY_Destroy(ConvParam *param)
{
    //Destroy nothing
    return 0;
}

//MKLDNN Methods
int MKLDNN_GetBufferSize(ConvParam *param, int* buffer_size, int* processed_kernel_size)
{
    *buffer_size = 0;
    // *processed_kernel_size = param->kernel_h * param->kernel_w * param->input_channels * param->output_channels;
    *processed_kernel_size = 0;
    return 0;
}

#include <mm_malloc.h>

#define CHECK(f) do { \
    mkldnn_status_t s = f; \
    if (s != mkldnn_success) { \
        printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f, s); \
        exit(2); \
    } \
} while(0)

#define CHECK_TRUE(expr) do { \
    int e_ = expr; \
    if (!e_) { \
        printf("[%s:%d] %s failed\n", __FILE__, __LINE__, #expr); \
        exit(2); \
    } \
} while(0)

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

static void MKLDNN_init_data_memory(uint32_t dim, const int *dims,
        mkldnn_memory_format_t user_fmt, mkldnn_data_type_t mkldnn_f32,
        mkldnn_engine_t engine, float *data, mkldnn_primitive_t *memory)
{
    mkldnn_memory_desc_t prim_md;
    mkldnn_primitive_desc_t user_pd;
    CHECK(mkldnn_memory_desc_init(&prim_md, dim, dims, mkldnn_f32, user_fmt));
    CHECK(mkldnn_memory_primitive_desc_create(&user_pd, &prim_md, engine));
    CHECK(mkldnn_primitive_create(memory, user_pd, NULL, NULL));

    void *req = NULL;
    CHECK(mkldnn_memory_get_data_handle(*memory, &req));
    CHECK_TRUE(req == NULL);
    CHECK(mkldnn_memory_set_data_handle(*memory, data));
    CHECK(mkldnn_memory_get_data_handle(*memory, &req));
    CHECK_TRUE(req == data);
    CHECK(mkldnn_primitive_desc_destroy(user_pd));
}


mkldnn_status_t MKLDNN_prepare_reorder(
        mkldnn_primitive_t *user_memory, /** in */
        const_mkldnn_primitive_desc_t *prim_memory_pd, /** in */
        int dir_is_user_to_prim, /** in: user -> prim or prim -> user */
        mkldnn_primitive_t *prim_memory, /** out: memory primitive created */
        mkldnn_primitive_t *reorder, /** out: reorder primitive created */
        float *buffer)
{
    const_mkldnn_primitive_desc_t user_memory_pd;
    mkldnn_primitive_get_primitive_desc(*user_memory, &user_memory_pd);

    if (!mkldnn_memory_primitive_desc_equal(user_memory_pd, *prim_memory_pd)) {
        /* memory_create(&p, m, NULL) means allocate memory */
        CHECK(mkldnn_primitive_create(prim_memory, *prim_memory_pd,
                NULL, NULL));
        mkldnn_primitive_desc_t reorder_pd;
        if (dir_is_user_to_prim) {
            /* reorder primitive descriptor doesn't need engine, because it is
             * already appeared in in- and out- memory primitive descriptors */
            CHECK(mkldnn_reorder_primitive_desc_create(&reorder_pd,
                        user_memory_pd, *prim_memory_pd));
            mkldnn_primitive_at_t inputs = { *user_memory, 0 };
            const_mkldnn_primitive_t outputs[] = { *prim_memory };
            CHECK(mkldnn_primitive_create(reorder, reorder_pd, &inputs,
                        outputs));
        } else {
            CHECK(mkldnn_reorder_primitive_desc_create(&reorder_pd,
                        *prim_memory_pd, user_memory_pd));
            mkldnn_primitive_at_t inputs = { *prim_memory, 0 };
            const_mkldnn_primitive_t outputs[] = { *user_memory };
            CHECK(mkldnn_primitive_create(reorder, reorder_pd, &inputs,
                        outputs));
        }
        CHECK(mkldnn_memory_set_data_handle(*prim_memory, buffer));
        CHECK(mkldnn_primitive_desc_destroy(reorder_pd));
    } else {
        *prim_memory = NULL;
        *reorder = NULL;
    }
    return mkldnn_success;
}

int MKLDNN_Init(ConvParam *param)
{
    int input_dims[4] = {1, param->input_channels, param->input_h, param->input_w};
    int output_dims[4] = {1, param->output_channels, param->output_h, param->output_w};
    int kernel_dims[4] = {param->output_channels, param->input_channels, param->kernel_h, param->kernel_w};
    int bias_dims[1] = {param->output_channels};
    int strides[2] = {param->stride_h, param->stride_w};
    int padding_horizontal[2] = {param->pad_left, param->pad_right};
    int padding_vertical[2] = {param->pad_top, param->pad_bottom};
    mkldnn_engine_t engine;
    CHECK(mkldnn_engine_create(&engine, mkldnn_cpu, 0 /* idx */));

    mkldnn_primitive_t user_input_memory, user_output_memory, user_kernel_memory, user_bias_memory;
    MKLDNN_init_data_memory(4, input_dims, mkldnn_nchw, mkldnn_f32, engine, param->input_fp32, &user_input_memory);
    MKLDNN_init_data_memory(4, output_dims, mkldnn_nchw, mkldnn_f32, engine, param->output_fp32, &user_output_memory);
    MKLDNN_init_data_memory(4, kernel_dims, mkldnn_nchw, mkldnn_f32, engine, param->kernel_fp32, &user_kernel_memory);
    MKLDNN_init_data_memory(1, bias_dims, mkldnn_x, mkldnn_f32, engine, param->bias_fp32, &user_bias_memory);
    
    mkldnn_memory_desc_t conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md;
    CHECK(mkldnn_memory_desc_init(&conv_src_md, 4, input_dims, mkldnn_f32, mkldnn_any));
    CHECK(mkldnn_memory_desc_init(&conv_weights_md, 4, kernel_dims, mkldnn_f32, mkldnn_any));
    CHECK(mkldnn_memory_desc_init(&conv_bias_md, 1, bias_dims, mkldnn_f32, mkldnn_x));
    CHECK(mkldnn_memory_desc_init(&conv_dst_md, 4, output_dims, mkldnn_f32, mkldnn_any));
    
    /* Create convolution primitive descriptor*/
    mkldnn_convolution_desc_t conv_any_desc;
    CHECK(mkldnn_convolution_forward_desc_init(&conv_any_desc, mkldnn_forward,
                                               mkldnn_convolution_direct, &conv_src_md, &conv_weights_md,
                                               &conv_bias_md, &conv_dst_md, strides, padding_horizontal,
                                               padding_vertical, mkldnn_padding_zero));
    mkldnn_primitive_desc_t conv_pd;
    CHECK(mkldnn_primitive_desc_create(&conv_pd, &conv_any_desc, engine, NULL));

    mkldnn_primitive_t conv_internal_src_memory, conv_internal_weights_memory,
        conv_internal_dst_memory;
    

    /* create reorder primitives between user data and convolution srcs
     * if required */
    mkldnn_primitive_t conv_reorder_src, conv_reorder_weights, conv_reorder_dst;

    // src reorder
    const_mkldnn_primitive_desc_t src_pd = mkldnn_primitive_desc_query_pd(
        conv_pd, mkldnn_query_src_pd, 0);
    size_t conv_src_size = mkldnn_memory_primitive_desc_get_size(src_pd);
    float *conv_src_buffer = (float *)aligned_malloc(conv_src_size, 64);
    CHECK(MKLDNN_prepare_reorder(&user_input_memory, &src_pd, 1,
                          &conv_internal_src_memory, &conv_reorder_src, conv_src_buffer));
    // dst reorder
    const_mkldnn_primitive_desc_t dst_pd = mkldnn_primitive_desc_query_pd(
        conv_pd, mkldnn_query_dst_pd, 0);
    size_t conv_dst_size = mkldnn_memory_primitive_desc_get_size(dst_pd);
    float *conv_dst_buffer = (float *)aligned_malloc(conv_dst_size, 64);
    CHECK(MKLDNN_prepare_reorder(&user_output_memory, &dst_pd, 0,
                          &conv_internal_dst_memory, &conv_reorder_dst, conv_dst_buffer));

    // weights reorder
    const_mkldnn_primitive_desc_t weights_pd = mkldnn_primitive_desc_query_pd(
            conv_pd, mkldnn_query_weights_pd, 0);
    size_t conv_weights_size
            = mkldnn_memory_primitive_desc_get_size(weights_pd);
    float *conv_weights_buffer = (float *)aligned_malloc(conv_weights_size, 64);
    CHECK(MKLDNN_prepare_reorder(&user_kernel_memory, &weights_pd, 1,
            &conv_internal_weights_memory, &conv_reorder_weights,
            conv_weights_buffer));
    mkldnn_primitive_t init_primitive;
    mkldnn_stream_t init_stream;
    CHECK(mkldnn_stream_create(&init_stream, mkldnn_eager));
    CHECK(mkldnn_stream_submit(init_stream, 1, &conv_reorder_weights, NULL));
    CHECK(mkldnn_stream_wait(init_stream, 1, NULL));
    // print_floats(kernel, param->output_channels * param->input_channels, param->kernel_h * param->kernel_w);
    CHECK(mkldnn_stream_destroy(init_stream));

    // Setup handle for the preprocessed kernel func.
    mkldnn_primitive_t processed_kernel_memory;
    CHECK(mkldnn_primitive_create(
            &processed_kernel_memory, weights_pd, NULL, NULL));
    CHECK(mkldnn_memory_set_data_handle(
        processed_kernel_memory, conv_weights_buffer));

    // print_floats(conv_weights_buffer, conv_weights_size / 8 / sizeof(float), 8);
    mkldnn_primitive_t conv_src_memory = conv_internal_src_memory ?
        conv_internal_src_memory : user_input_memory;
    mkldnn_primitive_t conv_dst_memory = conv_internal_dst_memory ?
        conv_internal_dst_memory : user_output_memory;

    mkldnn_primitive_at_t conv_srcs[] = {
        mkldnn_primitive_at(conv_src_memory, 0),
        // mkldnn_primitive_at(processed_kernel_memory, 0),
        mkldnn_primitive_at(conv_reorder_weights, 0),
        mkldnn_primitive_at(user_bias_memory, 0),
    };
    const_mkldnn_primitive_t conv_dsts[] = { conv_internal_dst_memory };

    /* finally create a convolution primitive */
    mkldnn_primitive_t conv_primitives;
    CHECK(mkldnn_primitive_create(&conv_primitives, conv_pd, conv_srcs, conv_dsts));

    param->forward_primitives.clear();
    if (conv_src_memory == conv_internal_src_memory)
        param->forward_primitives.push_back(conv_reorder_src);
    param->forward_primitives.push_back(conv_primitives);
    if (conv_dst_memory == conv_internal_dst_memory)
        param->forward_primitives.push_back(conv_reorder_dst);
    printf("forward primitive num %ld\n", param->forward_primitives.size());

    CHECK(mkldnn_primitive_desc_destroy(conv_pd));

    return 0;
}

int MKLDNN_Forward(ConvParam *param)
{
    mkldnn_stream_t stream;
    CHECK(mkldnn_stream_create(&stream, mkldnn_eager));
    CHECK(mkldnn_stream_submit(stream, param->forward_primitives.size(), param->forward_primitives.data(), NULL));
    CHECK(mkldnn_stream_wait(stream, param->forward_primitives.size(), NULL));
    // print_floats(conv_dst_buffer, param->output_channels * param->output_h, param->output_w);
    // printf("-----------------------------\n");
    // print_floats(param->output_fp32, param->output_channels * param->output_h, param->output_w);
    return 0;
}

int MKLDNN_Destroy(ConvParam *param)
{
    return 0;
}

//Class wrappers
ConvBooster::ConvBooster()
    : GetBufferSize(NULL), Init(NULL), Forward(NULL)
{
}

//Conditional algo selecter
int ConvBooster::SelectAlgo(ConvParam* param)
{
    if (param->group == param->input_channels)
    {
        this->algo = DEPTHWISE;
    }
    else if (param->group == 1 && param->kernel_h == 3 && param->kernel_w == 3 && param->stride_h == 1 && param->stride_w == 1  && param->input_h > 8 && param->input_w > 8 &&  param->output_channels % 4 == 0 && param->input_channels % 4 == 0)
    {
        this->algo = WINOGRADF63FUSED;
        //this->algo = WINOGRADF63;
    }
    else if (param->group == 1 && param->kernel_w > 1 && param->kernel_h > 1)
    {
        //this->algo = SGECONV;
        this->algo = IM2COL;
    }
    else if (param->group == 1)
    {
        this->algo = IM2COL;
        //this->algo = NAIVE;
    }
    else
    {
        LOGE("Partial group conv is not yet supported. If you need it, try develop your own im2col method.");
        return -1;
    }
    return this->SetFuncs();
}

//Force algo selecter
int ConvBooster::ForceSelectAlgo(ConvAlgo algo)
{
    this->algo = algo;
    return this->SetFuncs();
}

int ConvBooster::SetFuncs()
{
    switch (this->algo)
    {
        case NAIVE:
            this->GetBufferSize = NAIVE_GetBufferSize;
            this->Init = NAIVE_Init;
            this->Forward = NAIVE_Forward;
            return 0;
        case IM2COL:
            this->GetBufferSize = IM2COL_GetBufferSize;
            this->Init = IM2COL_Init;
            this->Forward = IM2COL_Forward;
            return 0;
        case WINOGRADF63:
            this->GetBufferSize = WINOGRADF63_GetBufferSize;
            this->Init = WINOGRADF63_Init;
            this->Forward = WINOGRADF63_Forward;
            return 0;
        case WINOGRADF63FUSED:
            this->GetBufferSize = WINOGRADF63FUSED_GetBufferSize;
            this->Init = WINOGRADF63FUSED_Init;
            this->Forward = WINOGRADF63FUSED_Forward;
            return 0;
        case DEPTHWISE:
            this->GetBufferSize = DEPTHWISE_GetBufferSize;
            this->Init = DEPTHWISE_Init;
            this->Forward = DEPTHWISE_Forward;
            return 0;
        case MKLDNN:
            this->GetBufferSize = MKLDNN_GetBufferSize;
            this->Init = MKLDNN_Init;
            this->Forward = MKLDNN_Forward;
            return 0;
        default:
            LOGE("This algo is not supported on AVX2.");
            this->GetBufferSize = NULL;
            this->Init = NULL;
            this->Forward = NULL;
            return -1;
    }
}
}; // namespace booster
