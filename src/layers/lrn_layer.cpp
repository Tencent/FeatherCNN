//Tencent is pleased to support the open source community by making FeatherCNN available.

//Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.

//Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
//in compliance with the License. You may obtain a copy of the License at
//
//https://opensource.org/licenses/BSD-3-Clause
//
//Unless required by applicable law or agreed to in writing, software distributed
//under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//CONDITIONS OF ANY KIND, either express or implied. See the License for the
//specific language governing permissions and limitations under the License.

#include "lrn_layer.h"
#include "../mempool.h"
#ifdef FEATHER_ARM
#include "arm/generic_kernels.h"
#include "arm/power.h"
#else
#include "general/generic_kernels.h"
#include "general/power.h"
#endif
#include <cmath>

namespace feather
{
LRNLayer::LRNLayer(const LayerParameter* layer_param, const RuntimeParameter<float>* rt_param)
    :   local_size(5),
        alpha(1.),
        beta(0.75),
        k(1.),
        Layer(layer_param, rt_param)
{
    local_size = layer_param->lrn_param()->local_size();
    assert(local_size % 2 == 1);
    alpha = layer_param->lrn_param()->alpha();
    beta = layer_param->lrn_param()->beta();
    k = layer_param->lrn_param()->k();
    _pre_pad = (local_size - 1) / 2;
    printf("localsize %ld alpha %f beta %f k %f\n", local_size, alpha, beta, k);
}

int LRNLayer::Init()
{
    auto p_blob = _bottom_blobs[bottom(0)];
    size_t width = p_blob->width();
    size_t height = p_blob->height();
    size_t channels = p_blob->channels();
    alpha_over_size = alpha / local_size;
#if 0
    MEMPOOL_CHECK_RETURN(private_mempool.Alloc(&_square, sizeof(float) * width * height * channels));
    MEMPOOL_CHECK_RETURN(private_mempool.Alloc(&_square_sum, sizeof(float) * width * height * channels));
#else
    size_t padded_size = width * height * (channels + 2 * _pre_pad);
    MEMPOOL_CHECK_RETURN(private_mempool.Alloc(&_padded_sqr_data, sizeof(float) * padded_size));
    MEMPOOL_CHECK_RETURN(private_mempool.Alloc(&_scale_data, sizeof(float) * width * height * channels));
    memset(_padded_sqr_data, 0, sizeof(float) * padded_size);
#endif
    return 0;
}

int WithinChannels()
{
    //Not implemented yet.
    fprintf(stderr, "Within Channels not implemented\n");
    return -1;
}

#if 0
int LRNLayer::AcrossChannels()
{
    auto p_blob = _bottom_blobs[bottom(0)];
    size_t width = p_blob->width();
    size_t height = p_blob->height();
    size_t channels = p_blob->channels();

    auto bottom_data = p_blob->data();
    auto top_data = _top_blobs[top(0)]->data();

    printf("chw (%ld %ld %ld)\n", channels, height, width);

    size_t img_size = width * height;
    for (int i = 0; i < img_size * channels; ++i)
    {
        float v = *(p_blob->data() + i);
        _square[i] = v * v;
    }
    //Across channels
    for (int c = 0; c < channels; ++c)
    {
        memset(_square_sum, 0, width * height * sizeof(float));
        float* sum_ptr = _square_sum;
        float* out_ptr = top_data + c * img_size;
        for (int m = c - local_size / 2; m < c + local_size / 2; ++m)
        {
            if (m >= channels || m < 0)
                continue;
            float* sqr_ptr = _square + m * img_size;
            for (int i = 0; i < img_size; ++i)
            {
                sum_ptr[i] += alpha_over_size * sqr_ptr[i];
            }
        }
        for (int i = 0; i < img_size; ++i)
        {
            float power = pow(alpha_over_size * sum_ptr[i] + k, -beta);
            out_ptr[i] = bottom_data[i] * power;
        }
    }

    for (int i = 0; i < channels * img_size; ++i)
    {
        //printf("%f\n", top_data[i]);
    }
    //exit(0);
    return 0;
}
#else
int LRNLayer::AcrossChannels()
{
    auto   p_blob = _bottom_blobs[bottom(0)];
    size_t width = p_blob->width();
    size_t height = p_blob->height();
    size_t channels = p_blob->channels();
    size_t img_size = width * height;
    size_t buf_size = channels * width * height;

    auto bottom_data = p_blob->data();
    auto top_data = _top_blobs[top(0)]->data();

    //printf("chw (%ld %ld %ld)\n", channels, height, width);
    for (int i = 0; i < buf_size; ++i)
    {
        _scale_data[i] = k;
    }
    float * sqr_ptr = _padded_sqr_data + _pre_pad * img_size;
    for (int i = 0; i < buf_size; ++i)
    {
        sqr_ptr[i] = bottom_data[i] * bottom_data[i];
    }
    //Create scale for the first channel
    for (int c = _pre_pad; c < local_size; ++c)
    {
        const float* img = _padded_sqr_data + img_size * c;
        for (int i = 0; i < img_size; ++i)
        {
            _scale_data[i] = alpha_over_size * img[i] + _scale_data[i];
        }
    }
    for (int c = 1; c < channels; ++c)
    {
        float* scale_data_c = _scale_data + img_size * c;
        //Copy from previous scale
        memcpy(scale_data_c, scale_data_c - img_size, sizeof(float) * img_size);
        //Add head and subtract tail
        for (int i = 0; i < img_size; ++i)
        {
            scale_data_c[i] = alpha_over_size * ((_padded_sqr_data + img_size * (c + local_size - 1))[i] - (_padded_sqr_data + img_size * (c - 1))[i]) + scale_data_c[i];
        }
    }

//  float32x4_t v_beta = vdupq_n_f32(-beta);
// #pragma omp parallel for num_threads(num_threads)
//  for(int i = 0; i < channels * img_size; i+=4)
//  {
//    float32x4_t v_scale_data = vld1q_f32(_scale_data + i);
//    float32x4_t v_bottom_data = vld1q_f32(bottom_data + i);
//    v_scale_data = pow_ps(v_scale_data, v_beta);
//    v_bottom_data = vmulq_f32(v_scale_data, v_bottom_data);
//    vst1q_f32(top_data + i, v_bottom_data);
//  }

    // if(channels * img_size % 4 != 0)
    {
        for (int i = 0; i < channels * img_size; i++)
        {
            float power = std::pow(_scale_data[i], -beta);
            top_data[i] = bottom_data[i] * power;
        }
    }
    return 0;
}
#endif
int LRNLayer::Forward()
{
    //Across channels mode only.
    return AcrossChannels();
}
};
