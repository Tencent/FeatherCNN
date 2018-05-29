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

#include "batchnorm_layer.h"
#include "scale_layer.h"//For fuse
#include "arm/generic_kernels.h"
#include <math.h>

namespace feather
{
int BatchNormLayer::Forward()
{
    size_t stride = input_width * input_height;
    bn_kernel(input_channels, stride, alpha, beta, scale_bias_data, scale_data, input, output, num_threads);
    return 0;
}

int BatchNormLayer::Fuse(Layer *next_layer)
{
    if (next_layer->type().compare("Scale") == 0)
    {
        printf("BN %s fuse Scale layer %s\n", this->name().c_str(), next_layer->name().c_str());
        fuse_scale = true;
        for (int i = 0; i < next_layer->weight_blob_num(); ++i)
        {
            Blob<float>* p_blob = new Blob<float>();
            p_blob->Copy(next_layer->weight_blob(i));
            _weight_blobs.push_back(p_blob);
        }
        scale_bias_term = ((ScaleLayer*) next_layer)->bias_term();
        return 1;
    }
    else if (next_layer->type().compare("ReLU") == 0)
    {
        printf("BN %s fuse ReLU layer %s\n", this->name().c_str(), next_layer->name().c_str());
        fuse_relu = true;
        return 1;
    }
    else
        return 0;
}

int BatchNormLayer::Init()
{
    printf("Init BN layer ");
    const Blob<float>* p_blob = _bottom_blobs[_bottom[0]];
    input_channels = p_blob->channels();
    input_height   = p_blob->height();
    input_width    = p_blob->width();
    printf("input %d %d %d\n", input_channels, input_width, input_height);
    MEMPOOL_CHECK_RETURN(private_mempool.Alloc(&alpha, input_channels * sizeof(float)));
    MEMPOOL_CHECK_RETURN(private_mempool.Alloc(&beta, input_channels * sizeof(float)));

    float *mean_data, *var_data;
    mean_data  = _weight_blobs[0]->data();
    var_data   = _weight_blobs[1]->data();
    float scale_factor = 1 / *(_weight_blobs[2]->data());
    float eps = 1e-5;
    for (int i = 0; i < input_channels; i++)
    {
        float sqrt_var = sqrt(var_data[i] * scale_factor + eps);
        alpha[i] = -(mean_data[i] * scale_factor) / sqrt_var;
        beta[i]  = 1 / sqrt_var;
    }
    if (fuse_scale)
    {
        scale_data = _weight_blobs[3]->data();
        if (scale_bias_term)
            scale_bias_data = _weight_blobs[4]->data();
        else
            scale_bias_data = NULL;
    }
    SetKernel();
    input = _bottom_blobs[_bottom[0]]->data();
    output = _top_blobs[_top[0]]->data();
    return 0;
}

int BatchNormLayer::SetKernel()
{
    int pat_code = 0;
    pat_code += (scale_bias_term) ? 0x1 : 0;
    pat_code += (fuse_scale) ? 0x10 : 0;
    pat_code += (fuse_relu) ? 0x100 : 0;
    printf("pat_code %x\n", pat_code);
    switch (pat_code)
    {
        case 0x000:
            bn_kernel = batchnorm<false, false, false>;
            break;
        case 0x001:
            bn_kernel = batchnorm<true, false, false>;
            break;
        case 0x010:
            bn_kernel = batchnorm<false, true, false>;
            break;
        case 0x011:
            bn_kernel = batchnorm<true, true, false>;
            break;
        case 0x100:
            bn_kernel = batchnorm<false, false, true>;
            break;
        case 0x101:
            bn_kernel = batchnorm<true, false, true>;
            break;
        case 0x110:
            bn_kernel = batchnorm<false, true, true>;
            break;
        case 0x111:
            bn_kernel = batchnorm<true, true, true>;
            break;
        default:
            fprintf(stderr, "Invalid pattern code 0x%x for batchnorm kernel\n", pat_code);
            return -1;
    }
    return 0;
}
};
