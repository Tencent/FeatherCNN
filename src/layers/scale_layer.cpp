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

#include "scale_layer.h"
#include "arm/generic_kernels.h"

namespace feather
{
int ScaleLayer::Forward()
{
    const float* input = _bottom_blobs[_bottom[0]]->data();
    float* output = _top_blobs[_top[0]]->data();
    size_t stride = input_width * input_height;
    scale_kernel(input_channels, stride, bias_data, scale_data, input, output, num_threads);
    return 0;
}

int ScaleLayer::Init()
{
    printf("Init Scale layer ");
    const Blob<float>* p_blob = _bottom_blobs[_bottom[0]];
    input_channels = p_blob->channels();
    input_height   = p_blob->height();
    input_width    = p_blob->width();
    printf("input %d %d %d", input_channels, input_width, input_height);
    scale_data = _weight_blobs[0]->data();
    //printf("bias_term %d\n", _bias_term ? 1: 0);
    if (_bias_term)
    {
        bias_data = _weight_blobs[1]->data();
        scale_kernel = scale<true>;
    }
    else
    {
        scale_kernel = scale<false>;
    }
    return 0;
}
};
