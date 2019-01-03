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

#include "filter_layer.h"
#include "booster/generic_kernels.h"
#include "math.h"

namespace feather
{
int FilterLayer::Forward()
{
    const  Blob<float>* bottom_blob = _bottom_blobs[_bottom[0]];
    const float* input = bottom_blob->data();

    size_t num = bottom_blob->num();
    size_t channels = bottom_blob->channels();
    size_t height = bottom_blob->height();
    size_t width = bottom_blob->width();

    float *output = _top_blobs[_top[0]]->data();
    float *p = output;

    size_t page_size = height * width;
    for (int i = 0; i < channels; i++)
    {
        if (fabs(select_weights[i] - 1.0) < 1e-5)
        {
            memcpy(p, input + i * page_size, page_size * (sizeof(float)));
            p += page_size;
        }
    }

    return 0;
}

int FilterLayer::GenerateTopBlobs()
{
    assert(_bottom.size() == 1);
    assert(_top.size() == 1);

    const Blob<float>* bottom_blob = _bottom_blobs[_bottom[0]];
    size_t num = bottom_blob->num();
    size_t channels = num_output;
    size_t height = bottom_blob->height();
    size_t width = bottom_blob->width();

    _top_blobs[_top[0]] = new Blob<float>(num, channels, height, width);
    _top_blobs[_top[0]]->Alloc();

    return 0;
}

int FilterLayer::ForwardReshape()
{
    assert(_bottom.size() == 1);
    assert(_top.size() == 1);

    const Blob<float>* bottom_blob = _bottom_blobs[_bottom[0]];
    size_t num = bottom_blob->num();
    size_t channels = num_output;
    size_t height = bottom_blob->height();
    size_t width = bottom_blob->width();

    _top_blobs[_top[0]]->ReshapeWithRealloc(num, channels, height, width);
    return this->Forward();
}

int FilterLayer::Init()
{
    select_weights = _weight_blobs[0]->data();

    const Blob<float>* bottom_blob = _bottom_blobs[_bottom[0]];
    size_t channels = bottom_blob->channels();
    return 0;
}
};
