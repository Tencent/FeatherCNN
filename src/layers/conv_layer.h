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

#pragma once

#include "../feather_simple_generated.h"
#include "../layer.h"

#include <assert.h>
#include <stdio.h>

namespace feather
{
class ConvLayer : public Layer
{
    public:
        ConvLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
            : Layer(layer_param, rt_param)
        {
            //From proto
            const ConvolutionParameter *conv_param = layer_param->convolution_param();
            bias_term = conv_param->bias_term();

            group = conv_param->group();
            kernel_height = conv_param->kernel_h();
            kernel_width = conv_param->kernel_w();

            stride_height = conv_param->stride_h();
            stride_width = conv_param->stride_w();

            padding_left = conv_param->pad_w();
            padding_top = conv_param->pad_h();
            padding_right = conv_param->pad_w();
            padding_bottom = conv_param->pad_h();

            assert(_weight_blobs.size() > 0);
            kernel_data = this->_weight_blobs[0]->data();
            output_channels = this->_weight_blobs[0]->num();
            // input_channels = this->_weight_blobs[0]->channels();
            if (bias_term)
            {
                assert(this->_weight_blobs.size() == 2);
                bias_data = this->_weight_blobs[1]->data();
            }
        }

        int GenerateTopBlobs()
        {
            //Conv layer has and only has one bottom blob.
            const Blob<float> *bottom_blob = _bottom_blobs[_bottom[0]];
            input_width = bottom_blob->width();
            input_height = bottom_blob->height();
            input_channels = bottom_blob->channels();
            if (stride_width == 0 || stride_height == 0)
            {
                stride_width = 1;
                stride_height = 1;
            }
            output_width = (input_width + padding_left + padding_right - kernel_width) / stride_width + 1;
            output_height = (input_height + padding_top + padding_bottom - kernel_height) / stride_height + 1;
#if 0
            printf("input channels %d\n", input_channels);
            assert(input_channels == bottom_blob->channels());
            printf("input w %lu\n", input_width);
            printf("padding_left %lu\n", padding_left);
            printf("padding_top %lu\n", padding_top);
            printf("stride_width %lu\n", stride_width);
            printf("stride_height %lu\n", stride_height);
            printf("output %ld %ld\n", output_width, output_height);
#endif
            _top_blobs[_top[0]] = new Blob<float>(1, output_channels, output_height, output_width);
            _top_blobs[_top[0]]->Alloc();
            return 0;
        }

        void simple_conv(
            const float *input,
            size_t input_channels,
            size_t input_width,
            size_t input_height,
            float *output,
            size_t output_channels,
            size_t output_width,
            size_t output_height,
            float *kernel_data,
            size_t kernel_width,
            size_t kernel_height,
            float *bias_data,
            bool bias_term,
            size_t stride_width,
            size_t stride_height,
            size_t padding_left,
            size_t padding_right,
            size_t padding_top,
            size_t padding_bottom)
        {
        }

        virtual int Forward()
        {
            const float *input = _bottom_blobs[_bottom[0]]->data();
            float *output = _top_blobs[_top[0]]->data();
            // conv(input, output);

            return -1;
        }

    protected:
        size_t input_channels;
        size_t input_width;
        size_t input_height;

        size_t output_channels;
        size_t output_width;
        size_t output_height;

        size_t kernel_width;
        size_t kernel_height;

        size_t stride_width;
        size_t stride_height;

        size_t padding_left;
        size_t padding_right;
        size_t padding_top;
        size_t padding_bottom;

        size_t group;

        bool bias_term;

        float *kernel_data;
        float *bias_data;
};
};
