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
#include "arm/generic_kernels.h"
#include "arm/depthwise.h"

#include <assert.h>
#include <stdio.h>

namespace feather
{
class ConvDepthwiseLayer : public ConvLayer
{
    public:
        ConvDepthwiseLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
            : ConvLayer(layer_param, rt_param)
        {
            //From proto
        }

        int Init()
        {
            int inputw = input_width + padding_left + padding_right;
            int inputh = input_height + padding_top + padding_bottom;
            MEMPOOL_CHECK_RETURN(private_mempool.Alloc(&padded_input, inputw * inputh * input_channels * sizeof(float)));
            return 0;
        }

        int Forward()
        {
            const float *input = _bottom_blobs[_bottom[0]]->data();
            float *output = _top_blobs[_top[0]]->data();
            int inputw = input_width + padding_left + padding_right;
            int inputh = input_height + padding_top + padding_bottom;
            pad_input(padded_input, input, input_channels, input_width, input_height, padding_left, padding_top, padding_right, padding_bottom);
            dwConv(output, padded_input, inputw, inputh, stride_width, stride_height, kernel_data, kernel_width, kernel_height, group, num_threads);
            if (bias_term)
            {
                size_t out_stride = output_width * output_height;
                for (int i = 0; i < output_channels; ++i)
                {
                    float bias = bias_data[i];
                    for (int j = 0; j < out_stride; ++j)
                    {
                        output[out_stride * i + j] = output[out_stride * i + j] + bias;
                    }
                }
            }
            return 0;
        }

    private:
        float* padded_input;

};
};
