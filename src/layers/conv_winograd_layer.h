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
#include "conv_layer.h"
#include "blob.h"

#include "arm/generic_kernels.h"
#include "arm/winograd_kernels.h"

#include <assert.h>
#include <stdio.h>

namespace feather
{
class ConvWinogradLayer : public ConvLayer
{
    public:
        ConvWinogradLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
            : ConvLayer(layer_param, rt_param)
        {
            _fusible = true;
        }


        int Forward()
        {
            float* common_mem = NULL;
            MEMPOOL_CHECK_RETURN(common_mempool->GetPtr(&common_mem));
            const size_t inputw = input_width + padding_left + padding_right;
            const size_t inputh = input_height + padding_top + padding_bottom;

            //Get addresses
            float *VT = common_mem;
            float *WT = VT + 16 * (inputw / 2 - 1) * (inputh / 2 - 1) * input_channels;            //Offset by sizeof VT
            float *padded_input = WT + 16 * (inputw / 2 - 1) * (inputh / 2 - 1) * output_channels; //Offset by sizeof WT
            pad_input(padded_input, input, input_channels, input_width, input_height, padding_left, padding_top, padding_right, padding_bottom);
            if (ext_pad_w || ext_pad_h)
            {
                int outputw = inputw - kernel_width + 1;
                int outputh = inputh - kernel_height + 1;
                float *tmp_out = padded_input + inputw * inputh * input_channels;
                //printf("ext_pad_w %ld, ext_pad_h %ld, output w %d, output h %d\n", ext_pad_w, ext_pad_h, outputh, outputw);
                winogradNonFusedTransform(tmp_out, output_channels, WT, VT, UT, padded_input, input_channels, inputw, inputh, winograd_out_type, bias_data, num_threads);
                int tw = outputw - ext_pad_w;
                int th = outputh - ext_pad_h;
                for (int c = 0; c < output_channels; ++c)
                {
                    float *outputp = tmp_out + c * outputh * outputw;
                    float *tp = output + c * th * tw;
                    for (int i = 0; i < th; ++i)
                    {
                        memcpy(tp + i * tw, outputp + i * outputw, sizeof(float) * tw);
                    }
                }
            }
            else
            {
                winogradNonFusedTransform(output, output_channels, WT, VT, UT, padded_input, input_channels, inputw, inputh, winograd_out_type, bias_data, num_threads);
            }
            return 0;

        }

        int Fuse(Layer *next_layer)
        {
            if (next_layer->type().compare("ReLU") == 0)
            {
                fuse_relu = true;
                return 1;
            }
            else
                return 0;
        }
        int Init()
        {
            size_t inputw = input_width + padding_left + padding_right;
            size_t inputh = input_height + padding_top + padding_bottom;

            ext_pad_w = inputw % 2;
            ext_pad_h = inputh % 2;
            padding_right += ext_pad_w;
            padding_bottom += ext_pad_h;
            inputw += ext_pad_w;
            inputh += ext_pad_h;

            size_t winograd_mem_size = 0;
            winograd_mem_size += 16 * (inputw / 2 - 1) * (inputh / 2 - 1) * input_channels;  //VT
            winograd_mem_size += 16 * (inputw / 2 - 1) * (inputh / 2 - 1) * output_channels; //WT
            winograd_mem_size += inputw * inputh * input_channels;                           //Padded Input
            if (ext_pad_w || ext_pad_h)
            {
                int outputw = inputw - kernel_width + 1;
                int outputh = inputh - kernel_height + 1;
                winograd_mem_size += outputw * outputh * output_channels;
            }
            float* ST = NULL;
            MEMPOOL_CHECK_RETURN(common_mempool->Request(winograd_mem_size * sizeof(float)));
            MEMPOOL_CHECK_RETURN(private_mempool.Alloc(&UT, 16 * input_channels * output_channels * sizeof(float)));
            MEMPOOL_CHECK_RETURN(private_mempool.Alloc(&ST, 16 * input_channels * output_channels * sizeof(float)));
            transformKernel(UT, kernel_data, input_channels, output_channels, ST);
            MEMPOOL_CHECK_RETURN(private_mempool.Free(&ST));

            if (bias_term && fuse_relu)
                winograd_out_type = BiasReLU;
            else if (bias_term)
                winograd_out_type = Bias;
            else if (fuse_relu)
                winograd_out_type = ReLU;
            else
                winograd_out_type = None;
            //Setup input and output pointers.
            input = _bottom_blobs[_bottom[0]]->data();
            output = _top_blobs[_top[0]]->data();

            return 0;
        }
    private:
        float* UT;
        float* input;
        float* output;
        size_t ext_pad_w;
        size_t ext_pad_h;

        bool fuse_relu;
        WinogradOutType winograd_out_type;
};
};
