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

#pragma once

#include "../feather_generated.h"
#include "../layer.h"

#include <booster/helper.h>
#include <booster/booster.h>
#include <assert.h>
#include <stdio.h>

namespace feather
{
class ConvLayer : public Layer<float>
{
    public:
        ConvLayer(const LayerParameter *layer_param, RuntimeParameter<float>* rt_param)
            : Layer<float>(layer_param, rt_param),
              conv_booster(),
              conv_param(),
              bias_data(NULL),
              kernel_data(NULL),
              processed_kernel(NULL)
        {
            this->_fusible = true;
            //From proto
            const ConvolutionParameter *conv_param_in = layer_param->convolution_param();

            conv_param.kernel_h = conv_param_in->kernel_h();
            conv_param.kernel_w = conv_param_in->kernel_w();
            conv_param.stride_h = conv_param_in->stride_h();
            conv_param.stride_w = conv_param_in->stride_w();
            conv_param.pad_left = conv_param_in->pad_w();
            conv_param.pad_bottom = conv_param_in->pad_h();
            conv_param.pad_right = conv_param_in->pad_w();
            conv_param.pad_top = conv_param_in->pad_h();
            conv_param.group = conv_param_in->group();
            kernel_data = this->_weight_blobs[0]->data();
            conv_param.output_channels = this->_weight_blobs[0]->num();
            conv_param.bias_term = conv_param_in->bias_term();
            conv_param.activation = booster::None;
            assert(this->_weight_blobs.size() > 0);

            if (conv_param.bias_term)
            {
                assert(this->_weight_blobs.size() == 2);
                conv_param.bias_fp32 = this->_weight_blobs[1]->data();
            }
        }

        int GenerateTopBlobs()
        {
            //Conv layer has and only has one bottom blob.
            const Blob<float> *bottom_blob = this->_bottom_blobs[this->_bottom[0]];
            conv_param.input_w = bottom_blob->width();
            conv_param.input_h = bottom_blob->height();
            conv_param.input_channels = bottom_blob->channels();
            conv_param.AssignOutputDim();
            conv_param.LogParams(this->name().c_str());
            this->_top_blobs[this->_top[0]] = new Blob<float>(1, conv_param.output_channels, conv_param.output_h, conv_param.output_w);
            this->_top_blobs[this->_top[0]]->Alloc();
            conv_booster.SelectAlgo(&this->conv_param);
            //conv_booster.ForceSelectAlgo(booster::NAIVE);
            //conv_booster.ForceSelectAlgo(booster::IM2COL);
            return 0;
        }

        int ForwardReshape()
        {
            const Blob<float> *bottom_blob = this->_bottom_blobs[this->_bottom[0]];
            conv_param.input_h = bottom_blob->height();
            conv_param.input_w = bottom_blob->width();
            conv_param.AssignOutputDim();
            this->_top_blobs[this->_top[0]]->ReshapeWithRealloc(1, conv_param.output_channels, conv_param.output_h, conv_param.output_w);
            return this->Forward();
        }

        int Forward()
        {
	    //conv_param.LogParams(this->name().c_str());
            float* input = this->_bottom_blobs[this->_bottom[0]]->data();
            float* output = this->_top_blobs[this->_top[0]]->data();
            float* buffer = NULL;
            conv_param.input_fp32 = input;
            conv_param.output_fp32 = output;
            
            MEMPOOL_CHECK_RETURN(this->common_mempool->GetPtr(&(conv_param.common_buffer_fp32)));
            conv_booster.Forward(&conv_param);
            return 0;
        }

        int Init()
        {
            int buffer_size = 0;
            int processed_kernel_size = 0;
            int ret = conv_booster.GetBufferSize(&conv_param, &buffer_size, &processed_kernel_size);
            MEMPOOL_CHECK_RETURN(this->private_mempool.Alloc(&processed_kernel, sizeof(float) * (processed_kernel_size)));
            conv_param.processed_kernel_fp32 = processed_kernel;
            conv_booster.Init(&conv_param);
            MEMPOOL_CHECK_RETURN(this->common_mempool->Request(sizeof(float) * buffer_size));
            return 0;
        }

        int Fuse(Layer *next_layer)
        {
            if (next_layer->type().compare("ReLU") == 0)
            {
                conv_param.activation = booster::ReLU;
                return 1;
            }
            else
            {
                return 0;
            }
        }

    protected:
        booster::ConvBooster conv_booster;
        booster::ConvParam conv_param;

        float *bias_data;

        float *kernel_data;
        float *processed_kernel;
};
};
