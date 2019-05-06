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

#include "../layer.h"

#include <booster/helper.h>
#include <booster/booster.h>
#include <assert.h>
#include <stdio.h>

namespace feather
{
class ConvLayer : public Layer
{
    public:
        ConvLayer(RuntimeParameter<float>* rt_param)
            : Layer(rt_param),
              conv_booster(),
              conv_param(),
              bias_data(NULL),
              processed_kernel(NULL)
        {
            // this->_fusible = true;
        }

        int LoadParam(const ncnn::ParamDict& pd)
        {
            int dilation_w = pd.get(2, 1);
            int dilation_h = pd.get(12, dilation_w);
            if ((dilation_w > 1) || (dilation_h > 1))
            {
                printf("Dilated convolution is not supported in FeatherCNN. Please refer to the ncnn repository.\n");
                return -200; //Not supported
            }

            int int8_scale_term = pd.get(8, 0);
            if (int8_scale_term)
            {
                printf("Dilated convolution is not supported in FeatherCNN. Please refer to the ncnn repository.\n");
                return -200; //Not supported
            }

            conv_param.kernel_w = pd.get(1, 0);
            conv_param.kernel_h = pd.get(11, conv_param.kernel_w);  
            conv_param.stride_w = pd.get(3, 1);
            conv_param.stride_h = pd.get(13, conv_param.stride_w);
            conv_param.pad_left = pd.get(4, 0);
            conv_param.pad_bottom = pd.get(14, conv_param.pad_left);
            conv_param.pad_right = pd.get(4, 0);
            conv_param.pad_top = pd.get(14, conv_param.pad_left);
            conv_param.group = pd.get(7, 1);
            conv_param.output_channels = pd.get(0, 0);
            conv_param.bias_term = pd.get(5, 0);
            conv_param.activation = booster::None;
            int weight_data_size = pd.get(6, 0);
            conv_param.input_channels = weight_data_size / conv_param.output_channels / conv_param.kernel_h / conv_param.kernel_w;

            // The params are known, therefore we can allocate space for weights.
            Blob<float> *conv_weights = new Blob<float>(this->name + "_weights");
            conv_weights->ReshapeWithRealloc(conv_param.output_channels, conv_param.input_channels, conv_param.kernel_h, conv_param.kernel_w);
            weights.push_back(conv_weights);
            if (conv_param.bias_term)
            {
                Blob<float> *bias_weights = new Blob<float>(this->name + "_bias");
                bias_weights->ReshapeWithRealloc(conv_param.output_channels, 1, 1, 1);
                weights.push_back(bias_weights);
            }
            return 0;
        }

        int Reshape()
        {
            // Allocate space for the layer's own top.
            const Blob<float> *bottom_blob = this->bottoms[0];
            conv_param.input_w = bottom_blob->width();
            conv_param.input_h = bottom_blob->height();
            if (conv_param.input_channels != bottom_blob->channels())
            {
                LOGE("Loaded convolution layer %s has %d input channels while bottom blob has %zu channels\n", this->name.c_str(), conv_param.input_channels, bottom_blob->channels());
                return -300; //Topology error
            }
            // printf("##########################\n");
            conv_param.LogParams(this->name.c_str());
            conv_param.AssignOutputDim();
            conv_param.LogParams(this->name.c_str());
            tops[0]->ReshapeWithRealloc(1, conv_param.output_channels, conv_param.output_h, conv_param.output_w);
            conv_booster.SelectAlgo(&this->conv_param);
            int buffer_size = 0;
            int dull = 0;
            int ret = conv_booster.GetBufferSize(&conv_param, &buffer_size, &dull);
            MEMPOOL_CHECK_RETURN(this->common_mempool->Request(sizeof(float) * buffer_size));
            return 0;
        }

        int LoadWeights(const ncnn::ModelBin& mb)
        {
            int weight_data_size = conv_param.input_channels * conv_param.output_channels * conv_param.kernel_h * conv_param.kernel_w;
            ncnn::Mat weight_data = mb.load(weight_data_size, 0);
            if (weight_data.empty())
                return -100;
            if (this->weights.empty())
                return -100;
            this->weights[0]->CopyDataFromMat(weight_data);

            if (conv_param.bias_term)
            {
                ncnn::Mat bias_data = mb.load(conv_param.output_channels, 1);
                if (bias_data.empty())
                    return -100;
                if (this->weights.size() < 2)
                {
                    LOGE("In layer %s: Bias weight blob not allocated.", this->name.c_str());
                    return -100;
                }
                weights[1]->CopyDataFromMat(bias_data);
            }
            return 0;
        }

        int Forward()
        {
	        // conv_param.LogParams(this->name().c_str());
            float* input  = this->bottoms[0]->data();
            float* output = this->tops[0]->data();
            float* buffer = NULL;
            MEMPOOL_CHECK_RETURN(this->common_mempool->GetPtr(&buffer));
            conv_booster.Forward(&conv_param, output, input, processed_kernel, buffer, bias_data);
            return 0;
        }

        int Init()
        {
            int buffer_size = 0;
            int processed_kernel_size = 0;
            int ret = conv_booster.GetBufferSize(&conv_param, &buffer_size, &processed_kernel_size);
            Blob<float> * processed_weights = new Blob<float>(this->name + "_proc_weights");
            processed_weights->ReshapeWithRealloc(1, 1, 1, processed_kernel_size);
            float* kernel_data = this->weights[0]->data();
            float* processed_kernel = processed_weights->data();
            conv_booster.Init(&conv_param, processed_kernel, kernel_data);
            delete this->weights[0];
            this->weights[0] = processed_weights;
            this->processed_kernel = processed_weights->data();
            if (conv_param.bias_term)
            {
                bias_data = this->weights[1]->data();
            }
            // MEMPOOL_CHECK_RETURN(this->common_mempool->Request(sizeof(float) * buffer_size));
            return 0;
        }
    
        int Fuse(Layer *next_layer)
        {
            if (next_layer->type.compare("ReLU") == 0)
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
        float *processed_kernel;
};
};