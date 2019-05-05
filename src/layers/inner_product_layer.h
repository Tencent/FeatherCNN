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
#include "booster/sgemv.h"

#include <assert.h>
#include <stdio.h>

namespace feather
{
class InnerProductLayer : public Layer
{
    public:
        InnerProductLayer(RuntimeParameter<float>* rt_param)
            : fuse_relu(false), Layer(rt_param)
        {
        }

        int Forward()
        {
            //this->bottom_blob(0)->PrintBlobInfo();
            //this->top_blob(0)->PrintBlobInfo();
            const float *input = bottoms[0]->data();
            float *output = tops[0]->data();
            sgemv_kernel((int)input_size, (int)output_size, input, kernel_data, output, rt_param->num_threads(), bias_data);
            return 0;
        }

        int Fuse(Layer *next_layer)
        {
            if (next_layer->type.compare("ReLU") == 0)
            {
                fuse_relu = true;
                return 1;
            }
            else
            {
                return 0;
            }
        }
        int Init()
        {
            Blob<float> *p_blob = new Blob<float>;
            printf("input size %d\n", input_size);
            p_blob->ReshapeWithRealloc(1, 1, 1, input_size * 8);
            this->kernel_data = weights[0]->data();
            float* buffer = p_blob->data();
            if (input_size % 8 == 0 && output_size % 8 == 0)
            {
                for (int i = 0; i < output_size / 8; i++)
                    matrixTranspose(this->kernel_data + i * 8 * input_size, 8, input_size, buffer);
            }
            delete p_blob;
            if (output_size % 8 == 0 && input_size % 8 == 0)
            {
                if (bias_term && fuse_relu)
                    sgemv_kernel = fully_connected_transpose_inference<true, true>;
                else if (bias_term && !fuse_relu)
                    sgemv_kernel = fully_connected_transpose_inference<true, false>;
                else if (!bias_term && fuse_relu)
                    sgemv_kernel = fully_connected_transpose_inference<false, true>;
                else if (!bias_term && !fuse_relu)
                    sgemv_kernel = fully_connected_transpose_inference<false, false>;
            }
            else
            {
                if (bias_term && fuse_relu)
                    sgemv_kernel = fully_connected_inference_direct<true, true>;
                else if (bias_term && !fuse_relu)
                    sgemv_kernel = fully_connected_inference_direct<true, false>;
                else if (!bias_term && fuse_relu)
                    sgemv_kernel = fully_connected_inference_direct<false, true>;
                else if (!bias_term && !fuse_relu)
                    sgemv_kernel = fully_connected_inference_direct<false, false>;
            }

            this->bias_data = this->weights[1]->data();
            return 0;
        }

        int Reshape()
        {
            // Allocate space for the layer's own top.
            const Blob<float> *bottom_blob = bottoms[0];
            int input_width = bottom_blob->width();
            int input_height = bottom_blob->height();
            int input_channels = bottom_blob->channels();
            // this->input_size = input_width * input_height * input_channels;
            if (input_size != bottom_blob->data_size())
            {
                LOGE("In Layer %s: Bottom %s data size %zu is inconsistant with expected input size %zu.", this->name.c_str(), bottom_blob->name.c_str(), bottom_blob->data_size(), input_size);
                return -100;
            }
            this->tops[0]->ReshapeWithRealloc(1, output_size, 1, 1);
            return 0;
        }

        int LoadParam(const ncnn::ParamDict &pd)
        {
            this->output_size = pd.get(0, 0);
            this->bias_term = pd.get(1, 0);
            this->weight_data_size = pd.get(2, 0);
            this->input_size = this->weight_data_size / this->output_size;
            
            // The params are known, therefore we can allocate space for weights.
            Blob<float> *fc_weights = new Blob<float>(this->name + "_weights");
            fc_weights->ReshapeWithRealloc(this->output_size, this->input_size, 1, 1);
            weights.push_back(fc_weights);
            if (this->bias_term)
            {
                Blob<float> *bias_weights = new Blob<float>(this->name + "_bias");
                bias_weights->ReshapeWithRealloc(output_size, 1, 1, 1);
                weights.push_back(bias_weights);
            }
            return 0;
        }

        int LoadWeights(const ncnn::ModelBin& mb)
        {
            printf("Loading dimension %zu %zu\n", output_size, input_size);
            printf("weight data size %zu\n", weight_data_size);
            ncnn::Mat weight_data = mb.load(weight_data_size, 0);
            if (weight_data.empty())
                return -100;
            if (this->weights.empty())
                return -100;
            this->weights[0]->CopyDataFromMat(weight_data);

            if (this->bias_term)
            {
                ncnn::Mat bias_data = mb.load(output_size, 1);
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
    protected:
        size_t weight_data_size;

        size_t input_size;
        size_t output_size;

        bool bias_term;

        float *kernel_data;
        float *bias_data;

        bool fuse_relu;
        void (*sgemv_kernel)(const int, const int, const float *, const float *, float *, const int, float*);
};
};
