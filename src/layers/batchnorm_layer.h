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

namespace feather
{
class BatchNormLayer : Layer
{
    public:
        BatchNormLayer(const LayerParameter* layer_param, const RuntimeParameter<float>* rt_param)
            : input_channels(0),
              input_width(0),
              input_height(0),
              alpha(NULL),
              beta(NULL),
              scale_bias_term(false),
              scale_bias_data(NULL),
              fuse_scale(false),
              scale_data(NULL),
              fuse_relu(false),
              Layer(layer_param, rt_param)
        {
            _fusible = true;
        }

        int Init();
        int Forward();
        int Fuse(Layer *);
    private:
        size_t input_channels;
        size_t input_width;
        size_t input_height;

        float* alpha;
        float* beta;
        bool fuse_scale;
        float* scale_data;
        bool   scale_bias_term;
        float* scale_bias_data;
        bool fuse_relu;

        float* input;
        float* output;

    private:
        int SetKernel();
        void (*bn_kernel)(const size_t channels, const size_t stride, const float* alpha, const float* beta, const float* bias_data, const float* scale_data, const float* input, float* output, const size_t num_threads);
};
};
