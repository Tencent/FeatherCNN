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
#include "blob.h"
#include <CLHPP/opencl_kernels.hpp>

#include <assert.h>
#include <stdio.h>

namespace feather
{
//#define USE_LEGACY_SGEMM

template <class Dtype>
class ConcatLayerCL: public Layer<Dtype>
{
    public:
        ConcatLayerCL(const LayerParameter* layer_param, RuntimeParameter<Dtype>* rt_param);
        virtual int GenerateTopBlobs();
        virtual int SetBuildOptions();
        virtual int SetKernelParameters();
        virtual int ForwardReshapeCL();
        virtual int ForwardCL();
        virtual int Fuse(Layer<Dtype> *next_layer);


    private:
        int output_height;
        int output_width;
        int output_channels;
        int input0_channels;
        int input1_channels;
        int channel_block_size;
        bool fuse_relu;
        bool divisible;

};
}; // namespace feather
