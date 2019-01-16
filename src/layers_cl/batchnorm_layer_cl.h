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
#include <booster/opencl_kernels.h>

namespace feather
{

template <class Dtype>
class BatchNormLayerCL : public Layer<Dtype>
{
    public:
        BatchNormLayerCL(const LayerParameter* layer_param, RuntimeParameter<Dtype>* rt_param);

        int InitCL();
        virtual int SetBuildOptions();
        virtual int SetKernelParameters();
        virtual int ForwardCL();
        virtual int SetWorkSize();
        virtual int ForwardReshapeCL();
        virtual int GenerateTopBlobs();
        void PadParamsDevice(Blob<Dtype>* blob, Dtype* data);
        int PreCalParams();
        int Fuse(Layer<Dtype> *next_layer);
    private:
        size_t output_channels;
        size_t output_width;
        size_t output_height;
        size_t channel_block_size;

        bool fuse_scale;
        bool fuse_relu;
        bool scale_bias_term;
        Dtype* scale_data;
        Dtype* scale_bias_data;
        Dtype* alpha;
        Dtype* beta;



};

};
