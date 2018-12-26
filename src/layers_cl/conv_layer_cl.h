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

#include "../feather_generated.h"
#include "../layer.h"
#include "blob.h"
#include <booster/opencl_kernels.h>
#include <booster/booster.h>

#include <assert.h>
#include <stdio.h>

namespace feather {
//#define USE_LEGACY_SGEMM

template <class Dtype>
class ConvLayerCL: public Layer<Dtype> {
public:
  ConvLayerCL(const LayerParameter *layer_param, RuntimeParameter<float>* rt_param);

    virtual int SetBuildOptions();
    virtual int ForwardCL();
    virtual int ForwardReshapeCL();
    virtual int SetKernelParameters();
    int GenerateTopBlobs();
    int Fuse(Layer<Dtype> *next_layer);

private:
    // uint32_t in_channel_grp_size;
    uint32_t channel_grp_size;
    booster::ConvBoosterCL<Dtype> conv_booster;
    booster::ConvParam conv_param;
    size_t conv_gws[3][3];
    size_t conv_lws[3][3];

    Dtype *bias_data;
    Dtype *kernel_data;
    //float *processed_kernel;

};
}; // namespace feather
