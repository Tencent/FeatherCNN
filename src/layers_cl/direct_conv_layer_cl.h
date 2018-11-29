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
#include <CL/opencl_kernels.h>

#include <assert.h>
#include <stdio.h>

namespace feather {
//#define USE_LEGACY_SGEMM
class DirectConvLayerCL : public Layer<uint16_t> {
public:
  DirectConvLayerCL(const LayerParameter *layer_param, RuntimeParameter<float>* rt_param);

    int InitCL();
    virtual int SetWorkSize();
    virtual int ForwardCL();
    virtual int ForwardReshapeCL();
    virtual int SetKernelParameters();
    void FinetuneKernel();
    inline void AssignOutputSize();
    int GenerateTopBlobs();
    int Fuse(Layer *next_layer);

private:
    bool fuse_relu;
    size_t in_channel_grp_size;
    size_t out_channel_grp_size;

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

    bool is_dw;

    bool bias_term;

    uint16_t *kernel_data;
    uint16_t *bias_data;

};
}; // namespace feather
