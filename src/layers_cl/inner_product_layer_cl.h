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
#include <CL/opencl_kernels.h>
#include <assert.h>
#include <stdio.h>
#include <vector>

namespace feather {
class InnerProductLayerCL : public Layer<uint16_t> {
public:
  InnerProductLayerCL(const LayerParameter *layer_param, RuntimeParameter<float>* rt_param);

  int InitCL();
  virtual void SetBuildOptions();
  virtual int SetWorkSize();
  virtual int SetKernelParameters();
  virtual int ForwardCL();
  virtual int ForwardReshapeCL();
  void FinetuneKernel();
  int GenerateTopBlobs();
  int Fuse(Layer *next_layer);

protected:
  uint32_t input_width;
  uint32_t input_height;
  uint32_t channel_grp_size;

  uint32_t output_channels;

  uint16_t *kernel_data;
  uint16_t *bias_data;

  bool bias_term;

  bool fuse_relu;
};
}; // namespace feather
