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
class EltwiseLayerCL : public Layer<uint16_t> {
public:
  EltwiseLayerCL(const LayerParameter* layer_param, RuntimeParameter<float>* rt_param)
      : Layer<uint16_t>(layer_param, rt_param) {
    _fusible = true;
    fuse_relu = false;
    InitCL();
  }

  int InitCL();
  int GenerateTopBlobs();
  int SetKernelParameters();
  void FinetuneKernel();
  int ForwardCL();

  int Fuse(Layer *next_layer) {
    if (next_layer->type().compare("ReLU") == 0) {
      printf("Eltwise %s fuse ReLU layer %s\n", this->name().c_str(), next_layer->name().c_str());
      fuse_relu = true;
      return 1;
    } else {
      return 0;
    }
  }

private:
  bool fuse_relu;

};
}; // namespace feather
