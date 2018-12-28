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

#include <assert.h>
#include <stdio.h>

namespace feather {
//#define USE_LEGACY_SGEMM

template <class Dtype>
class EltwiseLayerCL: public Layer<Dtype> {
public:
  EltwiseLayerCL(const LayerParameter* layer_param, RuntimeParameter<Dtype>* rt_param)
      : Layer<Dtype>(layer_param, rt_param) {
    this->_fusible = true;
    fuse_relu = false;
    InitCL();
  }

  int InitCL();
  int GenerateTopBlobs();
  virtual int SetBuildOptions();
  virtual int SetWorkSize();
  virtual int ResetWorkSize();
  virtual int SetKernelParameters();
  virtual int ForwardReshapeCL();
  virtual int ForwardCL();

  int Fuse(Layer<Dtype> *next_layer) {
    if (next_layer->type().compare("ReLU") == 0) {
      printf("Eltwise %s fuse ReLU layer %s\n", this->name().c_str(), next_layer->name().c_str());
      fuse_relu = true;
      return 1;
    } else {
      return 0;
    }
  }

private:
  size_t output_height;
  size_t output_width;
  size_t output_channels;
  size_t channel_grp_size;
  bool fuse_relu;

};
}; // namespace feather
