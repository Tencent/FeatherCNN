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
#include <booster/opencl_kernels.h>
#include <assert.h>
#include <stdio.h>
#include <string>
#include <map>
#include <sstream>

namespace feather {

template <class Dtype>
class InputLayerCL : public Layer<Dtype> {
public:
  InputLayerCL(const LayerParameter *layer_param, RuntimeParameter<float>* rt_param);
  int InitCL();
  int UintToDevice(const uint8_t* src_bgra);
  int FloatToDevice(const float* input_data);
  int CopyInput(std::string name, const float *input_data);
  int CopyInput(std::string name, const uint8_t* src_bgra);
  int ReshapeFloat(std::string name, int height, int width);
  int ResetWorkSizeFloat();
  int RunKernel(std::string kernel_type);
  virtual int SetWorkSize();
  virtual int SetBuildOptions();
  virtual int SetKernelParameters();
  int ResetInputAndArgs(size_t data_size);

  size_t input_size()
  {
      return this->_top_blobs.size();
  }

  std::string input_name(int idx) {
    auto it = this->_top_blobs.begin();
    for (int i = 0; i < idx; ++i) {
        ++it;
    }
    return it->first;
  }

private:
  size_t output_height;
  size_t output_width;
  size_t input_channels;
  size_t input_data_size;
  size_t channel_grp_size;

  cl::Image2D _cl_img2d;
  cl::Buffer _cl_fimage;

};
}; // namespace feather
