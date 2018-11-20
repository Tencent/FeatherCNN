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
#include <string>
#include <map>

namespace feather {
class InputLayerCL : public Layer<uint16_t> {
public:
  InputLayerCL(const LayerParameter *layer_param, RuntimeParameter<float>* rt_param)
      : Layer<uint16_t>(layer_param, rt_param) {
    //From proto
    const InputParameter *input_param = layer_param->input_param();
    size_t input_num = VectorLength(input_param->name());
    size_t input_dim_num = VectorLength(input_param->dim());
    assert(input_num > 0);
    assert(input_dim_num == input_num * 4);
    for (int i = 0; i < input_num; ++i) {
      size_t num = input_param->dim()->Get(i * 4);
      size_t channels = input_param->dim()->Get(i * 4 + 1);
      size_t height = input_param->dim()->Get(i * 4 + 2);
      size_t width = input_param->dim()->Get(i * 4 + 3);

      std::string input_name = input_param->name()->Get(i)->str();
      _top.push_back(input_name);
      _top_blobs[input_name] = new Blob<uint16_t>(num, channels, height, width);

      //_top_blobs[input_name]->PrintBlobInfo();
      LOGI("input_name cl %s (n c h w)=(%ld %ld %ld %ld)\n", input_name.c_str(), num, channels, height, width);
      this->InitCL();
    }
  }

  ~InputLayerCL() {
    int error_num;
    error_num = clEnqueueUnmapMemObject(this->rt_param->command_queue(), this->_cl_img2d, this->_map_img, 0, NULL, NULL);
    if (!checkSuccess(error_num)) {
      LOGE("fatal error: Deconstructor Unmapping _cl_img2d objects failed.");
    }

    error_num = clEnqueueUnmapMemObject(this->rt_param->command_queue(), this->_cl_fimage, this->_map_fdata, 0, NULL, NULL);
    if (!checkSuccess(error_num)) {
      LOGE("fatal error: Deconstructor Unmapping _cl_img2d objects failed.");
    }
  }

  int InitCL();
  int UintToDevice(const uint8_t* src_bgra);
  int FloatToDevice(const float* input_data);
  int RunKernel(int type);
  //int ForwardClImage();

  //int build();
  virtual int SetKernelParameters();

  int CopyInput(std::string name, const float *input_data) {
    this->FloatToDevice(input_data);
    this->RunKernel(0);
    return 0;
  }

  int CopyInput(std::string name, const uint8_t* src_bgra) {
    this->UintToDevice(src_bgra);
    this->RunKernel(1);
    return 0;
  }

  size_t input_size() {
    return _top_blobs.size();
  }

  std::string input_name(int idx) {
    auto it = _top_blobs.begin();
    for (int i = 0; i < idx; ++i) {
        ++it;
    }
    return it->first;
  }

  cl_mem _cl_img2d;
  cl_mem _cl_fimage;
  float* _map_fdata;
  uint8_t* _map_img;
};
}; // namespace feather
