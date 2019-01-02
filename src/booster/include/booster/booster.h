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

// Booster is the standalone backend of FeatherCNN, in order to facilitate unit testing
// and multi-purpose deployment. I am currently focusing on the fast convolution kernels,
// and will pack other operators as well. This backend library is now supporting
// AVX and Neon, and is going to supoort OpenCL/GLES in the future.
// Booster won't grow up into a hugh and abstract lib. I'll keep it simple and stupid.
// -- Haidong Lan @ Tencent AI Platform, 08/30/2018

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <map>
#include <string>
#include <vector>
#ifdef FEATHER_OPENCL
#include "CLHPP/clhpp_runtime.hpp"
#endif

namespace booster{

enum ConvAlgo{
    NAIVE,
    IM2COL,
    SGECONV,
    DEPTHWISE,
    WINOGRADF63,
    WINOGRADF63FUSED,
    WINOGRADF23,
};

enum ActivationType{
    None,
    ReLU,
};

struct ConvParam {
    int output_channels;
    int input_channels;
    int input_h;
    int input_w;
    int kernel_h;
    int kernel_w;
    int output_h;
    int output_w;
    int stride_h;
    int stride_w;
    int pad_left;
    int pad_bottom;
    int pad_right;
    int pad_top;
    int group;
    bool bias_term;
    ActivationType activation;
#ifdef FEATHER_OPENCL
    int channel_grp_size;
    int padded_input_channels;
    int padded_output_channels;
    int padded_input_h;
    int padded_input_w;
    int padded_output_h;
    int padded_output_w;
    bool padding_needed;
#endif

    void AssignOutputDim()
    {
        //Validate default values
        if (group == 0)	group = 1;
        if (stride_h == 0) stride_h = 1;
        if (stride_w == 0) stride_w = 1;
        output_h = (input_h + pad_top + pad_bottom - kernel_h) / stride_h + 1;
        output_w = (input_w + pad_left + pad_right - kernel_w) / stride_w + 1;
        if (group == input_channels)
        {
            output_channels = input_channels;
        }
    }
    void AssignPaddedDim()
    {
        input_h = input_h + pad_top + pad_bottom;
        input_w = input_w + pad_left + pad_right;
        pad_left = 0;
        pad_bottom = 0;
        pad_right = 0;
        pad_top = 0;
    }
    void LogParams(const char* layer_name)
    {
        printf("-----Layer %s ConvParam----\n", layer_name);
        printf("Input CxHxW=(%d, %d, %d)\n", input_channels, input_h, input_w);
        printf("Output CxHxW=(%d, %d, %d)\n", output_channels, output_h, output_w);
        printf("Group = %d\n", group);
        printf("Kernel HxW=(%d, %d)\n", kernel_h, kernel_w);
        printf("Stride HxW=(%d, %d)\n", stride_h, stride_w);
        printf("Paddings (%d %d %d %d)\n", pad_left, pad_bottom, pad_right, pad_top);
    }
};

typedef int (*GET_BUFFER_SIZE_FUNC)(ConvParam *param, int* buffer_size, int* processed_kernel_size);
typedef int (*INIT_FUNC)(ConvParam *param, float* processed_kernel, float* kernel);
typedef int (*FORWARD_FUNC)(ConvParam *param, float* output, float* input, float* kernel, float* buffer, float* bias_arr);

//ConvBooster doesn't allocate any memory.
class ConvBooster
{
public:
    ConvBooster();
    ~ConvBooster() {}
    int SelectAlgo(ConvParam* param);
    int ForceSelectAlgo(ConvAlgo algo);
    int SetFuncs();
    GET_BUFFER_SIZE_FUNC GetBufferSize;
    INIT_FUNC Init;
    FORWARD_FUNC Forward;

private:
    ConvAlgo algo;
};

#ifdef FEATHER_OPENCL

struct CLBuffers{
    cl::Buffer* input_mem;
    cl::Buffer* output_mem;
    cl::Buffer* weight_mem;
    cl::Buffer* bias_mem;
    cl::Buffer* input_trans_mem;
    cl::Buffer* out_trans_mem;
};


template <class Dtype>
class ConvBoosterCL
{
public:

  typedef int (*INIT_FUNC_CL)(std::map<std::string, clhpp_feather::CLKernelInfo>& cl_kernel_info_map);
  typedef int (*FORWARD_FUNC_CL)(cl::CommandQueue cmd_q,
                                  std::vector<std::string> kernel_names,
                                  std::map<std::string, clhpp_feather::CLKernelInfo>& cl_kernel_info_map,
                                  const ConvParam& param,
                                  clhpp_feather::OpenCLRuntime* cl_runtime,
                                  std::string layer_name);
  typedef int (*WEIGHT_REFORM_FUNC_CL)(const ConvParam& param,
                                  size_t n_grp_size,
                                  size_t c_grp_size,
                                  const Dtype* weight,
                                  Dtype* weight_reformed);
  typedef int (*SET_CONV_KERNEL_PARAMS_CL)(const ConvParam& param,
                                           const CLBuffers& buffers,
                                  std::vector<std::string> kernel_names,
                                  std::map<std::string, clhpp_feather::CLKernelInfo>& cl_kernel_info_map,
                                  clhpp_feather::OpenCLRuntime* cl_runtime,
                                  bool is_reshape);
  typedef int (*SET_CONV_WORK_SIZE_CL)(const ConvParam& param,
                                  std::map<std::string, clhpp_feather::CLKernelInfo>& cl_kernel_info_map,
                                  std::vector<std::string> kernel_names,
                                  clhpp_feather::OpenCLRuntime* cl_runtime);
  typedef int (*SET_BUILD_OPTS_CL)(const ConvParam& param,
                                  bool is_fp16,
                                  const std::vector<std::string>& kernel_names,
                                  std::map<std::string, clhpp_feather::CLKernelInfo>& cl_kernel_info_map);

  ConvBoosterCL();
  ~ConvBoosterCL() {}
  int SelectAlgo(ConvParam* param);
  int ForceSelectAlgo(ConvAlgo algo);
  int SetFuncs();
  size_t GetWeightSize();
  const std::vector<std::string>& GetKernelNames();
  INIT_FUNC_CL Init;
  FORWARD_FUNC_CL Forward;
  WEIGHT_REFORM_FUNC_CL WeightReform;
  SET_CONV_KERNEL_PARAMS_CL SetConvKernelParams;
  SET_CONV_WORK_SIZE_CL SetConvWorkSize;
  SET_BUILD_OPTS_CL SetBuildOpts;
private:
  ConvAlgo algo;
  size_t weight_size;
  std::vector<std::string> kernel_names;

};
#endif

};
