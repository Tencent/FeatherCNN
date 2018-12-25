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

#include <booster/booster.h>
#include <booster/helper.h>
#include <booster/opencl_kernels.h>

#include <string.h>

namespace booster
{
//NAIVE Methods
int NAIVE_Init_CL(std::vector<std::string>& cl_kernel_names,
        std::vector<std::string>& cl_kernel_symbols,
        std::vector<std::string>& cl_kernel_functions)
{
    std::string func_name_conv = "convolution";
    std::string kernel_name_conv = "conv_1v1_buffer";
    auto it_source = booster::opencl_kernel_string_map.find("conv_1v1_buffer");
    std::string kernel_str_conv(it_source->second.begin(), it_source->second.end());


    cl_kernel_names.push_back(kernel_name_conv);
    cl_kernel_symbols.push_back(kernel_str_conv);
    cl_kernel_functions.push_back(func_name_conv);
    return 0;
}

int NAIVE_Forward_CL()
{
    return 0;
}

template <typename Dtype>
int NAIVE_Weight_Reform_CL(const ConvParam param,
                          size_t n_grp_size,
                          size_t c_grp_size,
                          const Dtype* weight,
                          Dtype* weight_reformed)
{
    size_t w_num = param.output_channels;
    size_t w_channels = param.input_channels;
    size_t w_hw = param.kernel_h * param.kernel_w;

    for (int i = 0; i < w_num; ++i) {
      for (int k = 0; k < w_channels; ++k) {
        for (int j = 0; j < w_hw; ++j) {
          int src_idx = (i * w_channels + k) * w_hw + j;
          int dst_idx = (i / n_grp_size) * w_hw * param.ic_padded * n_grp_size +
                      j * param.ic_padded * n_grp_size +
                      ( k / c_grp_size ) * n_grp_size * c_grp_size +
                      ( i % n_grp_size ) * c_grp_size +
                      k % c_grp_size;
          weight_reformed[dst_idx] = weight[src_idx];
        }
      }
    }
    return 0;
}

int DEPTHWISE_Init_CL(std::vector<std::string>& cl_kernel_names,
        std::vector<std::string>& cl_kernel_symbols,
        std::vector<std::string>& cl_kernel_functions)
{
    std::string func_name_depthwise = "convolution_depthwise";
    std::string kernel_name_depthwise_conv = "depthwise_conv_1v1_buffer";
    auto it_source = booster::opencl_kernel_string_map.find("depthwise_conv_1v1_buffer");
    std::string kernel_str_depthwise_conv(it_source->second.begin(), it_source->second.end());

    cl_kernel_names.push_back(kernel_name_depthwise_conv);
    cl_kernel_symbols.push_back(kernel_str_depthwise_conv);
    cl_kernel_functions.push_back(func_name_depthwise);

    return 0;
}

int DEPTHWISE_Forward_CL()
{
    return 0;
}

template <typename Dtype>
int DEPTHWISE_Weight_Reform_CL(const ConvParam param,
                          size_t n_grp_size,
                          size_t c_grp_size,
                          const Dtype* weight,
                          Dtype* weight_reformed)
{
    size_t w_num = param.output_channels;
    size_t w_hw = param.kernel_h * param.kernel_w;

    for (int i = 0; i < w_num; ++i) {
      for (int j = 0; j < w_hw; ++j) {
        int dst_idx = (i / n_grp_size * w_hw + j)
                    * n_grp_size
                    + i % n_grp_size;
        int src_idx = i * w_hw + j;
        weight_reformed[dst_idx] = weight[src_idx];
      }
    }
    return 0;
}



int BOTH_Set_Conv_Kernel_Params_CL(const ConvParam param, CLBuffers buffers, std::vector<cl::Kernel> kernels, bool is_reshape)
{
    int param_idx = 0;
    bool set_kernel_arg_success = true;
    if (!is_reshape){
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, *buffers. input_mem));
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, *buffers.weight_mem));
        if (param.bias_term) {
          set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, *buffers.bias_mem));
        }
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, *buffers.output_mem));
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, param.ic_padded));
        if (param.group != param.input_channels) {
          set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, param.oc_padded));
        }
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, param.input_h));
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, param.input_w));
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, param.output_h));
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, param.output_w));
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, param.kernel_h));
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, param.kernel_w));
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, param.stride_h));
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, param.stride_w));
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, param.pad_top));
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, param.pad_left));
        if (!set_kernel_arg_success) {
          LOGE("Failed setting conv OpenCL kernels[0] arguments.");
          return -1;
        }
    }
    else
    {
        param_idx = param.group != param.input_channels ? 6 : 5;
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(0, *buffers.input_mem));
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(3, *buffers.output_mem));
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, param.input_h));
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, param.input_w));
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, param.output_h));
        set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, param.output_w));
        if (!set_kernel_arg_success) {
          LOGE("Failed setting conv reshape OpenCL kernels[0] arguments.");
          return -1;
        }

    }
    return 0;
}


int WINOGRADF23_Init_CL(ConvParam *param, float* processed_kernel, float* kernel)
{
    return 0;
}

int WINOGRADF23_Forward_CL(ConvParam *param, float* output, float* input, float* processed_kernel, float* buffer, float* bias_arr)
{
    return 0;
}

//Class wrappers
template <class Dtype>
ConvBoosterCL<Dtype>::ConvBoosterCL()
    :Init(NULL), Forward(NULL)
{
}

template <class Dtype>
size_t ConvBoosterCL<Dtype>::GetWeightSize()
{
    return this->weight_size;
}
//Conditional algo selecter
template <class Dtype>
int ConvBoosterCL<Dtype>::SelectAlgo(ConvParam* param)
{
    if (param->group == param->input_channels)
    {
        this->algo = DEPTHWISE;
        this->weight_size = param->kernel_h * param->kernel_w * param->oc_padded;
    }
    else if (param->group == 1)
    {
       this->algo = NAIVE;
       this->weight_size = param->kernel_h * param->kernel_w * param->oc_padded * param->ic_padded;
    }
    else
    {
        LOGE("Partial group conv is not yet supported. If you need it, try develop your own im2col method.");
        return -1;
    }
    return this->SetFuncs();
}

//Force algo selecter
template <class Dtype>
int ConvBoosterCL<Dtype>::ForceSelectAlgo(ConvAlgo algo)
{
    this->algo = algo;
    return this->SetFuncs();
}



template <class Dtype>
int ConvBoosterCL<Dtype>::SetFuncs()
{
    switch (this->algo)
    {
    case NAIVE:
        this->Init = NAIVE_Init_CL;
        this->Forward = NAIVE_Forward_CL;
        this->WeightReform = NAIVE_Weight_Reform_CL;
        this->SetConvKernelParams = BOTH_Set_Conv_Kernel_Params_CL;
        return 0;
    case DEPTHWISE:
        this->Init = DEPTHWISE_Init_CL;
        this->Forward = DEPTHWISE_Forward_CL;
        this->WeightReform = DEPTHWISE_Weight_Reform_CL;
        this->SetConvKernelParams = BOTH_Set_Conv_Kernel_Params_CL;
        return 0;
    default:
        LOGE("This algo is not supported on GPU.");
        this->Init = NULL;
        this->Forward = NULL;
        this->WeightReform = NULL;
        return -1;
    }
}
template class ConvBoosterCL<float>;
template class ConvBoosterCL<uint16_t>;
}; // namespace booster
