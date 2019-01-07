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

#include <booster/booster.h>
#include <booster/helper.h>
#include <booster/opencl_kernels.h>

#include <string.h>

namespace booster
{

int Insert(const std::string& cl_program_name,
           const std::string& cl_kernel_name,
           std::map<std::string, clhpp_feather::CLKernelInfo>& cl_kernel_info_map)
{
    auto it_source = booster::opencl_kernel_string_map.find(cl_program_name);
    if (it_source != booster::opencl_kernel_string_map.end())
    {
        cl_kernel_info_map[cl_kernel_name].program_name = cl_program_name;
        cl_kernel_info_map[cl_kernel_name].kernel_name = cl_kernel_name;

        std::string conv_kernel_source(it_source->second.begin(), it_source->second.end());
        cl_kernel_info_map[cl_kernel_name].kernel_source = conv_kernel_source;
    }
    else
    {
        LOGE("can't find program %s!", cl_program_name.c_str());
        return -1;
    }

    return 0;
}

//NAIVE Methods
int NAIVE_Init_CL(std::map<std::string, clhpp_feather::CLKernelInfo>& cl_kernel_info_map)
{
    if (Insert("buffer_transform", "pad_int", cl_kernel_info_map) ||
            Insert("conv_1v1_buffer", "convolution", cl_kernel_info_map))
    {
        return -1;
    }

    return 0;
}


template <typename Dtype>
int NAIVE_Weight_Reform_CL(const ConvParam& param,
                           size_t n_grp_size,
                           size_t c_grp_size,
                           const Dtype* weight,
                           Dtype* weight_reformed)
{
    size_t w_num = param.output_channels;
    size_t w_channels = param.input_channels;
    size_t w_hw = param.kernel_h * param.kernel_w;

    for (int i = 0; i < w_num; ++i)
    {
        for (int k = 0; k < w_channels; ++k)
        {
            for (int j = 0; j < w_hw; ++j)
            {
                int src_idx = (i * w_channels + k) * w_hw + j;
                int dst_idx = (i / n_grp_size) * w_hw * param.padded_input_channels * n_grp_size +
                              j * param.padded_input_channels * n_grp_size +
                              (k / c_grp_size) * n_grp_size * c_grp_size +
                              (i % n_grp_size) * c_grp_size +
                              k % c_grp_size;
                weight_reformed[dst_idx] = weight[src_idx];
            }
        }
    }
    return 0;
}

int DEPTHWISE_Init_CL(std::map<std::string, clhpp_feather::CLKernelInfo>& cl_kernel_info_map)
{
    if (Insert("buffer_transform", "pad_int", cl_kernel_info_map) ||
            Insert("depthwise_conv_1v1_buffer", "convolution_depthwise", cl_kernel_info_map))
    {
        return -1;
    }

    return 0;
}


template <typename Dtype>
int DEPTHWISE_Weight_Reform_CL(const ConvParam& param,
                               size_t n_grp_size,
                               size_t c_grp_size,
                               const Dtype* weight,
                               Dtype* weight_reformed)
{
    size_t w_num = param.output_channels;
    size_t w_hw = param.kernel_h * param.kernel_w;

    for (int i = 0; i < w_num; ++i)
    {
        for (int j = 0; j < w_hw; ++j)
        {
            int dst_idx = (i / n_grp_size * w_hw + j)
                          * n_grp_size
                          + i % n_grp_size;
            int src_idx = i * w_hw + j;
            weight_reformed[dst_idx] = weight[src_idx];
        }
    }
    return 0;
}

int BOTH_Forward_CL(cl::CommandQueue cmd_q,
                    std::vector<std::string> kernel_names,
                    std::map<std::string, clhpp_feather::CLKernelInfo>& cl_kernel_info_map,
                    const ConvParam& param,
                    clhpp_feather::OpenCLRuntime* cl_runtime,
                    std::string layer_name)
{
    const clhpp_feather::CLKernelInfo& conv_kernel_info = cl_kernel_info_map[kernel_names[1]];
    const cl::Kernel& conv_kernel = conv_kernel_info.kernel;
    const std::vector<size_t>& conv_gws = conv_kernel_info.gws;
    const std::vector<size_t>& conv_lws = conv_kernel_info.lws;
#ifdef TIMING_CL
    cl::Event event;
    cmd_q.finish();
    timespec tpstart, tpend;
    clock_gettime(CLOCK_MONOTONIC, &tpstart);

    int error_num = cmd_q.enqueueNDRangeKernel(
                        conv_kernel, cl::NullRange, cl::NDRange(conv_gws[0], conv_gws[1], conv_gws[2]),
                        cl::NDRange(conv_lws[0], conv_lws[1], conv_lws[2]), nullptr, &event);

    if (!checkSuccess(error_num))
    {
        LOGE("Failed enqueuing the conv kernel.");
        return -1;
    }

    event.wait();
    clock_gettime(CLOCK_MONOTONIC, &tpend);
    double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
    LOGI("[%s] Execution time in %lf ms with %s\n", this->name().c_str(), timedif / 1000.0, k_name.c_str());

    cl::Event profileEvent = event;
    double queued_nanos_ = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
    double submit_nanos_ = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
    double start_nanos_  = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    double stop_nanos_   = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double submit_kerel_time = (submit_nanos_ - queued_nanos_) / 1000.0 / 1000.0;
    double start_kerel_time = (start_nanos_ - submit_nanos_) / 1000.0 / 1000.0;
    double stop_kerel_time = (stop_nanos_ - start_nanos_) / 1000.0 / 1000.0;
    LOGI("[%s] [%s] Execution time in kernel: %0.5f, %0.5f, %0.5f\n",
         this->name().c_str(), kernel_names[i].c_str(), submit_kerel_time, start_kerel_time, stop_kerel_time);
#else

    std::string key_gws = layer_name + "_" + kernel_names[1] + "_gws";
    std::string key_lws = layer_name + "_" + kernel_names[1] + "_lws";
    if (clhpp_feather::IsTuning())
    {
        //warm up
        int error_num = cmd_q.enqueueNDRangeKernel(
                            conv_kernel, cl::NullRange, cl::NDRange(conv_gws[0], conv_gws[1], conv_gws[2]),
                            cl::NDRange(conv_lws[0], conv_lws[1], conv_lws[2]), nullptr, nullptr);
        if (!checkSuccess(error_num))
        {
            LOGE("Failed enqueuing the conv kernel.");
            return -1;
        }
        //run
        std::vector<std::vector<size_t> > gws_list;
        std::vector<std::vector<size_t> > lws_list;
        gws_list.push_back(conv_gws);
        lws_list.push_back(conv_lws);
        uint64_t kwg_size = 0;
        cl_runtime->GetKernelMaxWorkGroupSize(conv_kernel, kwg_size);
        cl_runtime->tuner().TunerArry(kwg_size, param.output_h, param.output_w,
                                 conv_gws, conv_lws, gws_list, lws_list);
        double opt_time = std::numeric_limits<double>::max();
        int min_tune = -1;
        for (int j = 0; j < gws_list.size(); j++)
        {
            cmd_q.finish();
            timespec tpstart, tpend;
            clock_gettime(CLOCK_MONOTONIC, &tpstart);
            int error_num = cmd_q.enqueueNDRangeKernel(
                                conv_kernel, cl::NullRange, cl::NDRange(gws_list[j][0], gws_list[j][1], gws_list[j][2]),
                                cl::NDRange(lws_list[j][0], lws_list[j][1], lws_list[j][2]), nullptr, nullptr);
            if (!checkSuccess(error_num))
            {
                LOGE("Failed enqueuing the conv kernel.");
                return -1;
            }

            cmd_q.finish();
            clock_gettime(CLOCK_MONOTONIC, &tpend);
            double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
            timedif /= 1000.0;
            //LOGI("tuner kernel_name [%s] tuner %d cost %.3f ms", kernel_names[1].c_str(), j, timedif);
            if (timedif < opt_time)
            {
                opt_time = timedif;
                min_tune = j;
            }
        }
        
        cl_runtime->tuner().set_layer_kernel_wks(key_gws, gws_list[min_tune], opt_time);
        cl_runtime->tuner().set_layer_kernel_wks(key_lws, lws_list[min_tune], opt_time);
        //LOGI("tuner layer_name %s %s min_tune [%d]",layer_name.c_str(), key_gws.c_str(), min_tune);
    }
    else if (clhpp_feather::IsTunned())
    {
        std::vector<size_t> tmp_gws;
        std::vector<size_t> tmp_lws;
        cl_runtime->tuner().get_layer_kernel_wks(key_gws, tmp_gws);
        cl_runtime->tuner().get_layer_kernel_wks(key_lws, tmp_lws);
        int error_num = cmd_q.enqueueNDRangeKernel(
                            conv_kernel, cl::NullRange, cl::NDRange(tmp_gws[0], tmp_gws[1], tmp_gws[2]),
                            cl::NDRange(tmp_lws[0], tmp_lws[1], tmp_lws[2]), nullptr, nullptr);
        if (!checkSuccess(error_num))
        {
            LOGE("Failed enqueuing the conv kernel.");
            return -1;
        }
    }
    else if (clhpp_feather::IsTunerInProcess())
    {
        //run
        std::vector<std::vector<size_t> > gws_list;
        std::vector<std::vector<size_t> > lws_list;
        gws_list.push_back(conv_gws);
        lws_list.push_back(conv_lws);
        uint64_t kwg_size = 0;
        cl_runtime->GetKernelMaxWorkGroupSize(conv_kernel, kwg_size);
        cl_runtime->tuner().IsTunerInProcess(kwg_size, param.output_h, param.output_w,
                                 conv_gws, conv_lws, gws_list, lws_list);
        cmd_q.finish();
        timespec tpstart, tpend;
        clock_gettime(CLOCK_MONOTONIC, &tpstart);
        int j = 0;
        int error_num = cmd_q.enqueueNDRangeKernel(
                            conv_kernel, cl::NullRange, cl::NDRange(gws_list[j][0], gws_list[j][1], gws_list[j][2]),
                            cl::NDRange(lws_list[j][0], lws_list[j][1], lws_list[j][2]), nullptr, nullptr);
        if (!checkSuccess(error_num))
        {
            LOGE("Failed enqueuing the conv kernel.");
            return -1;
        }
        cmd_q.finish();
        clock_gettime(CLOCK_MONOTONIC, &tpend);
        double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
        timedif /= 1000.0;
        cl_runtime->tuner().set_layer_kernel_wks(key_gws, gws_list[j], timedif);
        cl_runtime->tuner().set_layer_kernel_wks(key_lws, lws_list[j], timedif);
    }
    else
    {
        int error_num = cmd_q.enqueueNDRangeKernel(
                            conv_kernel, cl::NullRange, cl::NDRange(conv_gws[0], conv_gws[1], conv_gws[2]),
                            cl::NDRange(conv_lws[0], conv_lws[1], conv_lws[2]), nullptr, nullptr);
        if (!checkSuccess(error_num))
        {
            LOGE("Failed enqueuing the conv kernel.");
            return -1;
        }
    }

#endif

    return 0;
}

int BOTH_Set_Conv_Kernel_Params_CL(const ConvParam& param,
                                   const CLBuffers& buffers,
                                   std::vector<std::string> kernel_names,
                                   std::map<std::string, clhpp_feather::CLKernelInfo>& cl_kernel_info_map,
                                   clhpp_feather::OpenCLRuntime* cl_runtime,
                                   bool is_reshape)
{
    int error_num;
    int param_idx = 0;
    bool set_kernel_arg_success = true;
    clhpp_feather::CLKernelInfo& conv_kernel_info = cl_kernel_info_map[kernel_names[1]];
    cl::Kernel& conv_kernel = conv_kernel_info.kernel;

    if (!is_reshape)
    {
        const cl::Program& conv_program = conv_kernel_info.program;
        const std::string& conv_kernel_name = conv_kernel_info.kernel_name;
        cl_runtime->BuildKernel(conv_kernel_name, cl_kernel_info_map);
        if (!checkSuccess(error_num))
        {
            LOGE("Failed to create conv OpenCL cl_conv_kernel.");
            return 1;
        }
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, *buffers. input_mem));
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, *buffers.weight_mem));
        if (param.bias_term)
        {
            set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, *buffers.bias_mem));
        }
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, *buffers.output_mem));
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, param.padded_input_channels));
        if (param.group != param.input_channels)
        {
            set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, param.padded_output_channels));
        }
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, param.input_h));
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, param.input_w));
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, param.output_h));
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, param.output_w));
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, param.kernel_h));
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, param.kernel_w));
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, param.stride_h));
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, param.stride_w));
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, param.pad_top));
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, param.pad_left));
        if (!set_kernel_arg_success)
        {
            LOGE("Failed setting conv OpenCL conv_kernel arguments.");
            return -1;
        }
    }
    else
    {
        param_idx = param.group != param.input_channels ? 6 : 5;
        int out_idx = param.bias_term ? 3 : 2;
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(0, *buffers.input_mem));
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(out_idx, *buffers.output_mem));
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, param.input_h));
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, param.input_w));
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, param.output_h));
        set_kernel_arg_success &= checkSuccess(conv_kernel.setArg(param_idx++, param.output_w));
        if (!set_kernel_arg_success)
        {
            LOGE("Failed setting conv reshape OpenCL conv_kernel arguments.");
            return -1;
        }

    }
    return 0;
}

int BOTH_Set_Conv_Work_Size_CL(const ConvParam& param,
                               std::map<std::string, clhpp_feather::CLKernelInfo>& cl_kernel_info_map,
                               std::vector<std::string> kernel_names,
                               clhpp_feather::OpenCLRuntime* cl_runtime)
{
    clhpp_feather::CLKernelInfo& conv_kernel_info = cl_kernel_info_map[kernel_names[1]];
    const cl::Kernel& conv_kernel = conv_kernel_info.kernel;
    std::vector<size_t>& conv_gws = conv_kernel_info.gws;
    std::vector<size_t>& conv_lws = conv_kernel_info.lws;

    if (conv_gws.size() != 0 || conv_lws.size() != 0)
    {
        conv_gws.clear();
        conv_lws.clear();
    }


    int h_lws = param.output_h > 32 ? 16 : 8;
    int w_lws = param.output_w > 32 ? 16 : 8;
    int c_blk_size = 4;
    if (param.padded_input_channels % 8 == 0 && param.padded_output_channels % 8 == 0)
    {
        c_blk_size = 8;
    }

    size_t conv_gws_dim0 = (param.output_h / h_lws + !!(param.output_h % h_lws)) * h_lws;
    size_t conv_gws_dim1 = (param.output_w / w_lws + !!(param.output_w % w_lws)) * w_lws;
    size_t conv_gws_dim2 = param.padded_output_channels / c_blk_size;

    size_t conv_lws_dim0 = h_lws;
    size_t conv_lws_dim1 = w_lws;
    size_t conv_lws_dim2 = (conv_gws_dim2 > 4 && conv_gws_dim2 % 4 == 0) ? 4 : 1;

    conv_gws.push_back(conv_gws_dim0);
    conv_gws.push_back(conv_gws_dim1);
    conv_gws.push_back(conv_gws_dim2);

    conv_lws.push_back(conv_lws_dim0);
    conv_lws.push_back(conv_lws_dim1);
    conv_lws.push_back(conv_lws_dim2);

    cl_runtime->FineTuneGroupSize(conv_kernel, param.output_h, param.output_w, conv_gws.data(), conv_lws.data());

    return 0;
}

int BOTH_Set_Build_Opts(const ConvParam& param,
                        bool is_fp16,
                        const std::vector<std::string>& kernel_names,
                        std::map<std::string, clhpp_feather::CLKernelInfo>& cl_kernel_info_map)
{
    clhpp_feather::CLKernelInfo& pad_kernel_info = cl_kernel_info_map[kernel_names[0]];
    std::vector<std::string>& pad_build_options = pad_kernel_info.build_options;

    clhpp_feather::CLKernelInfo& conv_kernel_info = cl_kernel_info_map[kernel_names[1]];
    std::vector<std::string>& build_options = conv_kernel_info.build_options;
    std::ostringstream ss;
    ss << param.channel_grp_size;
    build_options.push_back("-DN=" + ss.str());
    if (is_fp16)
        build_options.push_back("-DDATA_TYPE=half");
    else
        build_options.push_back("-DDATA_TYPE=float");

    pad_build_options = build_options;

    if (param.bias_term)
    {
        build_options.push_back("-DBIAS");
    }
    switch (param.activation)
    {
        case booster::ReLU:
            build_options.push_back("-DUSE_RELU");
            break;
        case booster::None:
            break;
        default:
            break;
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
    : Init(NULL), Forward(NULL)
{
}

template <class Dtype>
size_t ConvBoosterCL<Dtype>::GetWeightSize()
{
    return this->weight_size;
}

template <class Dtype>
const std::vector<std::string>& ConvBoosterCL<Dtype>::GetKernelNames()
{
    return this->kernel_names;
}
//Conditional algo selecter
template <class Dtype>
int ConvBoosterCL<Dtype>::SelectAlgo(ConvParam* param)
{
    if (param->group == param->input_channels)
    {
        this->algo = DEPTHWISE;
        this->weight_size = param->kernel_h * param->kernel_w * param->padded_output_channels;
    }
    else if (param->group == 1)
    {
        this->algo = NAIVE;
        this->weight_size = param->kernel_h * param->kernel_w * param->padded_output_channels * param->padded_input_channels;
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
            this->Forward = BOTH_Forward_CL;
            this->WeightReform = NAIVE_Weight_Reform_CL;
            this->SetConvKernelParams = BOTH_Set_Conv_Kernel_Params_CL;
            this->SetConvWorkSize = BOTH_Set_Conv_Work_Size_CL;
            this->SetBuildOpts = BOTH_Set_Build_Opts;
            this->kernel_names.push_back("pad_int");
            this->kernel_names.push_back("convolution");
            return 0;
        case DEPTHWISE:
            this->Init = DEPTHWISE_Init_CL;
            this->Forward = BOTH_Forward_CL;
            this->WeightReform = DEPTHWISE_Weight_Reform_CL;
            this->SetConvKernelParams = BOTH_Set_Conv_Kernel_Params_CL;
            this->SetConvWorkSize = BOTH_Set_Conv_Work_Size_CL;
            this->SetBuildOpts = BOTH_Set_Build_Opts;
            this->kernel_names.push_back("pad_int");
            this->kernel_names.push_back("convolution_depthwise");
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
