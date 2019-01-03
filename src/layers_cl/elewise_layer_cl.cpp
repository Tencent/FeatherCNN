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
#include "elewise_layer_cl.h"

using namespace std;

namespace feather
{

template <class Dtype>
int EltwiseLayerCL<Dtype>::InitCL()
{
    std::string program_name = "eltwise_buffer";
    std::string kernel_name = "eltwise";
    auto it_source = booster::opencl_kernel_string_map.find(program_name);
    if (it_source != booster::opencl_kernel_string_map.end())
    {
        this->cl_kernel_info_map[kernel_name].program_name = program_name;
        this->cl_kernel_info_map[kernel_name].kernel_name = kernel_name;
        this->cl_kernel_info_map[kernel_name].kernel_source = std::string(it_source->second.begin(), it_source->second.end());
    }
    else
    {
        LOGE("can't find program %s!", program_name.c_str());
        return -1;
    }
    return 0;
}

template <class Dtype>
int EltwiseLayerCL<Dtype>::GenerateTopBlobs()
{
    assert(this->_bottom.size() == 2);
    assert(this->_bottom_blobs.size() == 2);
    assert(this->_bottom_blobs[this->_bottom[0]]->data_size() == this->_bottom_blobs[this->_bottom[1]]->data_size());
    Blob<Dtype>* p_blob = new Blob<Dtype>();
    p_blob->CopyShape(this->_bottom_blobs[this->_bottom[0]]);
    p_blob->AllocDevice(this->rt_param->context(), p_blob->data_size_padded_c());
    this->output_height = p_blob->height();
    this->output_width = p_blob->width();
    this->output_channels = p_blob->get_channels_padding();
    this->_top_blobs[this->_top[0]] = p_blob;
    SetWorkSize();

    return 0;
}

template <class Dtype>
int EltwiseLayerCL<Dtype>::SetWorkSize()
{
    clhpp_feather::CLKernelInfo& eltwise_kernel_info = this->cl_kernel_info_map["eltwise"];
    std::vector<size_t>& eltwise_gws = eltwise_kernel_info.gws;
    std::vector<size_t>& eltwise_lws = eltwise_kernel_info.lws;
    size_t padded_input_c = this->_bottom_blobs[this->_bottom[0]]->get_channels_padding();
    size_t padded_output_c = this->_top_blobs[this->_top[0]]->get_channels_padding();

    if (eltwise_gws.size() != 0 || eltwise_lws.size() != 0)
    {
        eltwise_gws.clear();
        eltwise_lws.clear();
    }

    int h_lws = this->output_height > 32 ? 16 : 8;
    int w_lws = this->output_width > 32 ? 16 : 8;

    int c_blk_size = 4;
    if (padded_input_c % 16 == 0 && padded_output_c % 16 == 0)
    {
        c_blk_size = 16;
    }
    else if (padded_input_c % 8 == 0 && padded_output_c % 8 == 0)
    {
        c_blk_size = 8;
    }
    this->channel_grp_size = c_blk_size;

    size_t eltwise_gws_dim0 = (this->output_height / h_lws + !!(this->output_height % h_lws)) * h_lws;
    size_t eltwise_gws_dim1 = (this->output_width / w_lws  + !!(this->output_width % w_lws)) * w_lws;
    size_t eltwise_gws_dim2 = padded_output_c / c_blk_size;

    size_t eltwise_lws_dim0 = h_lws;
    size_t eltwise_lws_dim1 = w_lws;
    size_t eltwise_lws_dim2 = (eltwise_gws_dim2 > 4 && eltwise_gws_dim2 % 4 == 0) ? 4 : 1;

    eltwise_gws.push_back(eltwise_gws_dim0);
    eltwise_gws.push_back(eltwise_gws_dim1);
    eltwise_gws.push_back(eltwise_gws_dim2);

    eltwise_lws.push_back(eltwise_lws_dim0);
    eltwise_lws.push_back(eltwise_lws_dim1);
    eltwise_lws.push_back(eltwise_lws_dim2);

    return 0;
}

template <class Dtype>
int EltwiseLayerCL<Dtype>::ResetWorkSize()
{
    clhpp_feather::CLKernelInfo& eltwise_kernel_info = this->cl_kernel_info_map["eltwise"];
    std::vector<size_t>& eltwise_gws = eltwise_kernel_info.gws;
    std::vector<size_t>& eltwise_lws = eltwise_kernel_info.lws;

    int h_lws = this->output_height > 32 ? 16 : 8;
    int w_lws = this->output_width > 32 ? 16 : 8;

    size_t eltwise_gws_dim0 = (this->output_height / h_lws + !!(this->output_height % h_lws)) * h_lws;
    size_t eltwise_gws_dim1 = (this->output_width / w_lws  + !!(this->output_width % w_lws)) * w_lws;

    size_t eltwise_lws_dim0 = h_lws;
    size_t eltwise_lws_dim1 = w_lws;

    eltwise_gws[0] = eltwise_gws_dim0;
    eltwise_gws[1] = eltwise_gws_dim1;
    eltwise_lws[0] = eltwise_lws_dim0;
    eltwise_lws[1] = eltwise_lws_dim1;

    return 0;
}

template <class Dtype>
int EltwiseLayerCL<Dtype>::SetBuildOptions()
{
    std::ostringstream ss;
    clhpp_feather::CLKernelInfo& eltwise_kernel_info = this->cl_kernel_info_map["eltwise"];
    std::vector<std::string>& build_options = eltwise_kernel_info.build_options;
    ss << this->channel_grp_size;
    build_options.push_back("-DN=" + ss.str());
    if (std::is_same<Dtype, uint16_t>::value)
        build_options.push_back("-DDATA_TYPE=half");
    else
        build_options.push_back("-DDATA_TYPE=float");
    return 0;
}



template <class Dtype>
int EltwiseLayerCL<Dtype>::SetKernelParameters()
{
    int error_num;
    bool set_kernel_arguments_success = true;
    int param_idx = 0;
    this->rt_param->cl_runtime()->BuildKernel("eltwise", this->cl_kernel_info_map);
    clhpp_feather::CLKernelInfo& eltwise_kernel_info = this->cl_kernel_info_map["eltwise"];
    std::vector<size_t>& eltwise_gws = eltwise_kernel_info.gws;
    std::vector<size_t>& eltwise_lws = eltwise_kernel_info.lws;
    cl::Kernel& cl_kernel = eltwise_kernel_info.kernel;

    cl::Buffer* input_mem1 = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    cl::Buffer* input_mem2 = this->_bottom_blobs[this->_bottom[1]]->data_cl();
    cl::Buffer* output_mem = this->_top_blobs[this->_top[0]]->data_cl();


    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, *input_mem1));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, *input_mem2));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, *output_mem));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->output_height));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->output_width));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->output_channels));

    this->rt_param->cl_runtime()->FineTuneGroupSize(cl_kernel, this->output_height, this->output_width, eltwise_gws.data(), eltwise_lws.data());
    if (!set_kernel_arguments_success)
    {
        LOGE("Failed setting inner product OpenCL cl_kernels[0] arguments. ");
        return -1;
    }

    return 0;
}

template <class Dtype>
int EltwiseLayerCL<Dtype>::ForwardReshapeCL()
{
    if (this->output_height == this->_bottom_blobs[this->_bottom[0]]->height() &&
            this->output_width == this->_bottom_blobs[this->_bottom[0]]->width())
        return this->ForwardCL();

    bool set_kernel_arg_success = true;
    clhpp_feather::CLKernelInfo& eltwise_kernel_info = this->cl_kernel_info_map["eltwise"];
    std::vector<size_t>& eltwise_gws = eltwise_kernel_info.gws;
    std::vector<size_t>& eltwise_lws = eltwise_kernel_info.lws;
    cl::Kernel& cl_kernel = eltwise_kernel_info.kernel;


    this->output_height = this->_bottom_blobs[this->_bottom[0]]->height();
    this->output_width = this->_bottom_blobs[this->_bottom[0]]->width();
    if (this->_top_blobs[this->_top[0]]->ReshapeWithReallocDevice(this->rt_param->context(),
            this->_top_blobs[this->_top[0]]->num(),
            this->_top_blobs[this->_top[0]]->channels(),
            this->output_height, this->output_width) == 2)
    {
        cl::Buffer* output_mem = this->_top_blobs[this->_top[0]]->data_cl();
        set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(2, *output_mem));
    }

    cl::Buffer* input_mem1 = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    cl::Buffer* input_mem2 = this->_bottom_blobs[this->_bottom[1]]->data_cl();
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(0, *input_mem1));
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(1, *input_mem2));
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(3, this->output_height));
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(4, this->output_width));

    if (!set_kernel_arg_success)
    {
        LOGE("Failed setting conv OpenCL cl_kernels[0] arguments.");
        return 1;
    }

    this->ResetWorkSize();
    this->rt_param->cl_runtime()->FineTuneGroupSize(cl_kernel, this->output_height, this->output_width, eltwise_gws.data(), eltwise_lws.data());
    return this->ForwardCL();
}

template <class Dtype>
int EltwiseLayerCL<Dtype>::ForwardCL()
{
    clhpp_feather::CLKernelInfo& eltwise_kernel_info = this->cl_kernel_info_map["eltwise"];
    std::vector<size_t>& eltwise_gws = eltwise_kernel_info.gws;
    std::vector<size_t>& eltwise_lws = eltwise_kernel_info.lws;
    cl::Kernel& cl_kernel = eltwise_kernel_info.kernel;

#ifdef TIMING_CL
    this->rt_param->command_queue().finish();
    std::string cl_program_name = eltwise_kernel_info.program_name;
    timespec tpstart, tpend;
    cl::Event event;
    clock_gettime(CLOCK_MONOTONIC, &tpstart);
    int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                        cl_kernel, cl::NullRange, cl::NDRange(eltwise_gws[0], eltwise_gws[1], eltwise_gws[2]),
                        cl::NDRange(eltwise_lws[0], eltwise_lws[1], eltwise_lws[2]), nullptr, &event);
    if (!checkSuccess(error_num))
    {
        LOGE("Failed enqueuing the element wise kernel.");
        return -1;
    }

    event.wait();
    clock_gettime(CLOCK_MONOTONIC, &tpend);
    double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
    LOGI("[%s] Execution time in %lf ms with %s\n", this->name().c_str(), timedif / 1000.0, cl_program_name.c_str());

    cl::Event profileEvent = event;
    double queued_nanos_ = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
    double submit_nanos_ = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
    double start_nanos_  = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    double stop_nanos_   = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double submit_kerel_time = (submit_nanos_ - queued_nanos_) / 1000.0 / 1000.0;
    double start_kerel_time = (start_nanos_ - submit_nanos_) / 1000.0 / 1000.0;
    double stop_kerel_time = (stop_nanos_ - start_nanos_) / 1000.0 / 1000.0;
    LOGI("[%s] [%s] Execution time in kernel: %0.5f, %0.5f, %0.5f\n",
         this->name().c_str(), cl_program_name.c_str(), submit_kerel_time, start_kerel_time, stop_kerel_time);

#else
    std::string key_gws = this->name() + "_" + "eltwise" + "_gws";
    std::string key_lws = this->name() + "_" + "eltwise" + "_lws";
    if (clhpp_feather::IsTuning())
    {
        //warm up
        int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                        cl_kernel, cl::NullRange, cl::NDRange(eltwise_gws[0], eltwise_gws[1], eltwise_gws[2]),
                        cl::NDRange(eltwise_lws[0], eltwise_lws[1], eltwise_lws[2]), nullptr, nullptr);
        if (!checkSuccess(error_num))
        {
            LOGE("Failed enqueuing the element wise kernel.");
            return -1;
        }
        //run
        std::vector<std::vector<size_t> > gws_list;
        std::vector<std::vector<size_t> > lws_list;
        gws_list.push_back(eltwise_gws);
        lws_list.push_back(eltwise_lws);
        uint64_t kwg_size = 0;
        this->rt_param->cl_runtime()->GetKernelMaxWorkGroupSize(cl_kernel, kwg_size);
        this->rt_param->cl_runtime()->tuner().TunerArry(kwg_size, this->output_height, this->output_width,
                                 eltwise_gws, eltwise_lws, gws_list, lws_list);
        double opt_time = std::numeric_limits<double>::max();
        int min_tune = -1;
        for (int j = 0; j < gws_list.size(); j++)
        {
            this->rt_param->command_queue().finish();
            timespec tpstart, tpend;
            clock_gettime(CLOCK_MONOTONIC, &tpstart);
            int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                                cl_kernel, cl::NullRange, cl::NDRange(gws_list[j][0], gws_list[j][1], gws_list[j][2]),
                                cl::NDRange(lws_list[j][0], lws_list[j][1], lws_list[j][2]), nullptr, nullptr);
            if (!checkSuccess(error_num))
            {
                LOGE("Failed enqueuing the conv kernel.");
                return -1;
            }

            this->rt_param->command_queue().finish();
            clock_gettime(CLOCK_MONOTONIC, &tpend);
            double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
            timedif /= 1000.0;
            //LOGI("tuner kernel_name [elewise] tuner %d cost %.3f ms", j, timedif);
            if (timedif < opt_time)
            {
                opt_time = timedif;
                min_tune = j;
            }
        }
        
        this->rt_param->cl_runtime()->tuner().set_layer_kernel_wks(key_gws, gws_list[min_tune], opt_time);
        this->rt_param->cl_runtime()->tuner().set_layer_kernel_wks(key_lws, lws_list[min_tune], opt_time);
        //LOGI("tuner layer_name %s %s min_tune [%d]",layer_name.c_str(), key_gws.c_str(), min_tune);
    }
    else if (clhpp_feather::IsTunned())
    {
        std::vector<size_t> tmp_gws;
        std::vector<size_t> tmp_lws;
        this->rt_param->cl_runtime()->tuner().get_layer_kernel_wks(key_gws, tmp_gws);
        this->rt_param->cl_runtime()->tuner().get_layer_kernel_wks(key_lws, tmp_lws);
        int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                            cl_kernel, cl::NullRange, cl::NDRange(tmp_gws[0], tmp_gws[1], tmp_gws[2]),
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
        gws_list.push_back(eltwise_gws);
        lws_list.push_back(eltwise_lws);
        uint64_t kwg_size = 0;
        this->rt_param->cl_runtime()->GetKernelMaxWorkGroupSize(cl_kernel, kwg_size);
        this->rt_param->cl_runtime()->tuner().TunerArry(kwg_size, this->output_height, this->output_width,
                                 eltwise_gws, eltwise_lws, gws_list, lws_list);
        this->rt_param->command_queue().finish();
        timespec tpstart, tpend;
        clock_gettime(CLOCK_MONOTONIC, &tpstart);
        int j = 0;
        int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                            cl_kernel, cl::NullRange, cl::NDRange(gws_list[j][0], gws_list[j][1], gws_list[j][2]),
                            cl::NDRange(lws_list[j][0], lws_list[j][1], lws_list[j][2]), nullptr, nullptr);
        if (!checkSuccess(error_num))
        {
            LOGE("Failed enqueuing the conv kernel.");
            return -1;
        }
        this->rt_param->command_queue().finish();
        clock_gettime(CLOCK_MONOTONIC, &tpend);
        double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
        timedif /= 1000.0;
        this->rt_param->cl_runtime()->tuner().set_layer_kernel_wks(key_gws, gws_list[j], timedif);
        this->rt_param->cl_runtime()->tuner().set_layer_kernel_wks(key_lws, lws_list[j], timedif);
    }
    else {
        int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                        cl_kernel, cl::NullRange, cl::NDRange(eltwise_gws[0], eltwise_gws[1], eltwise_gws[2]),
                        cl::NDRange(eltwise_lws[0], eltwise_lws[1], eltwise_lws[2]), nullptr, nullptr);
        if (!checkSuccess(error_num))
        {
            LOGE("Failed enqueuing the element wise kernel.");
            return -1;
        }
    }  
#endif

    return 0;
}

template class EltwiseLayerCL<float>;
template class EltwiseLayerCL<uint16_t>;

}; // namespace feather
