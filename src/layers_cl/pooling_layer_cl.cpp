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
#include "pooling_layer_cl.h"

namespace feather
{

template <class Dtype>
PoolingLayerCL<Dtype>::PoolingLayerCL(const LayerParameter *layer_param, RuntimeParameter<Dtype>* rt_param)
    : stride_height(1),
      stride_width(1),
      Layer<Dtype>(layer_param, rt_param)
{
    const PoolingParameter *pooling_param = layer_param->pooling_param();
    this->kernel_height = pooling_param->kernel_h();
    this->kernel_width = pooling_param->kernel_w();
    this->pad_height = pooling_param->pad_h();
    this->pad_width = pooling_param->pad_w();
    this->stride_height = pooling_param->stride_h();
    this->stride_width = pooling_param->stride_w();
    this->stride_height = (this->stride_height <= 0) ? 1 : this->stride_height;
    this->stride_width  = (this->stride_width  <= 0) ? 1 : this->stride_width;
    this->global_pooling = pooling_param->global_pooling();
    this->method = pooling_param->pool();

    InitCL();
}

template <class Dtype>
int PoolingLayerCL<Dtype>::InitCL()
{

    std::string program_name = "pooling_buffer";
    std::string kernel_name = "pooling";
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
int PoolingLayerCL<Dtype>::GenerateTopBlobs()
{
    //Only accept a single bottom blob.
    const Blob<Dtype> *bottom_blob = this->_bottom_blobs[this->_bottom[0]];
    this->input_height = bottom_blob->height();
    this->input_width = bottom_blob->width();
    this->input_channels = bottom_blob->channels();
    AssignOutputSize();
    this->output_channels = this->input_channels;
    this->_top_blobs[this->_top[0]] = new Blob<Dtype>(1, this->output_channels, this->output_height, this->output_width);
    this->_top_blobs[this->_top[0]]->AllocDevice(this->rt_param->context(), this->_top_blobs[this->_top[0]]->data_size_padded_c());
    SetWorkSize();

    return 0;
}


template <class Dtype>
int PoolingLayerCL<Dtype>::SetWorkSize()
{
    clhpp_feather::CLKernelInfo& pool_kernel_info = this->cl_kernel_info_map["pooling"];
    std::vector<size_t>& pool_gws = pool_kernel_info.gws;
    std::vector<size_t>& pool_lws = pool_kernel_info.lws;
    size_t padded_input_c = this->_bottom_blobs[this->_bottom[0]]->get_channels_padding();
    size_t padded_output_c = this->_top_blobs[this->_top[0]]->get_channels_padding();

    if (pool_gws.size() != 0 || pool_lws.size() != 0)
    {
        pool_gws.clear();
        pool_lws.clear();
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
    this->channel_block_size = c_blk_size;

    size_t pool_gws_dim0 = (this->output_height / h_lws + !!(this->output_height % h_lws)) * h_lws;
    size_t pool_gws_dim1 = (this->output_width / w_lws  + !!(this->output_width % w_lws)) * w_lws;
    size_t pool_gws_dim2 = padded_output_c / c_blk_size;

    size_t pool_lws_dim0 = h_lws;
    size_t pool_lws_dim1 = w_lws;
    size_t pool_lws_dim2 = (pool_gws_dim2 > 4 && pool_gws_dim2 % 4 == 0) ? 4 : 1;

    pool_gws.push_back(pool_gws_dim0);
    pool_gws.push_back(pool_gws_dim1);
    pool_gws.push_back(pool_gws_dim2);

    pool_lws.push_back(pool_lws_dim0);
    pool_lws.push_back(pool_lws_dim1);
    pool_lws.push_back(pool_lws_dim2);

    return 0;
}


template <class Dtype>
int PoolingLayerCL<Dtype>::SetBuildOptions()
{
    std::string ave_opt = "-DAVE_POOLING";
    clhpp_feather::CLKernelInfo& pool_kernel_info = this->cl_kernel_info_map["pooling"];
    std::vector<std::string>& build_options = pool_kernel_info.build_options;
    std::ostringstream ss;
    ss << this->channel_block_size;

    switch (this->method)
    {
        case PoolingParameter_::PoolMethod_MAX_:
            if (std::is_same<Dtype, uint16_t>::value)
                build_options.push_back("-DMIN_VAL=-HALF_MAX");
            else
                build_options.push_back("-DMIN_VAL=-FLT_MAX");

            break;
        case PoolingParameter_::PoolMethod_AVE:
            build_options.push_back(ave_opt);

            break;
        default:
            LOGE("Unsupported pool method\n");

            break;
    }

    build_options.push_back("-DN=" + ss.str());
    if (std::is_same<Dtype, uint16_t>::value)
        build_options.push_back("-DDATA_TYPE=half");
    else
        build_options.push_back("-DDATA_TYPE=float");
    return 0;
}

template <class Dtype>
int PoolingLayerCL<Dtype>::SetKernelParameters()
{
    int error_num;
    bool set_kernel_arguments_success = true;
    int param_idx = 0;
    this->rt_param->cl_runtime()->BuildKernel("pooling", this->cl_kernel_info_map);
    clhpp_feather::CLKernelInfo& pool_kernel_info = this->cl_kernel_info_map["pooling"];
    std::vector<size_t>& pool_gws = pool_kernel_info.gws;
    std::vector<size_t>& pool_lws = pool_kernel_info.lws;
    cl::Kernel& cl_kernel = pool_kernel_info.kernel;


    cl::Buffer* input_mem = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    cl::Buffer* output_mem = this->_top_blobs[this->_top[0]]->data_cl();
    uint32_t real_channels = this->_top_blobs[this->_top[0]]->get_channels_padding();

    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, *input_mem));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, *output_mem));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, real_channels));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->input_height));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->input_width));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->output_height));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->output_width));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->kernel_height));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->kernel_width));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->stride_height));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->stride_width));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->pad_height));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->pad_width));
    if (!set_kernel_arguments_success)
    {
        LOGE("Failed setting pooling OpenCL cl_kernels[0] arguments.");
        return 1;
    }

    this->rt_param->cl_runtime()->FineTuneGroupSize(cl_kernel, this->output_height, this->output_width, pool_gws.data(), pool_lws.data());

    // this->FineTuneGroupSize(this->cl_kernels[0], this->_top_blobs[this->_top[0]]->height(), this->_top_blobs[this->_top[0]]->width());
    return 0;
}

template <class Dtype>
int PoolingLayerCL<Dtype>::ForwardCL()
{
    clhpp_feather::CLKernelInfo& pool_kernel_info = this->cl_kernel_info_map["pooling"];
    std::vector<size_t>& pool_gws = pool_kernel_info.gws;
    std::vector<size_t>& pool_lws = pool_kernel_info.lws;
    cl::Kernel& cl_kernel = pool_kernel_info.kernel;
#ifdef TIMING_CL
    this->rt_param->command_queue().finish();
    timespec tpstart, tpend;
    cl::Event event;
    std::string cl_program_name = pool_kernel_info.program_name;
    clock_gettime(CLOCK_MONOTONIC, &tpstart);

    int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(cl_kernel, cl::NullRange, cl::NDRange(pool_gws[0], pool_gws[1], pool_gws[2]),
                    cl::NDRange(pool_lws[0], pool_lws[1], pool_lws[2]), nullptr, &event);
    if (!checkSuccess(error_num))
    {
        LOGE("Failed enqueuing the pooling kernel. %d", error_num);
        return -1;
    }

    this->cl_events[0].wait();
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

    std::string key_gws = this->name() + "_" + "pooling" + "_gws";
    std::string key_lws = this->name() + "_" + "pooling" + "_lws";
    if (clhpp_feather::IsTuning())
    {
        //warm up
        int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                            cl_kernel, cl::NullRange, cl::NDRange(pool_gws[0], pool_gws[1], pool_gws[2]),
                            cl::NDRange(pool_lws[0], pool_lws[1], pool_lws[2]), nullptr, nullptr);
        if (!checkSuccess(error_num))
        {
            LOGE("Failed enqueuing the element pooling kernel.");
            return -1;
        }
        //run
        std::vector<std::vector<size_t> > gws_list;
        std::vector<std::vector<size_t> > lws_list;
        gws_list.push_back(pool_gws);
        lws_list.push_back(pool_lws);
        uint64_t kwg_size = 0;
        this->rt_param->cl_runtime()->GetKernelMaxWorkGroupSize(cl_kernel, kwg_size);
        this->rt_param->cl_runtime()->tuner().TunerArry(kwg_size, this->output_height, this->output_width,
                pool_gws, pool_lws, gws_list, lws_list);
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
                LOGE("Failed enqueuing the pooling kernel.");
                return -1;
            }

            this->rt_param->command_queue().finish();
            clock_gettime(CLOCK_MONOTONIC, &tpend);
            double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
            timedif /= 1000.0;
            //LOGI("tuner kernel_name [inner_product] tuner %d cost %.3f ms", j, timedif);
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
            LOGE("Failed enqueuing the pooling kernel.");
            return -1;
        }
    }
    else if (clhpp_feather::IsTunerInProcess())
    {
        //run
        std::vector<std::vector<size_t> > gws_list;
        std::vector<std::vector<size_t> > lws_list;
        gws_list.push_back(pool_gws);
        lws_list.push_back(pool_lws);
        uint64_t kwg_size = 0;
        this->rt_param->cl_runtime()->GetKernelMaxWorkGroupSize(cl_kernel, kwg_size);
        this->rt_param->cl_runtime()->tuner().IsTunerInProcess(kwg_size, this->output_height, this->output_width,
                pool_gws, pool_lws, gws_list, lws_list);
        this->rt_param->command_queue().finish();
        timespec tpstart, tpend;
        clock_gettime(CLOCK_MONOTONIC, &tpstart);
        int j = 0;
        int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                            cl_kernel, cl::NullRange, cl::NDRange(gws_list[j][0], gws_list[j][1], gws_list[j][2]),
                            cl::NDRange(lws_list[j][0], lws_list[j][1], lws_list[j][2]), nullptr, nullptr);
        if (!checkSuccess(error_num))
        {
            LOGE("Failed enqueuing the pooling kernel.");
            return -1;
        }
        this->rt_param->command_queue().finish();
        clock_gettime(CLOCK_MONOTONIC, &tpend);
        double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
        timedif /= 1000.0;
        this->rt_param->cl_runtime()->tuner().set_layer_kernel_wks(key_gws, gws_list[j], timedif);
        this->rt_param->cl_runtime()->tuner().set_layer_kernel_wks(key_lws, lws_list[j], timedif);
    }
    else
    {
        int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(cl_kernel, cl::NullRange, cl::NDRange(pool_gws[0], pool_gws[1], pool_gws[2]),
                        cl::NDRange(pool_lws[0], pool_lws[1], pool_lws[2]), nullptr, nullptr);
        if (!checkSuccess(error_num))
        {
            LOGE("Failed enqueuing the pooling kernel. %d", error_num);
            return -1;
        }
    }

#endif
    return 0;
}

template <class Dtype>
int PoolingLayerCL<Dtype>::ForwardReshapeCL()
{
    if (this->input_height == this->_bottom_blobs[this->_bottom[0]]->height() &&
            this->input_width == this->_bottom_blobs[this->_bottom[0]]->width())
        return this->ForwardCL();

    bool set_kernel_arg_success = true;
    clhpp_feather::CLKernelInfo& pool_kernel_info = this->cl_kernel_info_map["pooling"];
    std::vector<size_t>& pool_gws = pool_kernel_info.gws;
    std::vector<size_t>& pool_lws = pool_kernel_info.lws;
    cl::Kernel& cl_kernel = pool_kernel_info.kernel;

    this->input_height = this->_bottom_blobs[this->_bottom[0]]->height();
    this->input_width = this->_bottom_blobs[this->_bottom[0]]->width();
    AssignOutputSize();
    if (this->output_height <= 1 || this->output_width < 1)
    {
        LOGE("invalid output size in forward reshape");
        return -1;
    }
    if (this->_top_blobs[this->_top[0]]->ReshapeWithReallocDevice(this->rt_param->context(),
            this->_top_blobs[this->_top[0]]->num(),
            this->_top_blobs[this->_top[0]]->channels(),
            this->output_height, this->output_width) == 2)
    {
        cl::Buffer* output_mem = this->_top_blobs[this->_top[0]]->data_cl();
        set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(1, *output_mem));
    }

    int param_idx = 3;
    cl::Buffer* input_mem = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(0, *input_mem));
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->input_height));
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->input_width));
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->output_height));
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->output_width));


    if (!set_kernel_arg_success)
    {
        LOGE("Failed setting pooling reshape cl_kernels arguments.");
        return 1;
    }
    this->ResetWorkSize("pooling", this->output_height, this->output_width);
    this->rt_param->cl_runtime()->FineTuneGroupSize(cl_kernel, this->output_height, this->output_width, pool_gws.data(), pool_lws.data());
    return this->ForwardCL();
}


template <class Dtype>
inline void PoolingLayerCL<Dtype>::AssignOutputSize()
{
    if (this->global_pooling)
    {
        this->kernel_height = this->input_height;
        this->kernel_width = this->input_width;
        this->output_height = 1;
        this->output_width = 1;
    }
    else
    {
        //General pooling.
        this->output_height = static_cast<int>(ceil(static_cast<float>(this->input_height + 2 * this->pad_height - this->kernel_height) / this->stride_height)) + 1;
        this->output_width = static_cast<int>(ceil(static_cast<float>(this->input_width + 2 * this->pad_width - this->kernel_width) / this->stride_width)) + 1;
    }
}


template class PoolingLayerCL<float>;
template class PoolingLayerCL<uint16_t>;

}; // namespace feather
