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

#include "inner_product_layer_cl.h"

namespace feather
{

template <class Dtype>
InnerProductLayerCL<Dtype>::InnerProductLayerCL(const LayerParameter *layer_param, RuntimeParameter<Dtype>* rt_param)
    : fuse_relu(false), Layer<Dtype>(layer_param, rt_param)
{
    const InnerProductParameter *inner_product_param = layer_param->inner_product_param();
    bias_term = inner_product_param->bias_term();
    this->output_channels = this->_weight_blobs[0]->num();
    assert(this->_weight_blobs.size() > 0);
    this->kernel_data = this->_weight_blobs[0]->data();
    if (this->bias_term)
    {
        assert(this->_weight_blobs.size() == 2);
        this->bias_data = this->_weight_blobs[1]->data();
    }
    this->_fusible = true;

    InitCL();
}


template <class Dtype>
int InnerProductLayerCL<Dtype>::SetWorkSize()
{
    clhpp_feather::CLKernelInfo& fc_kernel_info = this->cl_kernel_info_map["inner_product"];
    std::vector<size_t>& fc_gws = fc_kernel_info.gws;
    std::vector<size_t>& fc_lws = fc_kernel_info.lws;
    size_t padded_input_c = this->_bottom_blobs[this->_bottom[0]]->get_channels_padding();
    size_t padded_output_c = this->_top_blobs[this->_top[0]]->get_channels_padding();
    if (fc_gws.size() != 0 || fc_lws.size() != 0)
    {
        fc_gws.clear();
        fc_lws.clear();
    }
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
    size_t fc_gws_dim0 = 1;
    size_t fc_gws_dim1 = 1;
    size_t fc_gws_dim2 = padded_output_c / c_blk_size;
    size_t fc_lws_dim0 = 1;
    size_t fc_lws_dim1 = 1;
    size_t fc_lws_dim2 = (fc_gws_dim2 > 4 && fc_gws_dim2 % 4 == 0) ? 4 : 1;
    fc_gws.push_back(fc_gws_dim0);
    fc_gws.push_back(fc_gws_dim0);
    fc_gws.push_back(fc_gws_dim2);
    fc_lws.push_back(fc_lws_dim0);
    fc_lws.push_back(fc_lws_dim1);
    fc_lws.push_back(fc_lws_dim2);
    return 0;
}

template <class Dtype>
int InnerProductLayerCL<Dtype>::InitCL()
{
    std::string program_name = "inner_product_buffer";
    std::string kernel_name = "inner_product";
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
int InnerProductLayerCL<Dtype>::SetBuildOptions()
{
    std::ostringstream ss;
    clhpp_feather::CLKernelInfo& fc_kernel_info = this->cl_kernel_info_map["inner_product"];
    std::vector<std::string>& build_options = fc_kernel_info.build_options;
    ss << this->channel_grp_size;

    build_options.push_back("-DN=" + ss.str());
    if (std::is_same<Dtype, uint16_t>::value)
        build_options.push_back("-DDATA_TYPE=half");
    else
        build_options.push_back("-DDATA_TYPE=float");

    if (bias_term)
    {
        build_options.push_back("-DBIAS");
    }
    if (fuse_relu)
    {
        build_options.push_back("-DUSE_RELU");
    }
    return 0;
}

template <class Dtype>
int InnerProductLayerCL<Dtype>::SetKernelParameters()
{
    int error_num;
    bool set_kernel_arg_success = true;
    int param_idx = 0;
    this->rt_param->cl_runtime()->BuildKernel("inner_product", this->cl_kernel_info_map);
    clhpp_feather::CLKernelInfo& fc_kernel_info = this->cl_kernel_info_map["inner_product"];
    std::vector<size_t>& fc_gws = fc_kernel_info.gws;
    std::vector<size_t>& fc_lws = fc_kernel_info.lws;
    cl::Kernel& cl_kernel = fc_kernel_info.kernel;

    uint32_t b_channel_padding = this->_bottom_blobs[this->_bottom[0]]->get_channels_padding();

    uint32_t w_num = this->_weight_blobs[0]->num();
    uint32_t w_channel = this->_weight_blobs[0]->channels();

    uint32_t real_weight_size = this->input_height * this->input_width * this->_weight_blobs[0]->get_num_padding() * b_channel_padding;

    this->_weight_blobs[0]->AllocDevice(this->rt_param->context(), real_weight_size);

    std::vector<Dtype> weight_padding(real_weight_size, 0.0f);

    uint32_t kernel_size = this->input_height * this->input_width * b_channel_padding;
    uint32_t hw_size = this->input_height * this->input_width;
    uint32_t num_channel_grp = b_channel_padding / this->channel_grp_size;
    uint32_t c_grp_size = 1 /* this->channel_grp_size */;
    uint32_t n_grp_size = this->channel_grp_size;

    for (int n = 0; n < w_num; ++n)
    {
        for (int m = 0; m < w_channel; ++m)
        {
            int c_idx = m / hw_size;
            int hw_idx = m % hw_size;
            int src_idx = n * w_channel + m;
            /* naive arrangement */
            //int dst_idx = (n * this->input_height * this->input_width + hw_idx) * b_channel_padding + c_idx;

            /* re-arrangement*/
            int dst_idx = (n / n_grp_size) * hw_size * b_channel_padding * n_grp_size +
                          hw_idx * b_channel_padding * n_grp_size +
                          (c_idx / c_grp_size) * n_grp_size * c_grp_size +
                          (n % n_grp_size) * c_grp_size +
                          c_idx % c_grp_size;
            //printf("(%d, %d, %d) %d, %d\n", n, c_idx, hw_idx, src_idx, dst_idx);

            weight_padding[dst_idx] = kernel_data[src_idx];
        }
    }

    this->_weight_blobs[0]->WriteToDevice(this->rt_param->command_queue(), weight_padding.data(), real_weight_size);
    this->_weight_blobs[0]->Free();

    if (bias_term)
    {
        uint32_t real_bias_size = this->_weight_blobs[0]->get_num_padding();
        this->_weight_blobs[1]->AllocDevice(this->rt_param->context(), real_bias_size);
        std::vector<Dtype> bias_padding(real_bias_size, 0);
        memcpy(bias_padding.data(), bias_data, w_num * sizeof(Dtype));
        this->_weight_blobs[1]->WriteToDevice(this->rt_param->command_queue(), bias_padding.data(), real_bias_size);
        this->_weight_blobs[1]->Free();
    }


    cl::Buffer* input_mem = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    cl::Buffer* weight_mem = this->_weight_blobs[0]->data_cl();
    cl::Buffer* output_mem = this->_top_blobs[this->_top[0]]->data_cl();
    uint32_t out_real_channels = this->_top_blobs[this->_top[0]]->get_channels_padding();
    uint32_t use_relu = fuse_relu;

    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(param_idx++, *input_mem));
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(param_idx++, *weight_mem));
    if (bias_term)
    {
        set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(param_idx++, *this->_weight_blobs[1]->data_cl()));
    }
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(param_idx++, *output_mem));
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(param_idx++, b_channel_padding));
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(param_idx++, out_real_channels));
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->input_height));
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->input_width));

    this->rt_param->cl_runtime()->FineTuneGroupSize(cl_kernel, 1, 1, fc_gws.data(), fc_lws.data());
    if (!set_kernel_arg_success)
    {
        LOGE("Failed setting inner product OpenCL cl_kernels[0] arguments. ");
        return 1;
    }
    return 0;
}

template <class Dtype>
int InnerProductLayerCL<Dtype>::ForwardCL()
{
    clhpp_feather::CLKernelInfo& fc_kernel_info = this->cl_kernel_info_map["inner_product"];
    std::vector<size_t>& fc_gws = fc_kernel_info.gws;
    std::vector<size_t>& fc_lws = fc_kernel_info.lws;
    cl::Kernel& cl_kernel = fc_kernel_info.kernel;

#ifdef TIMING_CL
    this->rt_param->command_queue().finish();
    std::string cl_program_name = fc_kernel_info.program_name;
    timespec tpstart, tpend;
    cl::Event event;
    clock_gettime(CLOCK_MONOTONIC, &tpstart);

    int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                        cl_kernel, cl::NullRange, cl::NDRange(fc_gws[0], fc_gws[1], fc_gws[2]),
                        cl::NDRange(fc_lws[0], fc_lws[1], fc_lws[2]), nullptr, &event);

    if (!checkSuccess(error_num))
    {
        LOGE("Failed enqueuing the inner product kernel.");
        return -1;
    }
    event.wait();
    clock_gettime(CLOCK_MONOTONIC, &tpend);
    double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
    LOGI("[%s] Execution time in %lf ms with %s\n", this->name().c_str(), timedif / 1000.0, cl_program_name.c_str());

    cl::Event profileEvent = event[0];
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
    int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                        cl_kernel, cl::NullRange, cl::NDRange(fc_gws[0], fc_gws[1], fc_gws[2]),
                        cl::NDRange(fc_lws[0], fc_lws[1], fc_lws[2]), nullptr, nullptr);

    if (!checkSuccess(error_num))
    {
        LOGE("Failed enqueuing the inner product kernel.");
        return -1;
    }

#endif

    return 0;
}

template <class Dtype>
int InnerProductLayerCL<Dtype>::ForwardReshapeCL()
{
    if (this->input_height == this->_bottom_blobs[this->_bottom[0]]->height() &&
            this->input_width == this->_bottom_blobs[this->_bottom[0]]->width())
        return this->ForwardCL();
    else
    {
        LOGE("inner product does not support variable input size");
        return -1;
    }
}

template <class Dtype>
int InnerProductLayerCL<Dtype>::GenerateTopBlobs()
{
    const Blob<Dtype> *bottom_blob = this->_bottom_blobs[this->_bottom[0]];
    this->input_width = bottom_blob->width();
    this->input_height = bottom_blob->height();
    this->_top_blobs[this->_top[0]] = new Blob<Dtype>(1, this->output_channels, 1, 1);
    this->_top_blobs[this->_top[0]]->AllocDevice(this->rt_param->context(), this->_top_blobs[this->_top[0]]->data_size_padded_c());

    SetWorkSize();

    return 0;
}



template <class Dtype>
int InnerProductLayerCL<Dtype>::Fuse(Layer<Dtype> *next_layer)
{
    if (next_layer->type().compare("ReLU") == 0)
    {
        fuse_relu = true;
        return 1;
    }
    else
    {
        return 0;
    }
}

template class InnerProductLayerCL<float>;
template class InnerProductLayerCL<uint16_t>;

}; // namespace feather
