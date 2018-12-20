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

#include "inner_product_layer_cl.h"

namespace feather {

template <class Dtype>
InnerProductLayerCL<Dtype>::InnerProductLayerCL(const LayerParameter *layer_param, RuntimeParameter<float>* rt_param)
      : fuse_relu(false), Layer<Dtype>(layer_param, rt_param) {
    const InnerProductParameter *inner_product_param = layer_param->inner_product_param();
    bias_term = inner_product_param->bias_term();
    this->output_channels = this->_weight_blobs[0]->num();
    assert(this->_weight_blobs.size() > 0);
    this->kernel_data = this->_weight_blobs[0]->data();
    if (this->bias_term) {
      assert(this->_weight_blobs.size() == 2);
      this->bias_data = this->_weight_blobs[1]->data();
    }
    this->_fusible = true;

    InitCL();
  }

template <class Dtype>
int InnerProductLayerCL<Dtype>::InitCL() {
    std::string func_name = "inner_product";
    this->cl_kernel_functions.push_back(func_name);
    std::string kernel_name_inner_product = "inner_product_buffer";
    auto it_source1 = booster::opencl_kernel_string_map.find("inner_product_buffer");
    std::string kernel_str_inner_product(it_source1->second.begin(),it_source1->second.end());

    this->cl_kernel_names.push_back(kernel_name_inner_product);
    this->cl_kernel_symbols.push_back(kernel_str_inner_product);

    cl::Kernel kernel;
    this->kernels.push_back(kernel);
    cl::Event event;
    this->events.push_back(event);

    return 0;
}

template <class Dtype>
int InnerProductLayerCL<Dtype>::SetBuildOptions() {
    std::ostringstream ss;
    ss << this->channel_group_size;
    this->build_options.push_back("-DN=" + ss.str());
    if(std::is_same<Dtype, uint16_t>::value)
      this->build_options.push_back("-DDATA_TYPE=half");
    else
      this->build_options.push_back("-DDATA_TYPE=float");

    if (bias_term) {
      this->build_options.push_back("-DBIAS");
    }
    if (fuse_relu) {
      this->build_options.push_back("-DUSE_RELU");
    }
    return 0;
}

template <class Dtype>
int InnerProductLayerCL<Dtype>::SetKernelParameters() {
    int error_num;
    bool set_kernel_arg_success = true;
    int param_idx = 0;

    uint32_t b_channel_padding = this->_bottom_blobs[this->_bottom[0]]->get_channels_padding();

    uint32_t w_num = this->_weight_blobs[0]->num();
    uint32_t w_channel = this->_weight_blobs[0]->channels();

    //LOGI("%s ---- this->input_height: %d, this->input_width: %d, b_channel_padding: %d, w_num: %d, w_channel: %d",_top[0].c_str(), this->input_height, this->input_width, b_channel_padding, w_num, w_channel);

    uint32_t real_weight_size = this->input_height * this->input_width * this->_weight_blobs[0]->get_num_padding() * b_channel_padding;

    this->_weight_blobs[0]->AllocDevice(this->rt_param->context(), real_weight_size);

    std::vector<Dtype> weight_padding(real_weight_size, 0.0f);

    uint32_t hw_size = this->input_height * this->input_width;

    size_t N = this->channel_group_size;
    auto channel_groups = b_channel_padding / N;
    for (int i = 0; i < w_num; ++i) {
      for (int j = 0; j < w_channel; ++j) {
        int c_idx = j / hw_size;
        int hw_idx = j % hw_size;
        int src_idx = i * w_channel + j;
        int dst_idx = (((i / N * channel_groups + c_idx / N) * hw_size + hw_idx) * N + c_idx % N) * N + i % N;
        weight_padding[dst_idx] = kernel_data[src_idx];
      }
    }

    this->_weight_blobs[0]->WriteToDevice(this->rt_param->command_queue(), weight_padding.data(), real_weight_size);
    this->_weight_blobs[0]->Free();

    if (bias_term) {
        uint32_t real_bias_size = this->_weight_blobs[0]->get_num_padding();
        this->_weight_blobs[1]->AllocDevice(this->rt_param->context(), real_bias_size);
        std::vector<Dtype> bias_padding(real_bias_size, 0);
        memcpy(bias_padding.data(), bias_data, w_num * sizeof(Dtype));
        this->_weight_blobs[1]->WriteToDevice(this->rt_param->command_queue(), bias_padding.data(), real_bias_size);
        this->_weight_blobs[1]->Free();
    }

    /* build kernel */
    //kernels[0] = clCreateKernel(this->cl_programs[0], this->cl_kernel_functions[0].c_str(), &error_num);
    this->kernels[0] = cl::Kernel(this->cl_programs[0], this->cl_kernel_functions[0].c_str(), &error_num);
    if (!checkSuccess(error_num)) {
      LOGE("Failed to create innerProduct OpenCL kernels[0]. ");
      return -1;
    }

    cl::Buffer* input_mem = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    cl::Buffer* weight_mem = this->_weight_blobs[0]->data_cl();
    cl::Buffer* output_mem = this->_top_blobs[this->_top[0]]->data_cl();
    uint32_t out_real_channels = this->_top_blobs[this->_top[0]]->get_channels_padding();
    uint32_t use_relu = fuse_relu;

    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, *input_mem));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, *weight_mem));
    if (bias_term) {
      set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, *this->_weight_blobs[1]->data_cl()));
    }
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, *output_mem));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, b_channel_padding));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, out_real_channels));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->input_height));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->input_width));

    this->FineTuneGroupSize(this->kernels[0], this->_top_blobs[this->_top[0]]->height(), this->_top_blobs[this->_top[0]]->width());
    if (!set_kernel_arg_success) {
      LOGE("Failed setting inner product OpenCL kernels[0] arguments. ");
      return 1;
    }
    return 0;
  }

template <class Dtype>
int InnerProductLayerCL<Dtype>::ForwardCL() {
#ifdef TIMING_CL
    this->rt_param->command_queue().finish();
    timespec tpstart, tpend;
    clock_gettime(CLOCK_MONOTONIC, &tpstart);

    // int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
    //     kernels[0], cl::NullRange, cl::NDRange(global_work_size[0], global_work_size[1], global_work_size[2]),
    //     cl::NDRange(local_work_size[0], local_work_size[1], local_work_size[2]), nullptr, &events[0]);
    int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
        this->kernels[0], cl::NullRange, cl::NDRange(this->global_work_size[0], this->global_work_size[1], this->global_work_size[2]),
        cl::NDRange(this->local_work_size[0], this->local_work_size[1], this->local_work_size[2]), nullptr, &this->events[0]);

    if (!checkSuccess(error_num)) {
      LOGE("Failed enqueuing the inner product kernel.");
      return -1;
    }
    this->events[0].wait();
    clock_gettime(CLOCK_MONOTONIC, &tpend);
    double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
    LOGI("[%s] Execution time in %lf ms with %s\n", this->name().c_str(), timedif / 1000.0, this->cl_kernel_names[0].c_str());
    
    cl::Event profileEvent = this->events[0];
    double queued_nanos_ = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
    double submit_nanos_ = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
    double start_nanos_  = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    double stop_nanos_   = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double submit_kerel_time = (submit_nanos_ - queued_nanos_) / 1000.0 / 1000.0;
    double start_kerel_time = (start_nanos_ - submit_nanos_) / 1000.0 / 1000.0;
    double stop_kerel_time = (stop_nanos_ - start_nanos_) / 1000.0 / 1000.0;
    LOGI("[%s] [%s] Execution time in kernel: %0.5f, %0.5f, %0.5f\n",
     this->name().c_str(), this->cl_kernel_names[0].c_str(), submit_kerel_time, start_kerel_time, stop_kerel_time);

#else
    int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
        this->kernels[0], cl::NullRange, cl::NDRange(this->global_work_size[0], this->global_work_size[1], this->global_work_size[2]),
        cl::NDRange(this->local_work_size[0], this->local_work_size[1], this->local_work_size[2]), nullptr, nullptr);

    if (!checkSuccess(error_num)) {
      LOGE("Failed enqueuing the inner product kernel.");
      return -1;
    }

#endif

    return 0;
  }

template <class Dtype>
int InnerProductLayerCL<Dtype>::ForwardReshapeCL() {
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
void InnerProductLayerCL<Dtype>::FinetuneKernel() {
    uint32_t padded_output_c = this->_top_blobs[this->_top[0]]->get_channels_padding();
    this->channel_group_size = this->_top_blobs[this->_top[0]]->channel_grp();
    this->global_work_size[2] = padded_output_c / this->channel_group_size;

    std::string cur_kname = this->cl_kernel_names[0];
    std::string cur_kstr = this->cl_kernel_symbols[0];
    this->cl_kernel_names.clear();
    this->cl_kernel_symbols.clear();
    this->cl_kernel_names.push_back(cur_kname);
    this->cl_kernel_symbols.push_back(cur_kstr);
}

template <class Dtype>
int InnerProductLayerCL<Dtype>::GenerateTopBlobs() {
    const Blob<Dtype> *bottom_blob = this->_bottom_blobs[this->_bottom[0]];
    this->input_width = bottom_blob->width();
    this->input_height = bottom_blob->height();
    this->_top_blobs[this->_top[0]] = new Blob<Dtype>(1, this->output_channels, 1, 1);
    this->_top_blobs[this->_top[0]]->AllocDevice(this->rt_param->context(), this->_top_blobs[this->_top[0]]->data_size_padded_c());


    FinetuneKernel();
    SetWorkSize();


    return 0;
}

template <class Dtype>
int InnerProductLayerCL<Dtype>::SetWorkSize()
{
    this->global_work_size[0] = 1;
    this->global_work_size[1] = 1;
    //this->globalWorkSize[2] = output_channels / 4;
    this->local_work_size[0] = 1;
    this->local_work_size[1] = 1;
    this->local_work_size[2] = 1;
    return 0;
}

template <class Dtype>
int InnerProductLayerCL<Dtype>::Fuse(Layer<Dtype> *next_layer) {
    if (next_layer->type().compare("ReLU") == 0) {
      fuse_relu = true;
      return 1;
    } else {
      return 0;
    }
  }

template class InnerProductLayerCL<float>;
template class InnerProductLayerCL<uint16_t>;

}; // namespace feather
