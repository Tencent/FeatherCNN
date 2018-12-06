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

InnerProductLayerCL::InnerProductLayerCL(const LayerParameter *layer_param, RuntimeParameter<float>* rt_param)
      : fuse_relu(false), Layer<uint16_t>(layer_param, rt_param) {
    const InnerProductParameter *inner_product_param = layer_param->inner_product_param();
    bias_term = inner_product_param->bias_term();
    this->output_channels = this->_weight_blobs[0]->num();
    assert(this->_weight_blobs.size() > 0);
    this->kernel_data = this->_weight_blobs[0]->data();
    if (this->bias_term) {
      assert(this->_weight_blobs.size() == 2);
      this->bias_data = this->_weight_blobs[1]->data();
    }
    _fusible = true;

    InitCL();
  }

int InnerProductLayerCL::InitCL() {
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

void InnerProductLayerCL::SetBuildOptions() {
    std::ostringstream ss;
    ss << channel_grp_size;
    this->build_options.push_back("-DN=" + ss.str());
    this->build_options.push_back("-DDATA_TYPE=half");
    if (bias_term) {
      this->build_options.push_back("-DBIAS");
    }
    if (fuse_relu) {
      this->build_options.push_back("-DUSE_RELU");
    }
}

int InnerProductLayerCL::SetKernelParameters() {
    int error_num;
    bool set_kernel_arg_success = true;
    int param_idx = 0;

    uint32_t b_channel_padding = this->_bottom_blobs[this->_bottom[0]]->get_channels_padding();

    uint32_t w_num = this->_weight_blobs[0]->num();
    uint32_t w_channel = this->_weight_blobs[0]->channels();

    //LOGI("%s ---- this->input_height: %d, this->input_width: %d, b_channel_padding: %d, w_num: %d, w_channel: %d",_top[0].c_str(), this->input_height, this->input_width, b_channel_padding, w_num, w_channel);

    uint32_t real_weight_size = this->input_height * this->input_width * this->_weight_blobs[0]->get_num_padding() * b_channel_padding;

    this->_weight_blobs[0]->AllocDevice(this->rt_param->context(), real_weight_size);

    std::vector<uint16_t> weight_padding(real_weight_size, 0.0f);

    uint32_t kernel_size = this->input_height * this->input_width * b_channel_padding;
    uint32_t hw_size = this->input_height * this->input_width;
    uint32_t num_channel_grp = b_channel_padding / this->channel_grp_size;
    uint32_t c_grp_size = 1 /* this->channel_grp_size */;
    uint32_t n_grp_size = this->channel_grp_size;

    for (int n = 0; n < w_num; ++n) {
      for (int m = 0; m < w_channel; ++m) {
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

    if (bias_term) {
        uint32_t real_bias_size = this->_weight_blobs[0]->get_num_padding();
        this->_weight_blobs[1]->AllocDevice(this->rt_param->context(), real_bias_size);
        // float bias_padding[real_bias_size];
        // memset(bias_padding, 0.0f, real_bias_size * sizeof(float));
        std::vector<uint16_t> bias_padding(real_bias_size, 0);
        memcpy(bias_padding.data(), bias_data, w_num * sizeof(uint16_t));
        this->_weight_blobs[1]->WriteToDevice(this->rt_param->command_queue(), bias_padding.data(), real_bias_size);
        this->_weight_blobs[1]->Free();
    }

    /* build kernel */
    //kernels[0] = clCreateKernel(this->cl_programs[0], this->cl_kernel_functions[0].c_str(), &error_num);
    kernels[0] = cl::Kernel(this->cl_programs[0], this->cl_kernel_functions[0].c_str(), &error_num);
    if (!checkSuccess(error_num)) {
      LOGE("Failed to create innerProduct OpenCL kernels[0]. ");
      return -1;
    }

    cl::Buffer* input_mem = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    cl::Buffer* weight_mem = this->_weight_blobs[0]->data_cl();
    cl::Buffer* output_mem = this->_top_blobs[this->_top[0]]->data_cl();
    uint32_t out_real_channels = this->_top_blobs[this->_top[0]]->get_channels_padding();
    uint32_t use_relu = fuse_relu;

    set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, *input_mem));
    set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, *weight_mem));
    if (bias_term) {
      set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, *_weight_blobs[1]->data_cl()));
    }
    set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, *output_mem));
    set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, b_channel_padding));
    set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, out_real_channels));
    set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, this->input_height));
    set_kernel_arg_success &= checkSuccess(kernels[0].setArg(param_idx++, this->input_width));

    FineTuneGroupSize(this->kernels[0], this->_top_blobs[this->_top[0]]->height(), this->_top_blobs[this->_top[0]]->width());
    if (!set_kernel_arg_success) {
      LOGE("Failed setting inner product OpenCL kernels[0] arguments. ");
      return 1;
    }
    return 0;
  }

int InnerProductLayerCL::ForwardCL() {
#ifdef TIMING_CL
    clFinish(commandQueue);
    timespec tpstart, tpend;
    clock_gettime(CLOCK_MONOTONIC, &tpstart);

    int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
        kernels[0], cl::NullRange, cl::NDRange(global_work_size[0], global_work_size[1], global_work_size[2]),
        cl::NDRange(local_work_size[0], local_work_size[1], local_work_size[2]), nullptr, &events[0]);

    if (!checkSuccess(error_num)) {
      LOGE("Failed enqueuing the inner product kernel.");
      return -1;
    }

    events[0].wait();
    clock_gettime(CLOCK_MONOTONIC, &tpend);
    double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
    LOGI("[%s] Execution time in %lf ms with %s\n", this->name().c_str(), timedif / 1000.0, kernel_names[0].c_str());
    double start_nanos_ = events[0].getProfilingInfo<CL_PROFILING_COMMAND_START>();
    double stop_nanos_  = events[0].getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double kerel_time = (stop_nanos_ - start_nanos_) / 1000.0 / 1000.0;
    LOGI("[%s] Execution time in kernel: %0.5f ms with %s\n", this->name().c_str(), kerel_time, kernel_names[0].c_str());

#else
    int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
        kernels[0], cl::NullRange, cl::NDRange(global_work_size[0], global_work_size[1], global_work_size[2]),
        cl::NDRange(local_work_size[0], local_work_size[1], local_work_size[2]), nullptr, nullptr);

    if (!checkSuccess(error_num)) {
      LOGE("Failed enqueuing the inner product kernel.");
      return -1;
    }

#endif

    return 0;
  }

int InnerProductLayerCL::ForwardReshapeCL() {
    if (this->input_height == this->_bottom_blobs[this->_bottom[0]]->height() &&
        this->input_width == this->_bottom_blobs[this->_bottom[0]]->width())
        return this->ForwardCL();
    else
    {
        LOGE("inner product does not support variable input size");
        return -1;
    }
}

void InnerProductLayerCL::FinetuneKernel() {
    std::string cur_kname;
    std::string cur_kstr;
    uint32_t padded_input_c = this->_bottom_blobs[this->_bottom[0]]->get_channels_padding();
    uint32_t padded_output_c = this->_top_blobs[this->_top[0]]->get_channels_padding();

    int group_size = 4;
    if (padded_input_c % 16 == 0 && padded_output_c % 16 == 0) {
      group_size = 16;
    } else if (padded_input_c % 8 == 0 && padded_output_c % 8 == 0) {
      group_size = 8;
    }

    cur_kname = this->cl_kernel_names[0];
    cur_kstr = this->cl_kernel_symbols[0];
    this->global_work_size[2] = padded_output_c / group_size;
    this->channel_grp_size = group_size;

    this->cl_kernel_names.clear();
    this->cl_kernel_symbols.clear();
    this->cl_kernel_names.push_back(cur_kname);
    this->cl_kernel_symbols.push_back(cur_kstr);

    std::ostringstream ss;
    ss << group_size;
    this->build_options.push_back("-DCHANNEL_GROUP_SIZE=" + ss.str());
    this->build_options.push_back("-DDATA_TYPE=half");
}

int InnerProductLayerCL::GenerateTopBlobs() {
    const Blob<uint16_t> *bottom_blob = this->_bottom_blobs[this->_bottom[0]];
    this->input_width = bottom_blob->width();
    this->input_height = bottom_blob->height();
    this->_top_blobs[this->_top[0]] = new Blob<uint16_t>(1, this->output_channels, 1, 1);
    this->_top_blobs[this->_top[0]]->AllocDevice(this->rt_param->context(), this->_top_blobs[this->_top[0]]->data_size_padded_c());


    FinetuneKernel();
    SetWorkSize();


    return 0;
}

int InnerProductLayerCL::SetWorkSize()
{
    this->global_work_size[0] = 1;
    this->global_work_size[1] = 1;
    //this->globalWorkSize[2] = output_channels / 4;
    this->local_work_size[0] = 1;
    this->local_work_size[1] = 1;
    this->local_work_size[2] = 1;
    return 0;
}

int InnerProductLayerCL::Fuse(Layer *next_layer) {
    if (next_layer->type().compare("ReLU") == 0) {
      fuse_relu = true;
      return 1;
    } else {
      return 0;
    }
  }

}; // namespace feather
