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
    std::string kernel_name_4o4 = "clInnerProduct4o4";
    auto it_source1 = booster::opencl_kernel_string_map.find("innerProductBufferReformW4o4_grp1");
    std::string kernel_str_4o4(it_source1->second.begin(),it_source1->second.end());

    std::string kernel_name_8o8 = "clInnerProduct8o8";
    auto it_source2 = booster::opencl_kernel_string_map.find("innerProductBufferReformW8o8_grp1");
    std::string kernel_str_8o8(it_source2->second.begin(),it_source2->second.end());

    std::string kernel_name_16o16 = "clInnerProduct16o16";
    auto it_source3 = booster::opencl_kernel_string_map.find("innerProductBufferReformW16o16_grp1");
    std::string kernel_str_16o16(it_source3->second.begin(),it_source3->second.end());


    this->cl_kernel_names.push_back(kernel_name_4o4);
    this->cl_kernel_names.push_back(kernel_name_8o8);
    this->cl_kernel_names.push_back(kernel_name_16o16);

    this->cl_kernel_symbols.push_back(kernel_str_4o4);
    this->cl_kernel_symbols.push_back(kernel_str_8o8);
    this->cl_kernel_symbols.push_back(kernel_str_16o16);

    cl_kernel kernel;
    this->kernels.push_back(kernel);
    cl_event event;
    this->events.push_back(event);

    return 0;
  }

int InnerProductLayerCL::SetKernelParameters() {
    int error_num;
    bool set_kernel_arguments_success = true;
    int param_idx = 0;

    size_t b_channel_padding = this->_bottom_blobs[this->_bottom[0]]->get_channels_padding();

    size_t w_num = this->_weight_blobs[0]->num();
    size_t w_channel = this->_weight_blobs[0]->channels();

    //LOGI("%s ---- this->input_height: %d, this->input_width: %d, b_channel_padding: %d, w_num: %d, w_channel: %d",_top[0].c_str(), this->input_height, this->input_width, b_channel_padding, w_num, w_channel);

    size_t real_weight_size = this->input_height * this->input_width * this->_weight_blobs[0]->get_num_padding() * b_channel_padding;

    this->_weight_blobs[0]->AllocDevice(this->rt_param->context(), real_weight_size);

    std::vector<uint16_t> weight_padding(real_weight_size, 0.0f);

    size_t kernel_size = this->input_height * this->input_width * b_channel_padding;
    size_t hw_size = this->input_height * this->input_width;
    size_t num_channel_grp = b_channel_padding / this->channel_grp_size;
    size_t c_grp_size = 1 /* this->channel_grp_size */;
    size_t n_grp_size = this->channel_grp_size;

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
        size_t real_bias_size = this->_weight_blobs[0]->get_num_padding();
        this->_weight_blobs[1]->AllocDevice(this->rt_param->context(), real_bias_size);
        // float bias_padding[real_bias_size];
        // memset(bias_padding, 0.0f, real_bias_size * sizeof(float));
        std::vector<uint16_t> bias_padding(real_bias_size, 0);
        memcpy(bias_padding.data(), bias_data, w_num * sizeof(uint16_t));
        this->_weight_blobs[1]->WriteToDevice(this->rt_param->command_queue(), bias_padding.data(), real_bias_size);
        this->_weight_blobs[1]->Free();
    }

    /* build kernel */
    kernels[0] = clCreateKernel(this->cl_programs[0], this->cl_kernel_functions[0].c_str(), &error_num);
    if (!checkSuccess(error_num)) {
      LOGE("Failed to create innerProduct OpenCL kernels[0]. ");
      return -1;
    }

    cl_mem input_mem = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    cl_mem weight_mem = this->_weight_blobs[0]->data_cl();
    cl_mem output_mem = this->_top_blobs[this->_top[0]]->data_cl();
    int out_real_channels = this->_top_blobs[this->_top[0]]->get_channels_padding();
    int use_relu = fuse_relu;

    cl_mem bias_mem;
    if (bias_term) {
      bias_mem = _weight_blobs[1]->data_cl();
    } else {
      std::vector<uint16_t> bias_vec(out_real_channels, 0);
      bias_mem = clCreateBuffer(this->rt_param->context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  out_real_channels * sizeof(uint16_t), bias_vec.data(), &error_num);
      if (!checkSuccess(error_num)) {
        LOGE("Failed to create OpenCL buffers[%d]", error_num);
        return -1;
      }
    }

    //int channel_grp = in_real_channels / 4;
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_mem), &input_mem));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_mem), &weight_mem));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_mem), &bias_mem));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_mem), &output_mem));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &b_channel_padding));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &out_real_channels));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->input_height));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->input_width));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &use_relu));

    FineTuneGroupSize(this->kernels[0], this->_top_blobs[this->_top[0]]->height(), this->_top_blobs[this->_top[0]]->width());
    if (!set_kernel_arguments_success) {
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

    int error_num = clEnqueueNDRangeKernel(this->rt_param->command_queue(), kernels[0], 3,
                    NULL, this->global_work_size, this->local_work_size, 0, NULL,&events[0]);
    if (!checkSuccess(error_num)) {
      LOGE("Failed enqueuing the inner product kernel. %s", errorNumberToString(error_num).c_str());
      return -1;
    }

    clWaitForEvents(1, &events[0]);
    clock_gettime(CLOCK_MONOTONIC, &tpend);
    double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
    LOGI("[%s] Execution time in %lf ms with %s\n", this->name().c_str(), timedif / 1000.0, kernel_names[0].c_str());

    cl_ulong time_start, time_end;
    double total_time;
    clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    total_time = time_end - time_start;
    LOGI("[%s] Execution time in kernel: %0.5f ms with %s\n", this->name().c_str(), total_time / 1000000.0, kernel_names[0].c_str());

    error_num = clReleaseEvent(events[0]);
    if (!checkSuccess(error_num)) {
      LOGE("Failed release event. %s", errorNumberToString(error_num).c_str());
      return -1;
    }

#else
    int error_num = clEnqueueNDRangeKernel(this->rt_param->command_queue(), kernels[0], 3,
                    NULL, this->global_work_size, this->local_work_size, 0, NULL, NULL);
    if (!checkSuccess(error_num)) {
      LOGE("Failed enqueuing the inner product kernel. %s", errorNumberToString(error_num).c_str());
      return -1;
    }
#endif

    return 0;
  }

void InnerProductLayerCL::FinetuneKernel() {
    std::string cur_kname;
    std::string cur_kstr;
    size_t padded_input_c = this->_bottom_blobs[this->_bottom[0]]->get_channels_padding();
    size_t padded_output_c = this->_top_blobs[this->_top[0]]->get_channels_padding();

    int kernel_idx = 0, group_size = 4;
    if (padded_input_c % 16 == 0 && padded_output_c % 16 == 0) {
      kernel_idx = 2;
      group_size = 16;
    } else if (padded_input_c % 8 == 0 && padded_output_c % 8 == 0) {
      kernel_idx = 1;
      group_size = 8;
    }

    cur_kname = this->cl_kernel_names[kernel_idx];
    cur_kstr = this->cl_kernel_symbols[kernel_idx];
    this->global_work_size[2] = padded_output_c / group_size;
    this->channel_grp_size = group_size;

    this->cl_kernel_names.clear();
    this->cl_kernel_symbols.clear();
    this->cl_kernel_names.push_back(cur_kname);
    this->cl_kernel_symbols.push_back(cur_kstr);
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
