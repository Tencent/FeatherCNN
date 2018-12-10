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

#include "elewise_layer_cl.h"

using namespace std;

namespace feather {

template <class Dtype>
int EltwiseLayerCL<Dtype>::InitCL() {
  string func_name = "eltwise";
  string kernel_name_eltwise = "eltwise_buffer";
  auto it_source = booster::opencl_kernel_string_map.find("eltwise_buffer");
  std::string kernel_str_eltwise(it_source->second.begin(), it_source->second.end());

  this->cl_kernel_functions.push_back(func_name);
  this->cl_kernel_names.push_back(kernel_name_eltwise);
  this->cl_kernel_symbols.push_back(kernel_str_eltwise);

  cl::Kernel kernel;
  this->kernels.push_back(kernel);
  cl::Event event;
  this->events.push_back(event);

  return 0;
}

template <class Dtype>
int EltwiseLayerCL<Dtype>::SetBuildOptions() {
  std::ostringstream ss;
  ss << this->channel_grp_size;
  this->build_options.push_back("-DN=" + ss.str());
  if(std::is_same<Dtype, uint16_t>::value)
      this->build_options.push_back("-DDATA_TYPE=half");
  else
      this->build_options.push_back("-DDATA_TYPE=float");
  return 0;
}

template <class Dtype>
int EltwiseLayerCL<Dtype>::GenerateTopBlobs() {
  assert(this->_bottom.size() == 2);
  assert(this->_bottom_blobs.size() == 2);
  assert(this->_bottom_blobs[this->_bottom[0]]->data_size() == this->_bottom_blobs[this->_bottom[1]]->data_size());
  Blob<Dtype>* p_blob = new Blob<Dtype>();
  p_blob->CopyShape(this->_bottom_blobs[this->_bottom[0]]);
  p_blob->AllocDevice(this->rt_param->context(), p_blob->data_size_padded_c());
  this->output_height = p_blob->height();
  this->output_width = p_blob->width();
  int output_channel = p_blob->get_channels_padding();
  this->_top_blobs[this->_top[0]] = p_blob;

  FinetuneKernel();
  SetWorkSize();

  return 0;
}

template <class Dtype>
int EltwiseLayerCL<Dtype>::SetWorkSize() {
    if (this->output_width > 32) this->group_size_w = 16;
    if (this->output_height > 32) this->group_size_h = 16;
    this->global_work_size[0] = (this->output_height / this->group_size_h + !!(this->output_height % this->group_size_h)) * this->group_size_h;
    this->global_work_size[1] = (this->output_width / this->group_size_w  + !!(this->output_width % this->group_size_w)) * this->group_size_w;
    this->local_work_size[0] = this->group_size_h;
    this->local_work_size[1] = this->group_size_w;

    if (this->global_work_size[2] > 4 && this->global_work_size[2] % 4 == 0) {
      this->local_work_size[2] = 4;
    } else {
      this->local_work_size[2] = 1;
    }
    return 0;
}

template <class Dtype>
int EltwiseLayerCL<Dtype>::SetKernelParameters() {
  int error_num;

  this->kernels[0] = cl::Kernel(this->cl_programs[0], this->cl_kernel_functions[0].c_str(), &error_num);
  if (!checkSuccess(error_num)) {
    LOGE("Failed to create Elementwise OpenCL kernels[0]. ");
    return -1;
  }

  cl::Buffer* input_mem1 = this->_bottom_blobs[this->_bottom[0]]->data_cl();
  cl::Buffer* input_mem2 = this->_bottom_blobs[this->_bottom[1]]->data_cl();
  cl::Buffer* output_mem = this->_top_blobs[this->_top[0]]->data_cl();

  uint32_t output_channels = this->_top_blobs[this->_top[0]]->get_channels_padding();

  bool set_kernel_arguments_success = true;
  int param_idx = 0;
  set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, *input_mem1));
  set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, *input_mem2));
  set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, *output_mem));
  set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, output_height));
  set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, output_width));
  set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, output_channels));

  this->FineTuneGroupSize(this->kernels[0], this->_top_blobs[this->_top[0]]->height(), this->_top_blobs[this->_top[0]]->width());
  if (!set_kernel_arguments_success) {
    LOGE("Failed setting inner product OpenCL kernels[0] arguments. ");
    return -1;
  }

  return 0;
}

template <class Dtype>
void EltwiseLayerCL<Dtype>::FinetuneKernel() {
  string cur_kname;
  string cur_kstr;
  size_t padded_input_c = this->_bottom_blobs[this->_bottom[0]]->get_channels_padding();
  size_t padded_output_c = this->_top_blobs[this->_top[0]]->get_channels_padding();

  int group_size = 4;
  if (padded_input_c % 16 == 0 && padded_output_c % 16 == 0) {
    group_size = 16;
  } else if (padded_input_c % 8 == 0 && padded_output_c % 8 == 0) {
    group_size = 8;
  }

  this->global_work_size[2] = padded_output_c / group_size;
  cur_kname = this->cl_kernel_names[0];
  cur_kstr = this->cl_kernel_symbols[0];
  this->channel_grp_size = group_size;

  this->cl_kernel_names.clear();
  this->cl_kernel_symbols.clear();
  this->cl_kernel_names.push_back(cur_kname);
  this->cl_kernel_symbols.push_back(cur_kstr);

}

template <class Dtype>
int EltwiseLayerCL<Dtype>::ForwardReshapeCL() {
    if (this->output_height == this->_bottom_blobs[this->_bottom[0]]->height() &&
        this->output_width == this->_bottom_blobs[this->_bottom[0]]->width())
        return this->ForwardCL();

    bool set_kernel_arg_success = true;

    this->output_height = this->_bottom_blobs[this->_bottom[0]]->height();
    this->output_width = this->_bottom_blobs[this->_bottom[0]]->width();
    if (this->_top_blobs[this->_top[0]]->ReshapeWithReallocDevice(this->rt_param->context(),
                                      this->_top_blobs[this->_top[0]]->num(),
                                      this->_top_blobs[this->_top[0]]->channels(),
                                      this->output_height, this->output_width) == 2)
    {
        cl::Buffer* output_mem = this->_top_blobs[this->_top[0]]->data_cl();
        set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(2, *output_mem));
    }

    cl::Buffer* input_mem1 = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    cl::Buffer* input_mem2 = this->_bottom_blobs[this->_bottom[1]]->data_cl();
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(0, *input_mem1));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(1, *input_mem2));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(3, this->output_height));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(4, this->output_width));

    if (!set_kernel_arg_success) {
      LOGE("Failed setting conv OpenCL kernels[0] arguments.");
      return 1;
    }

    this->SetWorkSize();
    this->FineTuneGroupSize(this->kernels[0], this->_top_blobs[this->_top[0]]->height(), this->_top_blobs[this->_top[0]]->width());
    return this->ForwardCL();
}

template <class Dtype>
int EltwiseLayerCL<Dtype>::ForwardCL() {
#ifdef TIMING_CL
  clFinish(this->rt_param->command_queue());
  timespec tpstart, tpend;
  clock_gettime(CLOCK_MONOTONIC, &tpstart);

  int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
        kernels[0], cl::NullRange, cl::NDRange(global_work_size[0], global_work_size[1], global_work_size[2]),
        cl::NDRange(local_work_size[0], local_work_size[1], local_work_size[2]), nullptr, &events[0]);
  if (!checkSuccess(error_num)) {
    LOGE("Failed enqueuing the element wise kernel.");
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
          this->kernels[0], cl::NullRange, cl::NDRange(this->global_work_size[0], this->global_work_size[1], this->global_work_size[2]),
          cl::NDRange(this->local_work_size[0], this->local_work_size[1], this->local_work_size[2]), nullptr, nullptr);
    if (!checkSuccess(error_num)) {
      LOGE("Failed enqueuing the element wise kernel.");
      return -1;
    }

#endif

  return 0;
}

template class EltwiseLayerCL<float>;
template class EltwiseLayerCL<uint16_t>;

}; // namespace feather
