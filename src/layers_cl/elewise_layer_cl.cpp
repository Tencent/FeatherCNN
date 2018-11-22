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


int EltwiseLayerCL::InitCL() {
  string func_name = "eltwise";
  
  string kernel_name_4o4 = "clEltwise4o4";
  auto it_source1 = booster::opencl_kernel_string_map.find("elewiseBuffer4o4");
  std::string kernel_str_4o4(it_source1->second.begin(),it_source1->second.end());

  string kernel_name_8o8 = "clEltwise8o8";
  auto it_source2 = booster::opencl_kernel_string_map.find("elewiseBuffer8o8");
  std::string kernel_str_8o8(it_source2->second.begin(),it_source2->second.end());

  string kernel_name_16o16 = "clEltwise16o16";
  auto it_source3 = booster::opencl_kernel_string_map.find("elewiseBuffer16o16");
  std::string kernel_str_16o16(it_source3->second.begin(),it_source3->second.end());


  this->cl_kernel_functions.push_back(func_name);
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

int EltwiseLayerCL::GenerateTopBlobs() {
  assert(_bottom.size() == 2);
  assert(_bottom_blobs.size() == 2);
  assert(_bottom_blobs[_bottom[0]]->data_size() == _bottom_blobs[_bottom[1]]->data_size());


  Blob<uint16_t>* p_blob = new Blob<uint16_t>();
  p_blob->CopyShape(_bottom_blobs[_bottom[0]]);

  p_blob->AllocDevice(rt_param->context(), p_blob->data_size_padded_c());

  int output_height = p_blob->height();
  int output_width = p_blob->width();
  int output_channel = p_blob->get_channels_padding();
  _top_blobs[_top[0]] = p_blob;

  int group_size_h = 8, group_size_w = 8;
  if (output_width > 32) group_size_w = 16;
  if (output_height > 32) group_size_h = 16;

  FinetuneKernel();

  this->global_work_size[2] = output_channel / 4;
  this->global_work_size[0] = (output_height / group_size_h + !!(output_height % group_size_h)) * group_size_h;
  this->global_work_size[1] = (output_width / group_size_w  + !!(output_width % group_size_w)) * group_size_w;
  this->local_work_size[0] = group_size_h;
  this->local_work_size[1] = group_size_w;

  if (this->global_work_size[2] > 4 && this->global_work_size[2] % 4 == 0) {
    this->local_work_size[2] = 4;
  } else {
    this->local_work_size[2] = 1;
  }

  return 0;
}

int EltwiseLayerCL::SetKernelParameters() {
  int error_num;

  kernels[0] = clCreateKernel(cl_programs[0], cl_kernel_functions[0].c_str(), &error_num);
  if (!checkSuccess(error_num)) {
    LOGE("Failed to create Elementwise OpenCL kernels[0]. ");
    return -1;
  }

  cl_mem input_mem1 = this->_bottom_blobs[this->_bottom[0]]->data_cl();
  cl_mem input_mem2 = this->_bottom_blobs[this->_bottom[1]]->data_cl();
  cl_mem output_mem = this->_top_blobs[this->_top[0]]->data_cl();

  int output_height = this->_top_blobs[this->_top[0]]->height();
  int output_width = this->_top_blobs[this->_top[0]]->width();
  int output_channels = this->_top_blobs[this->_top[0]]->get_channels_padding();

  bool set_kernel_arguments_success = true;
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], 0, sizeof(cl_mem), &input_mem1));
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], 1, sizeof(cl_mem), &input_mem2));
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], 2, sizeof(cl_mem), &output_mem));
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], 3, sizeof(cl_int), &output_height));
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], 4, sizeof(cl_int), &output_width));
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], 5, sizeof(cl_int), &output_channels));

  FineTuneGroupSize(this->kernels[0], _top_blobs[_top[0]]->height(), _top_blobs[_top[0]]->width());
  if (!set_kernel_arguments_success) {
    LOGE("Failed setting inner product OpenCL kernels[0] arguments. ");
    return -1;
  }

  return 0;
}

void EltwiseLayerCL::FinetuneKernel() {
  string cur_kname;
  string cur_kstr;
  // size_t padded_input_c = _bottom_blobs[_bottom[0]]->get_channels_padding();
  size_t padded_output_c = _top_blobs[_top[0]]->get_channels_padding();

  if (padded_output_c % 4 == 0 && padded_output_c % 8 != 0) {
    cur_kname = cl_kernel_names[0];
    cur_kstr = cl_kernel_symbols[0];
    this->global_work_size[2] = padded_output_c / 4;
  } else {
    if (padded_output_c % 8 == 0 && padded_output_c % 16 != 0) {
      cur_kname = cl_kernel_names[1];
      cur_kstr = cl_kernel_symbols[1];
      this->global_work_size[2] = padded_output_c / 8;
    } else {
      cur_kname = cl_kernel_names[2];
      cur_kstr = cl_kernel_symbols[2];
      this->global_work_size[2] = padded_output_c / 16;
    }
  }

  cl_kernel_names.clear();
  cl_kernel_symbols.clear();
  cl_kernel_names.push_back(cur_kname);
  cl_kernel_symbols.push_back(cur_kstr);
}


int EltwiseLayerCL::ForwardCL() {
  int error_num = clEnqueueNDRangeKernel(rt_param->command_queue(), kernels[0], 3, NULL, global_work_size, local_work_size, 0, NULL,&events[0]);
  if (!checkSuccess(error_num)) {
    LOGE("Failed enqueuing the element wise kernel. %s", errorNumberToString(error_num).c_str());
    return -1;
  }
#ifdef TIMING_CL
  clWaitForEvents(1, &events[0]);
  clock_gettime(CLOCK_MONOTONIC, &tpend);
  double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
  LOGI("[%s] Execution time in %lf ms with %s\n", this->name().c_str(), timedif / 1000.0, cl_kernel_names[0].c_str());

  cl_ulong time_start, time_end;
  double total_time;
  clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
  clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
  total_time = time_end - time_start;
  LOGI("[%s] Execution time in kernel: %0.5f ms with %s\n", this->name().c_str(), total_time / 1000000.0, cl_kernel_names[0].c_str());
#endif
  /* if we wanna do something for event in future */

  error_num = clReleaseEvent(events[0]);
  if (!checkSuccess(error_num)) {
    LOGE("Failed release event. %s", errorNumberToString(error_num).c_str());
    return -1;
  }
  return 0;
}

}; // namespace feather



