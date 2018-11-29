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

#include "pooling_layer_cl.h"

namespace feather {

PoolingLayerCL::PoolingLayerCL(const LayerParameter *layer_param, RuntimeParameter<float>* rt_param)
      : stride_height(1),
        stride_width(1),
        Layer<uint16_t>(layer_param, rt_param) {
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
    std::string ave_opt = "-DAVE_POOLING";
    switch (this->method) {
      case PoolingParameter_::PoolMethod_MAX_:
        break;
      case PoolingParameter_::PoolMethod_AVE:
        this->build_options.push_back(ave_opt);
        break;
      default:
        LOGE("Unsupported pool method\n");
        break;
    }
    // printf("kernel (%ld %ld) pad (%ld %ld) stride (%ld %ld) global_pooling %d\n",
    //     kernel_height, kernel_width, pad_height, pad_width, stride_height, stride_width, global_pooling);

    InitCL();
  }

int PoolingLayerCL::InitCL() {
    std::string func_name = "pooling";
    this->cl_kernel_functions.push_back(func_name);
    std::string kernel_name_4o4 = "clPooling4o4";
    auto it_source1 = booster::opencl_kernel_string_map.find("poolingBufferFix4o4");
    std::string kernel_str_4o4(it_source1->second.begin(),it_source1->second.end());

    std::string kernel_name_8o8 = "clPooling8o8";
    auto it_source2 = booster::opencl_kernel_string_map.find("poolingBufferFix8o8");
    std::string kernel_str_8o8(it_source2->second.begin(),it_source2->second.end());

    std::string kernel_name_16o16 = "clPooling16o16";
    auto it_source3 = booster::opencl_kernel_string_map.find("poolingBufferFix16o16");
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


int PoolingLayerCL::SetKernelParameters() {
    int error_num;
    bool set_kernel_arguments_success = true;
    int param_idx = 0;
    kernels[0] = clCreateKernel(this->cl_programs[0], this->cl_kernel_functions[0].c_str(), &error_num);
    if (!checkSuccess(error_num)) {
      LOGE("Failed to create pooling OpenCL kernels[0]. ");
      return 1;
    }

    cl_mem input_mem = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    cl_mem output_mem = this->_top_blobs[this->_top[0]]->data_cl();
    int real_channels = this->_top_blobs[this->_top[0]]->get_channels_padding();

    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_mem), &input_mem));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_mem), &output_mem));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &real_channels));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->input_height));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->input_width));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->output_height));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->output_width));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->kernel_height));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->kernel_width));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->stride_height));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->stride_width));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->pad_height));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->pad_width));

    if (!set_kernel_arguments_success) {
      LOGE("Failed setting pooling OpenCL kernels[0] arguments.");
      return 1;
    }

    FineTuneGroupSize(this->kernels[0], this->_top_blobs[this->_top[0]]->height(), this->_top_blobs[this->_top[0]]->width());
    return 0;
  }

int PoolingLayerCL::ForwardCL() {
#ifdef TIMING_CL
    clFinish(this->rt_param->command_queue());
    timespec tpstart, tpend;
    clock_gettime(CLOCK_MONOTONIC, &tpstart);

    // if(group <=0)	group = 1;
    // LOGI("Forward layer (GPU_CL) %s", this->name().c_str());
    // LOGI("kernel (GPU_CL) %dx%d", kernel_height, kernel_width);
    // LOGI("stride (GPU_CL) %d %d", stride_height, stride_width);
    // LOGI("input (GPU_CL) %dx%d", input_height, input_width);
    // LOGI("output (GPU_CL) %dx%d", output_height, output_width);
    // LOGI("padding (GPU_CL) %d %d", padding_left, padding_top);
    // LOGI("globalWorkSize (GPU_CL): %d, %d, %d", global_work_size[0], global_work_size[1], global_work_size[2]);
    int error_num = clEnqueueNDRangeKernel(this->rt_param->command_queue(), kernels[0], 3,
                    NULL, this->global_work_size, this->local_work_size, 0, NULL,&events[0]);
    if (!checkSuccess(error_num)) {
      LOGE("Failed enqueuing the conv kernel. %d", error_num);
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
        LOGE("Failed release event.");
        return -1;
    }

#else
    int error_num = clEnqueueNDRangeKernel(this->rt_param->command_queue(), kernels[0], 3,
                    NULL, this->global_work_size, this->local_work_size, 0, NULL, NULL);
    if (!checkSuccess(error_num)) {
      LOGE("Failed enqueuing the conv kernel. %d", error_num);
      return -1;
    }
#endif


    return 0;
  }

int PoolingLayerCL::ForwardReshapeCL() {
    if (this->input_height == this->_bottom_blobs[this->_bottom[0]]->height() &&
        this->input_width == this->_bottom_blobs[this->_bottom[0]]->width())
        return this->ForwardCL();

    bool set_kernel_arg_success = true;
    this->input_height = this->_bottom_blobs[this->_bottom[0]]->height();
    this->input_width = this->_bottom_blobs[this->_bottom[0]]->width();

    AssignOutputSize();
    if (this->_top_blobs[this->_top[0]]->ReshapeWithReallocDevice(this->rt_param->context(),
                                      this->_top_blobs[this->_top[0]]->num(),
                                      this->_top_blobs[this->_top[0]]->channels(),
                                      this->output_height, this->output_width) == 2)
    {
        cl_mem output_mem = _top_blobs[_top[0]]->data_cl();
        set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], 1, sizeof(cl_mem), &output_mem));
    }

    int param_idx = 3;
    cl_mem input_mem = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], 0, sizeof(cl_mem), &input_mem));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->input_height));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->input_width));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->output_height));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->output_width));

    SetWorkSize();
    FineTuneGroupSize(this->kernels[0], this->_top_blobs[this->_top[0]]->height(), this->_top_blobs[this->_top[0]]->width());
    return this->ForwardCL();
}


void PoolingLayerCL::FinetuneKernel() {
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

int PoolingLayerCL::GenerateTopBlobs() {
    //Only accept a single bottom blob.
    const Blob<uint16_t> *bottom_blob = this->_bottom_blobs[this->_bottom[0]];
    this->input_height = bottom_blob->height();
    this->input_width = bottom_blob->width();
    this->input_channels = bottom_blob->channels();

    AssignOutputSize();
    this->output_channels = this->input_channels;
    this->_top_blobs[this->_top[0]] = new Blob<uint16_t>(1, this->output_channels, this->output_height, this->output_width);
    this->_top_blobs[this->_top[0]]->AllocDevice(this->rt_param->context(), this->_top_blobs[this->_top[0]]->data_size_padded_c());
    FinetuneKernel();
    SetWorkSize();

    return 0;
}

inline void PoolingLayerCL::AssignOutputSize()
{
  if (this->global_pooling) {
    this->kernel_height = this->input_height;
    this->kernel_width = this->input_width;
    this->output_height = 1;
    this->output_width = 1;
  } else {
    //General pooling.
    this->output_height = static_cast<int>(ceil(static_cast<float>(this->input_height + 2 * this->pad_height - this->kernel_height) / this->stride_height)) + 1;
    this->output_width = static_cast<int>(ceil(static_cast<float>(this->input_width + 2 * this->pad_width - this->kernel_width) / this->stride_width)) + 1;
  }
}

int PoolingLayerCL::SetWorkSize()
{
    if (this->output_width > 32) this->group_size_w = 16;
    if (this->output_height > 32) this->group_size_h = 16;
    this->global_work_size[0] = (this->output_height / this->group_size_h + !!(this->output_height % this->group_size_h)) * this->group_size_h;
    this->global_work_size[1] = (this->output_width / this->group_size_w  + !!(this->output_width % this->group_size_w)) * this->group_size_w;
    this->local_work_size[0] = this->group_size_h;
    this->local_work_size[1] = this->group_size_w;
    if(this->global_work_size[2] > 4 && this->global_work_size[2] % 4 == 0) {
      this->local_work_size[2] = 4;
    } else {
      this->local_work_size[2] = 1;
    }
    return 0;
}


}; // namespace feather
