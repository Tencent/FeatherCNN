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

template <class Dtype>
PoolingLayerCL<Dtype>::PoolingLayerCL(const LayerParameter *layer_param, RuntimeParameter<float>* rt_param)
      : stride_height(1),
        stride_width(1),
        Layer<Dtype>(layer_param, rt_param) {
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
    // printf("kernel (%ld %ld) pad (%ld %ld) stride (%ld %ld) global_pooling %d\n",
    //     kernel_height, kernel_width, pad_height, pad_width, stride_height, stride_width, global_pooling);
    InitCL();
  }

template <class Dtype>
int PoolingLayerCL<Dtype>::InitCL() {
    std::string func_name = "pooling";
    this->cl_kernel_functions.push_back(func_name);
    std::string kernel_name_pooling = "pooling_buffer";
    auto it_source = booster::opencl_kernel_string_map.find("pooling_buffer");
    std::string kernel_str_pooling(it_source->second.begin(), it_source->second.end());

    this->cl_kernel_names.push_back(kernel_name_pooling);
    this->cl_kernel_symbols.push_back(kernel_str_pooling);


    cl::Kernel kernel;
    this->kernels.push_back(kernel);
    cl::Event event;
    this->events.push_back(event);
    return 0;
}

template <class Dtype>
int PoolingLayerCL<Dtype>::SetBuildOptions() {
  std::string ave_opt = "-DAVE_POOLING";
  switch (this->method) {
    case PoolingParameter_::PoolMethod_MAX_:
      if (std::is_same<Dtype, uint16_t>::value)
        this->build_options.push_back("-DMIN_VAL=-HALF_MAX");
      else
        this->build_options.push_back("-DMIN_VAL=-FLT_MAX");

      break;
    case PoolingParameter_::PoolMethod_AVE:
      this->build_options.push_back(ave_opt);

      break;
    default:
      LOGE("Unsupported pool method\n");

      break;
  }
  std::ostringstream ss;
  ss << this->channel_group_size;
  this->build_options.push_back("-DN=" + ss.str());
  if (std::is_same<Dtype, uint16_t>::value)
    this->build_options.push_back("-DDATA_TYPE=half");
  else
    this->build_options.push_back("-DDATA_TYPE=float");
  return 0;
}

template <class Dtype>
int PoolingLayerCL<Dtype>::SetKernelParameters() {
    int error_num;
    bool set_kernel_arguments_success = true;
    int param_idx = 0;
    this->kernels[0] = cl::Kernel(this->cl_programs[0], this->cl_kernel_functions[0].c_str(), &error_num);
    if (!checkSuccess(error_num)) {
      LOGE("Failed to create pooling OpenCL kernels[0]. ");
      return 1;
    }

    cl::Buffer* input_mem = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    cl::Buffer* output_mem = this->_top_blobs[this->_top[0]]->data_cl();
    uint32_t real_channels = this->_top_blobs[this->_top[0]]->get_channels_padding();

    set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, *input_mem));
    set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, *output_mem));
    set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, real_channels));
    set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->input_height));
    set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->input_width));
    set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->output_height));
    set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->output_width));
    set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->kernel_height));
    set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->kernel_width));
    set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->stride_height));
    set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->stride_width));
    set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->pad_height));
    set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->pad_width));
    if (!set_kernel_arguments_success) {
      LOGE("Failed setting pooling OpenCL kernels[0] arguments.");
      return 1;
    }

    this->FineTuneGroupSize(this->kernels[0], this->_top_blobs[this->_top[0]]->height(), this->_top_blobs[this->_top[0]]->width());
    return 0;
  }

template <class Dtype>
int PoolingLayerCL<Dtype>::ForwardCL() {
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
      LOGE("Failed enqueuing the pooling kernel. %d", error_num);
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
      LOGE("Failed enqueuing the pooling kernel. %d", error_num);
      return -1;
    }

#endif



    return 0;
}

template <class Dtype>
int PoolingLayerCL<Dtype>::ForwardReshapeCL() {
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
        cl::Buffer* output_mem = this->_top_blobs[this->_top[0]]->data_cl();
        set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(1, *output_mem));
    }

    int param_idx = 3;
    cl::Buffer* input_mem = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(0, *input_mem));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->input_height));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->input_width));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->output_height));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->output_width));


    if (!set_kernel_arg_success) {
      LOGE("Failed setting conv OpenCL kernels[0] arguments.");
      return 1;
    }
    SetWorkSize();
    this->FineTuneGroupSize(this->kernels[0], this->_top_blobs[this->_top[0]]->height(), this->_top_blobs[this->_top[0]]->width());
    return this->ForwardCL();
}

template <class Dtype>
void PoolingLayerCL<Dtype>::FinetuneKernel() {
    size_t padded_output_c = this->_top_blobs[this->_top[0]]->get_channels_padding();
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
int PoolingLayerCL<Dtype>::GenerateTopBlobs() {
    //Only accept a single bottom blob.
    const Blob<Dtype> *bottom_blob = this->_bottom_blobs[this->_bottom[0]];
    this->input_height = bottom_blob->height();
    this->input_width = bottom_blob->width();
    this->input_channels = bottom_blob->channels();
    AssignOutputSize();
    this->output_channels = this->input_channels;
    this->_top_blobs[this->_top[0]] = new Blob<Dtype>(1, this->output_channels, this->output_height, this->output_width);
    this->_top_blobs[this->_top[0]]->AllocDevice(this->rt_param->context(), this->_top_blobs[this->_top[0]]->data_size_padded_c());
    FinetuneKernel();
    SetWorkSize();

    return 0;
}

template <class Dtype>
inline void PoolingLayerCL<Dtype>::AssignOutputSize()
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

template <class Dtype>
int PoolingLayerCL<Dtype>::SetWorkSize()
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

template class PoolingLayerCL<float>;
template class PoolingLayerCL<uint16_t>;

}; // namespace feather
