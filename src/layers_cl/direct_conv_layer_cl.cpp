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
#include "direct_conv_layer_cl.h"

namespace feather {
//#define USE_LEGACY_SGEMM
DirectConvLayerCL::DirectConvLayerCL(const LayerParameter *layer_param, RuntimeParameter<float>* rt_param)
                        : fuse_relu(false), Layer<uint16_t>(layer_param, rt_param)
{
    _fusible = true;
    const ConvolutionParameter *conv_param = layer_param->convolution_param();
    bias_term = conv_param->bias_term();

    group = conv_param->group();
    if(group == 0)  group = 1;
    kernel_height = conv_param->kernel_h();
    kernel_width = conv_param->kernel_w();

    stride_height = conv_param->stride_h();
    stride_width = conv_param->stride_w();

    padding_left = conv_param->pad_w();
    padding_top = conv_param->pad_h();
    padding_right = conv_param->pad_w();
    padding_bottom = conv_param->pad_h();

    assert(_weight_blobs.size() > 0);

    kernel_data = this->_weight_blobs[0]->data();
    output_channels = this->_weight_blobs[0]->num();

    if(stride_width  == 0)	stride_width  = 1;
    if(stride_height == 0) 	stride_height = 1;
    if (bias_term)
    {
        assert(this->_weight_blobs.size() == 2);
        bias_data = this->_weight_blobs[1]->data();
    }
    InitCL();
}

int DirectConvLayerCL::InitCL()
{
    std::string func_name = "convolution";
    this->cl_kernel_functions.push_back(func_name);
    std::string kernel_name_4o4 = "clConvIn4o4";
    auto it_source0 = booster::opencl_kernel_string_map.find("convBufferReformW4o4_1v1_grp1");
    std::string kernel_str_4o4(it_source0->second.begin(), it_source0->second.end());

    std::string kernel_name_8o8 = "clConvIn8o8";
    auto it_source1 = booster::opencl_kernel_string_map.find("convBufferReformW8o8_1v1_grp1");
    std::string kernel_str_8o8(it_source1->second.begin(), it_source1->second.end());

    // string kernel_name_16o16 = "clConvIn16o16";
    // auto it_source2 = booster::opencl_kernel_string_map.find("convBufferReformW16o16_1v1_grp1");
    // std::string kernel_str_16o16(it_source2->second.begin(), it_source2->second.end());

    this->cl_kernel_names.push_back(kernel_name_4o4);
    this->cl_kernel_names.push_back(kernel_name_8o8);
    //this->cl_kernel_names.push_back(kernel_name_16o16);

    this->cl_kernel_symbols.push_back(kernel_str_4o4);
    this->cl_kernel_symbols.push_back(kernel_str_8o8);
    //this->cl_kernel_symbols.push_back(kernel_str_16o16);

    cl_kernel kernel;
    this->kernels.push_back(kernel);
    cl_event event;
    this->events.push_back(event);
    return 0;
}

int DirectConvLayerCL::SetKernelParameters()
{
    int error_num;
    int param_idx = 0;
    bool set_kernel_arg_success = true;
    int out_real_channels = this->_top_blobs[this->_top[0]]->get_channels_padding();
    int in_real_channels = this->_bottom_blobs[this->_bottom[0]]->get_channels_padding();

    //size_t c_grp_size = this->in_channel_grp_size;
    size_t c_grp_size = 1;
    size_t n_grp_size = this->out_channel_grp_size;
    size_t w_num = this->_weight_blobs[0]->num();
    size_t w_channels = this->_weight_blobs[0]->channels();
    size_t w_hw = this->_weight_blobs[0]->height() * this->_weight_blobs[0]->width();

    size_t real_weight_size = this->_weight_blobs[0]->data_size_padded_nc();
    this->_weight_blobs[0]->AllocDevice(rt_param->context(), real_weight_size);
    std::vector<uint16_t> weight_padding(real_weight_size, 0);


    // for (int i = 0; i < w_num; ++i) {
    //     for (int j = 0; j < w_hw; ++j) {
    //         for (int k = 0; k < w_channels; ++k) {
    //             int dst_idx = (i * w_hw + j) * this->_weight_blobs[0]->get_channels_padding() + k;
    //             int src_idx = (i * this->_weight_blobs[0]->channels() + k) * w_hw + j;
    //             weight_padding[dst_idx] = kernel_data[src_idx];
    //         }
    //     }
    // }
    // printf("%d, %d\n", c_grp_size, n_grp_size);

    for (int i = 0; i < w_num; ++i) {
      for (int k = 0; k < w_channels; ++k) {
        for (int j = 0; j < w_hw; ++j) {
          int src_idx = (i * w_channels + k) * w_hw + j;
          int dst_idx = (i / n_grp_size) * w_hw * this->_weight_blobs[0]->get_channels_padding() * n_grp_size +
                        j * this->_weight_blobs[0]->get_channels_padding() * n_grp_size +
                        ( k / c_grp_size ) * n_grp_size * c_grp_size +
                        ( i % n_grp_size ) * c_grp_size +
                        k % c_grp_size;
          weight_padding[dst_idx] = kernel_data[src_idx];
          //printf("%f, ", hs_halfToFloat(weight_padding[dst_idx]));
        }
      }
    }

    this->_weight_blobs[0]->WriteToDevice(rt_param->command_queue(), weight_padding.data(), real_weight_size);
    this->_weight_blobs[0]->Free();

    if (bias_term) {
      this->_weight_blobs[1]->AllocDevice(rt_param->context(), out_real_channels);
      std::vector<uint16_t> bias_padding(out_real_channels, 0);
      memcpy(bias_padding.data(), bias_data, out_real_channels * sizeof(uint16_t));
      // float bias_padding[out_real_channels];
      // memset(bias_padding, 0.0f, out_real_channels * sizeof(float));
      // for (int i = 0; i < this->_weight_blobs[0]->num(); ++i) {
      //   bias_padding[i] = bias_data[i];
      // }
      this->_weight_blobs[1]->WriteToDevice(rt_param->command_queue(), bias_padding.data(), out_real_channels);
      this->_weight_blobs[1]->Free();
    }

    kernels[0] = clCreateKernel(cl_programs[0], cl_kernel_functions[0].c_str(), &error_num);
    if (!checkSuccess(error_num)) {
      LOGE("Failed to create conv OpenCL kernels[0]. ");
      return 1;
    }


    cl_mem input_mem = _bottom_blobs[_bottom[0]]->data_cl();
    cl_mem weight_mem = _weight_blobs[0]->data_cl();
    cl_mem output_mem = _top_blobs[_top[0]]->data_cl();
    int use_relu = fuse_relu;

    cl_mem bias_mem;
    if (bias_term) {
      bias_mem = _weight_blobs[1]->data_cl();
    } else {
      std::vector<uint16_t> bias_vec(out_real_channels, 0);
      //half bias_arr[out_real_channels];
      //memset(bias_arr, 0, out_real_channels * sizeof(half));
      bias_mem = clCreateBuffer(rt_param->context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                out_real_channels * sizeof(uint16_t), bias_vec.data(), &error_num);
      if (!checkSuccess(error_num)) {
        LOGE("Failed to create OpenCL buffers[%d]", error_num);
        return -1;
      }
    }

    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_mem), &input_mem));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_mem), &weight_mem));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_mem), &bias_mem));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_mem), &output_mem));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &in_real_channels));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &out_real_channels));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->input_height));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->input_width));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->output_height));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->output_width));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->kernel_height));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->kernel_width));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->stride_height));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->stride_width));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->padding_top));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->padding_left));
    set_kernel_arg_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &use_relu));
    if (!set_kernel_arg_success) {
      LOGE("Failed setting conv OpenCL kernels[0] arguments.");
      return 1;
    }
    FineTuneGroupSize(this->kernels[0], this->_top_blobs[this->_top[0]]->height(), this->_top_blobs[this->_top[0]]->width());
    return 0;
  }

int DirectConvLayerCL::ForwardCL()
{
#ifdef TIMING_CL
    clFinish(rt_param->command_queue());
    timespec tpstart, tpend;
    clock_gettime(CLOCK_MONOTONIC, &tpstart);
#endif
    if(group <=0)	group = 1;
    // LOGI("Forward layer (GPU_CL) %s", this->name().c_str());
    // LOGI("kernel (GPU_CL) %dx%d", kernel_height, kernel_width);
    // LOGI("stride (GPU_CL) %d %d", stride_height, stride_width);
    // LOGI("input (GPU_CL) %dx%d", input_height, input_width);
    // LOGI("output (GPU_CL) %dx%d", output_height, output_width);
    // LOGI("padding (GPU_CL) %d %d", padding_left, padding_top);
    // LOGI("globalWorkSize (GPU_CL): %d, %d, %d", global_work_size[0], global_work_size[1], global_work_size[2]);
    int error_num = clEnqueueNDRangeKernel(rt_param->command_queue(), kernels[0], 3, NULL, this->global_work_size, this->local_work_size, 0, NULL,&events[0]);
    if (!checkSuccess(error_num)) {
      LOGE("Failed enqueuing the conv kernel. %d", error_num);
      return -1;
    }

		/* if we wanna do something for event in future */
#ifdef TIMING_CL
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
#endif

		error_num = clReleaseEvent(events[0]);
		if (!checkSuccess(error_num)) {
      LOGE("Failed release event.");
      return -1;
    }

    return 0;
  }

void DirectConvLayerCL::FinetuneKernel()
{
    std::string cur_kname;
    std::string cur_kstr;
    size_t padded_input_c = _bottom_blobs[_bottom[0]]->get_channels_padding();
    size_t padded_output_c = _top_blobs[_top[0]]->get_channels_padding();

    int kernel_idx = 0, group_size = 4;
    if (padded_input_c % 8 == 0 && padded_output_c % 8 == 0) {
      kernel_idx = 1;
      group_size = 8;
    }

    cur_kname = cl_kernel_names[kernel_idx];
    cur_kstr = cl_kernel_symbols[kernel_idx];
    this->global_work_size[2] = padded_output_c / group_size;
    this->in_channel_grp_size = group_size;
    this->out_channel_grp_size = group_size;

    cl_kernel_names.clear();
    cl_kernel_symbols.clear();

    cl_kernel_names.push_back(cur_kname);
    cl_kernel_symbols.push_back(cur_kstr);
  }

int DirectConvLayerCL::GenerateTopBlobs() {
    //Conv layer has and only has one bottom blob.

    const Blob<uint16_t> *bottom_blob = this->_bottom_blobs[this->_bottom[0]];

    this->input_width = bottom_blob->width();
    this->input_height = bottom_blob->height();
    this->input_channels = bottom_blob->channels();
    if (stride_width == 0 || stride_height == 0)
    {
        stride_width = 1;
        stride_height = 1;
    }
    this->output_width = (this->input_width + this->padding_left + this->padding_right - this->kernel_width) / this->stride_width + 1;
    this->output_height = (this->input_height + this->padding_top + this->padding_bottom - this->kernel_height) / this->stride_height + 1;
    if(this->group > 1 && this->group == this->input_channels)
    {
        this->output_channels = this->group;
    }

     this->_top_blobs[this->_top[0]] = new Blob<uint16_t>(1, this->output_channels, this->output_height, this->output_width);
     this->_top_blobs[this->_top[0]]->AllocDevice(rt_param->context(), this->_top_blobs[this->_top[0]]->data_size_padded_c());

    FinetuneKernel();

    int group_size_h = 8, group_size_w = 8;
    if (this->output_width > 32) group_size_w = 16;
    if (this->output_height > 32) group_size_h = 16;
    //LOGI("out: %d %d", output_height, output_width)
    this->global_work_size[0] = (this->output_height / group_size_h + !!(this->output_height % group_size_h)) * group_size_h;
    this->global_work_size[1] = (this->output_width / group_size_w  + !!(this->output_width % group_size_w)) * group_size_w;
    this->local_work_size[0] = group_size_h;
    this->local_work_size[1] = group_size_w;
    // int work_size_w = this->globalWorkSize[1] / 2;
    // if(work_size_w >= this->localWorkSize[1] && work_size_w % this->localWorkSize[1] == 0){
    //   this->globalWorkSize[1] = work_size_w;
    // }

    if (this->global_work_size[2] > 4 && this->global_work_size[2] % 4 == 0) {
      this->local_work_size[2] = 4;
    } else {
      this->local_work_size[2] = 1;
    }

    return 0;
  }


int DirectConvLayerCL::Fuse(Layer *next_layer)
{
    if (next_layer->type().compare("ReLU") == 0) {
      fuse_relu = true;
      return 1;
    } else {
      return 0;
    }
  }

}; // namespace feather
