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

template <class Dtype>
DirectConvLayerCL<Dtype>::DirectConvLayerCL(const LayerParameter *layer_param, RuntimeParameter<float>* rt_param)
                        : fuse_relu(false), Layer<Dtype>(layer_param, rt_param)
{
    this->_fusible = true;
    const ConvolutionParameter *conv_param = layer_param->convolution_param();
    this->bias_term = conv_param->bias_term();

    this->group = conv_param->group();
    if(this->group == 0)  this->group = 1;
    this->kernel_height = conv_param->kernel_h();
    this->kernel_width = conv_param->kernel_w();

    this->stride_height = conv_param->stride_h();
    this->stride_width = conv_param->stride_w();

    this->padding_left = conv_param->pad_w();
    this->padding_top = conv_param->pad_h();
    this->padding_right = conv_param->pad_w();
    this->padding_bottom = conv_param->pad_h();
    this->is_dw = false;

    assert(this->_weight_blobs.size() > 0);

    this->kernel_data = this->_weight_blobs[0]->data();
    this->output_channels = this->_weight_blobs[0]->num();

    if(this->stride_width  == 0)	this->stride_width  = 1;
    if(this->stride_height == 0) 	this->stride_height = 1;
    if (this->bias_term)
    {
        assert(this->_weight_blobs.size() == 2);
        this->bias_data = this->_weight_blobs[1]->data();
    }
    InitCL();
}

template <class Dtype>
int DirectConvLayerCL<Dtype>::InitCL()
{
    std::string func_name_conv = "convolution";
    std::string func_name_depthwise = "convolution_depthwise";
    this->cl_kernel_functions.push_back(func_name_conv);
    this->cl_kernel_functions.push_back(func_name_depthwise);


    std::string kernel_name_conv = "conv_1v1_buffer";
    auto it_source0 = booster::opencl_kernel_string_map.find("conv_1v1_buffer");
    std::string kernel_str_conv(it_source0->second.begin(), it_source0->second.end());

    std::string kernel_name_depthwise_conv = "depthwise_conv_1v1_buffer";
    auto it_source1 = booster::opencl_kernel_string_map.find("depthwise_conv_1v1_buffer");
    std::string kernel_str_depthwise_conv(it_source1->second.begin(), it_source1->second.end());

    this->cl_kernel_names.push_back(kernel_name_conv);
    this->cl_kernel_names.push_back(kernel_name_depthwise_conv);

    this->cl_kernel_symbols.push_back(kernel_str_conv);
    this->cl_kernel_symbols.push_back(kernel_str_depthwise_conv);

    cl::Kernel kernel;
    this->kernels.push_back(kernel);
    cl::Event event;
    this->events.push_back(event);
    return 0;
}

template <class Dtype>
int DirectConvLayerCL<Dtype>::SetBuildOptions() {
    std::ostringstream ss;
    ss << out_channel_grp_size;
    this->build_options.push_back("-DN=" + ss.str());
    //this->build_options.push_back("-DDATA_TYPE=half");
    if(std::is_same<Dtype, uint16_t>::value)
      this->build_options.push_back("-DDATA_TYPE=half");
    else
      this->build_options.push_back("-DDATA_TYPE=float");

    if (this->bias_term) {
      this->build_options.push_back("-DBIAS");
    }
    if (this->fuse_relu) {
      this->build_options.push_back("-DUSE_RELU");
    }
    return 0;
}

template <class Dtype>
int DirectConvLayerCL<Dtype>::SetKernelParameters()
{
    int error_num;
    int param_idx = 0;
    bool set_kernel_arg_success = true;
    uint32_t out_real_channels = this->_top_blobs[this->_top[0]]->get_channels_padding();
    uint32_t in_real_channels = this->_bottom_blobs[this->_bottom[0]]->get_channels_padding();

    //size_t c_grp_size = this->in_channel_grp_size;
    size_t c_grp_size = 1;
    size_t n_grp_size = this->out_channel_grp_size;
    size_t w_num = this->_weight_blobs[0]->num();
    size_t w_channels = this->_weight_blobs[0]->channels();
    size_t w_hw = this->_weight_blobs[0]->height() * this->_weight_blobs[0]->width();

    uint32_t real_weight_size = 0;
    if (this->is_dw){
      real_weight_size = w_hw * this->_weight_blobs[0]->get_num_padding();;
    } else {
      real_weight_size = this->_weight_blobs[0]->data_size_padded_nc();
    }
    this->_weight_blobs[0]->AllocDevice(this->rt_param->context(), real_weight_size);
    std::vector<Dtype> weight_padding(real_weight_size, 0);

    if (this->is_dw) {
      for (int i = 0; i < w_num; ++i) {
        for (int j = 0; j < w_hw; ++j) {
          // int dst_idx = j * this->_weight_blobs[0]->get_num_padding() + i;
          int dst_idx = (i / this->in_channel_grp_size * w_hw + j)
                        * this->in_channel_grp_size
                        + i % this->in_channel_grp_size;
          int src_idx = i * w_hw + j;
          weight_padding[dst_idx] = this->kernel_data[src_idx];
        }
      }
    }
    else {
      // for (int i = 0; i < w_num; ++i) {
      //     for (int j = 0; j < w_hw; ++j) {
      //         for (int k = 0; k < w_channels; ++k) {
      //             int dst_idx = (i * w_hw + j) * this->_weight_blobs[0]->get_channels_padding() + k;
      //             int src_idx = (i * this->_weight_blobs[0]->channels() + k) * w_hw + j;
      //             weight_padding[dst_idx] = kernel_data[src_idx];
      //         }
      //     }
      // }

      for (int i = 0; i < w_num; ++i) {
        for (int k = 0; k < w_channels; ++k) {
          for (int j = 0; j < w_hw; ++j) {
            int src_idx = (i * w_channels + k) * w_hw + j;
            int dst_idx = (i / n_grp_size) * w_hw * this->_weight_blobs[0]->get_channels_padding() * n_grp_size +
                          j * this->_weight_blobs[0]->get_channels_padding() * n_grp_size +
                          ( k / c_grp_size ) * n_grp_size * c_grp_size +
                          ( i % n_grp_size ) * c_grp_size +
                          k % c_grp_size;
            weight_padding[dst_idx] = this->kernel_data[src_idx];
          }
        }
      }
    }


    this->_weight_blobs[0]->WriteToDevice(this->rt_param->command_queue(), weight_padding.data(), real_weight_size);
    this->_weight_blobs[0]->Free();

    if (bias_term) {
      this->_weight_blobs[1]->AllocDevice(this->rt_param->context(), out_real_channels);
      std::vector<Dtype> bias_padding(out_real_channels, 0);
      memcpy(bias_padding.data(), this->bias_data, this->output_channels * sizeof(Dtype));
      this->_weight_blobs[1]->WriteToDevice(this->rt_param->command_queue(), bias_padding.data(), out_real_channels);
      this->_weight_blobs[1]->Free();
    }

    //kernels[0] = clCreateKernel(this->cl_programs[0], this->cl_kernel_functions[0].c_str(), &error_num);
    this->kernels[0] = cl::Kernel(this->cl_programs[0], this->cl_kernel_functions[0].c_str(), &error_num);
    if (!checkSuccess(error_num)) {
      LOGE("Failed to create conv OpenCL kernels[0]. ");
      return 1;
    }


    cl::Buffer* input_mem  = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    cl::Buffer* weight_mem = this->_weight_blobs[0]->data_cl();
    cl::Buffer* output_mem = this->_top_blobs[this->_top[0]]->data_cl();


    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, *input_mem));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, *weight_mem));
    if (bias_term) {
      set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, *this->_weight_blobs[1]->data_cl()));
    }
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, *output_mem));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, in_real_channels));
    if (!this->is_dw) {
      set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, out_real_channels));
    }
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->input_height));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->input_width));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->output_height));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->output_width));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->kernel_height));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->kernel_width));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->stride_height));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->stride_width));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->padding_top));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->padding_left));

    if (!set_kernel_arg_success) {
      LOGE("Failed setting conv OpenCL kernels[0] arguments.");
      return 1;
    }
    this->FineTuneGroupSize(this->kernels[0], this->_top_blobs[this->_top[0]]->height(), this->_top_blobs[this->_top[0]]->width());
    return 0;
}

template <class Dtype>
int DirectConvLayerCL<Dtype>::ForwardReshapeCL()
{

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
        set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(3, *output_mem));
    }

    int param_idx = this->is_dw ? 5 : 6;
    cl::Buffer* input_mem = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    LOGD("data_size 1 %d", this->_bottom_blobs[this->_bottom[0]]->data_size());
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(0, *input_mem));
    LOGD("data_size 2 %d", this->_bottom_blobs[this->_bottom[0]]->data_size());
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->input_height));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->input_width));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->output_height));
    set_kernel_arg_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->output_width));

    if (!set_kernel_arg_success) {
      LOGE("Failed setting conv OpenCL kernels[0] arguments.");
      return 1;
    }

    this->SetWorkSize();
    this->FineTuneGroupSize(this->kernels[0], this->_top_blobs[this->_top[0]]->height(), this->_top_blobs[this->_top[0]]->width());
    return this->ForwardCL();
}

template <class Dtype>
int DirectConvLayerCL<Dtype>::ForwardCL()
{
#ifdef TIMING_CL
    clFinish(this->rt_param->command_queue());
    timespec tpstart, tpend;
    clock_gettime(CLOCK_MONOTONIC, &tpstart);

    int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
        kernels[0], cl::NullRange, cl::NDRange(global_work_size[0], global_work_size[1], global_work_size[2]),
        cl::NDRange(local_work_size[0], local_work_size[1], local_work_size[2]), nullptr, &events[0]);
    if (!checkSuccess(error_num)) {
      LOGE("Failed enqueuing the conv kernel.");
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
      LOGE("Failed enqueuing the conv kernel.");
      return -1;
    }


#endif

    return 0;
  }

template <class Dtype>
void DirectConvLayerCL<Dtype>::FinetuneKernel()
{
    std::string cur_kname;
    std::string cur_kstr;
    std::string cur_func;
    size_t padded_input_c = this->_bottom_blobs[this->_bottom[0]]->get_channels_padding();
    size_t padded_output_c = this->_top_blobs[this->_top[0]]->get_channels_padding();

    int group_size = 4;
    if (padded_input_c % 8 == 0 && padded_output_c % 8 == 0) {
      group_size = 8;
    }

    int kernel_idx = this->is_dw ? 1 : 0;
    int func_idx = this->is_dw ? 1 : 0;
    cur_kname = this->cl_kernel_names[kernel_idx];
    cur_kstr = this->cl_kernel_symbols[kernel_idx];
    cur_func = this->cl_kernel_functions[func_idx];

    this->global_work_size[2] = padded_output_c / group_size;
    this->in_channel_grp_size = group_size;
    this->out_channel_grp_size = group_size;

    this->cl_kernel_names.clear();
    this->cl_kernel_symbols.clear();
    this->cl_kernel_functions.clear();

    this->cl_kernel_names.push_back(cur_kname);
    this->cl_kernel_symbols.push_back(cur_kstr);
    this->cl_kernel_functions.push_back(cur_func);
}

template <class Dtype>
int DirectConvLayerCL<Dtype>::GenerateTopBlobs() {
    //Conv layer has and only has one bottom blob.

    const Blob<Dtype> *bottom_blob = this->_bottom_blobs[this->_bottom[0]];

    this->input_width = bottom_blob->width();
    this->input_height = bottom_blob->height();
    this->input_channels = bottom_blob->channels();
    if (this->stride_width == 0 || this->stride_height == 0)
    {
        this->stride_width = 1;
        this->stride_height = 1;
    }
    AssignOutputSize();
    if(this->group > 1 && this->group == this->input_channels)
    {
        this->output_channels = this->group;
        this->is_dw = true;
    }
     this->_top_blobs[this->_top[0]] = new Blob<Dtype>(1, this->output_channels, this->output_height, this->output_width);
     this->_top_blobs[this->_top[0]]->AllocDevice(this->rt_param->context(), this->_top_blobs[this->_top[0]]->data_size_padded_c());

    FinetuneKernel();
    SetWorkSize();

    return 0;
}

template <class Dtype>
inline void DirectConvLayerCL<Dtype>:: AssignOutputSize()
{
    this->output_width = (this->input_width + this->padding_left + this->padding_right - this->kernel_width) / this->stride_width + 1;
    this->output_height = (this->input_height + this->padding_top + this->padding_bottom - this->kernel_height) / this->stride_height + 1;
}

template <class Dtype>
int DirectConvLayerCL<Dtype>::SetWorkSize()
{
    if (this->output_width > 32) this->group_size_w = 16;
    if (this->output_height > 32) this->group_size_h = 16;

    this->global_work_size[0] = (this->output_height / this->group_size_h + !!(this->output_height % this->group_size_h)) * this->group_size_h;
    this->global_work_size[1] = (this->output_width / this->group_size_w  + !!(this->output_width % this->group_size_w)) * this->group_size_w;
    this->local_work_size[0] = this->group_size_h;
    this->local_work_size[1] = this->group_size_w;
    // int work_size_w = this->global_work_size[1] / 2;
    // if(work_size_w >= this->local_work_size[1] && work_size_w % this->local_work_size[1] == 0){
    //   this->global_work_size[1] = work_size_w;
    // }
    if (this->global_work_size[2] > 4 && this->global_work_size[2] % 4 == 0) {
      this->local_work_size[2] = 4;
    } else {
      this->local_work_size[2] = 1;
    }
    return 0;
}


template <class Dtype>
int DirectConvLayerCL<Dtype>::Fuse(Layer<Dtype> *next_layer)
{
  if (next_layer->type().compare("ReLU") == 0) {
    fuse_relu = true;
    return 1;
  } else {
    return 0;
  }
}

template class DirectConvLayerCL<float>;
template class DirectConvLayerCL<uint16_t>;

}; // namespace feather
