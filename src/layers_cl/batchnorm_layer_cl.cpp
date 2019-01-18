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

#include "batchnorm_layer_cl.h"

namespace feather
{
template <class Dtype>
BatchNormLayerCL<Dtype>::BatchNormLayerCL(const LayerParameter* layer_param, RuntimeParameter<Dtype>* rt_param)
      : output_channels(0),
        output_width(0),
        output_height(0),
        scale_bias_term(false),
        scale_bias_data(NULL),
        fuse_scale(false),
        scale_data(NULL),
        fuse_relu(false),
        Layer<Dtype>(layer_param, rt_param)
{
    this->_fusible = true;
    InitCL();
}


template <class Dtype>
int BatchNormLayerCL<Dtype>::InitCL()
{
    std::string program_name = "batchnorm_buffer";
    std::string kernel_name = "batchnorm";
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
int BatchNormLayerCL<Dtype>::SetBuildOptions()
{

    std::ostringstream ss;
    clhpp_feather::CLKernelInfo& bn_kernel_info = this->cl_kernel_info_map["batchnorm"];
    std::vector<std::string>& build_options = bn_kernel_info.build_options;
    ss << this->channel_block_size;
    build_options.push_back("-DN=" + ss.str());
    if (std::is_same<Dtype, uint16_t>::value)
        build_options.push_back("-DDATA_TYPE=half");
    else
        build_options.push_back("-DDATA_TYPE=float");

    if (fuse_relu)
    {
        build_options.push_back("-DUSE_RELU");
    }
    return 0;
}

template <class Dtype>
void BatchNormLayerCL<Dtype>::PadParamsDevice(Blob<Dtype>* blob, Dtype* data)
{
    size_t data_size = blob->data_size_padded_n();
    blob->AllocDevice(this->rt_param->context(), data_size);
    std::vector<Dtype> padded(data_size, 0);
    memcpy(padded.data(), data, data_size * sizeof(Dtype));
    blob->WriteToDevice(this->rt_param->command_queue(), padded.data(), data_size);
    blob->Free();
}

template <class Dtype>
int BatchNormLayerCL<Dtype>::PreCalParams()
{
    float *mean_data = this->_weight_blobs[0]->data_float(),
          *var_data = this->_weight_blobs[1]->data_float();
    float scale_factor = 1 / *(this->_weight_blobs[2]->data_float());
    float eps = 1e-5;
    for (int i = 0; i < this->output_channels; i++)
    {
        float sqrt_var = sqrt(var_data[i] * scale_factor + eps);
        float alpha_float = -(mean_data[i] * scale_factor) / sqrt_var;
        float beta_float = 1 / sqrt_var;
        if (this->fuse_scale)
        {
            alpha_float *= this->scale_data[i];
            beta_float *= this->scale_data[i];
            if (this->scale_bias_term)
            {
                alpha_float += this->scale_bias_data[i];
            }
        }
        if (std::is_same<Dtype, uint16_t>::value)
        {
            alpha[i] = hs_floatToHalf(alpha_float);
            beta[i]  = hs_floatToHalf(beta_float);
        }
        else
        {
            alpha[i] = alpha_float;
            beta[i] = beta_float;
        }

    }

    return 0;
}

template <class Dtype>
int BatchNormLayerCL<Dtype>::SetKernelParameters()
{
    int error_num;
    bool set_kernel_arguments_success = true;
    int param_idx = 0;
    this->rt_param->cl_runtime()->BuildKernel("batchnorm", this->cl_kernel_info_map);
    clhpp_feather::CLKernelInfo& bn_kernel_info = this->cl_kernel_info_map["batchnorm"];
    std::vector<size_t>& bn_gws = bn_kernel_info.gws;
    std::vector<size_t>& bn_lws = bn_kernel_info.lws;
    cl::Kernel& cl_kernel = bn_kernel_info.kernel;
    cl::Buffer* input_mem = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    cl::Buffer* output_mem = this->_top_blobs[this->_top[0]]->data_cl();

    PreCalParams();
    PadParamsDevice(this->_weight_blobs[0], this->alpha);
    PadParamsDevice(this->_weight_blobs[1], this->beta);
    MEMPOOL_CHECK_RETURN(this->private_mempool.Free(&this->alpha));
    MEMPOOL_CHECK_RETURN(this->private_mempool.Free(&this->beta));
    if (this->fuse_scale)
    {
        free(this->scale_data);
        this->scale_data = NULL;
        if (this->scale_bias_term)
        {
          free(this->scale_bias_data);
          this->scale_bias_data = NULL;
        }
    }

    cl::Buffer* alpha_mem = this->_weight_blobs[0]->data_cl();
    cl::Buffer* beta_mem = this->_weight_blobs[1]->data_cl();


    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, *input_mem));
    set_kernel_arguments_success &=  checkSuccess(cl_kernel.setArg(param_idx++, *alpha_mem));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, *beta_mem));
    set_kernel_arguments_success &=
    checkSuccess(cl_kernel.setArg(param_idx++, this->output_height));
    set_kernel_arguments_success &=
    checkSuccess(cl_kernel.setArg(param_idx++, this->output_width));
    set_kernel_arguments_success &=
    checkSuccess(cl_kernel.setArg(param_idx++, this->output_channels));
    set_kernel_arguments_success &=
    checkSuccess(cl_kernel.setArg(param_idx++, *output_mem));
    this->rt_param->cl_runtime()->FineTuneGroupSize(cl_kernel, this->output_height, this->output_width, bn_gws.data(), bn_lws.data());
    return 0;
}
template <class Dtype>
int BatchNormLayerCL<Dtype>::ForwardCL()
{
    clhpp_feather::CLKernelInfo& bn_kernel_info = this->cl_kernel_info_map["batchnorm"];
    std::vector<size_t>& bn_gws = bn_kernel_info.gws;
    std::vector<size_t>& bn_lws = bn_kernel_info.lws;
    cl::Kernel& cl_kernel = bn_kernel_info.kernel;

#ifdef TIMING_CL
  this->rt_param->command_queue().finish();
  std::string cl_program_name = bn_kernel_info.program_name;
  timespec tpstart, tpend;
  cl::Event event;
  clock_gettime(CLOCK_MONOTONIC, &tpstart);
  int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                      cl_kernel, cl::NullRange, cl::NDRange(bn_gws[0], bn_gws[1], bn_gws[2]),
                      cl::NDRange(bn_lws[0], bn_lws[1], bn_lws[2]), nullptr, &event);
  if (!checkSuccess(error_num))
  {
      LOGE("Failed enqueuing the batchnorm kernel.");
      return -1;
  }

  event.wait();
  clock_gettime(CLOCK_MONOTONIC, &tpend);
  double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
  LOGI("[%s] Execution time in %lf ms with %s\n", this->name().c_str(), timedif / 1000.0, cl_program_name.c_str());

  cl::Event profileEvent = event;
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
  std::string key_gws = this->name() + "_" + "batchnorm" + "_gws";
  std::string key_lws = this->name() + "_" + "batchnorm" + "_lws";
  if (clhpp_feather::IsTuning())
  {
      //warm up
      int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                          cl_kernel, cl::NullRange, cl::NDRange(bn_gws[0], bn_gws[1], bn_gws[2]),
                          cl::NDRange(bn_lws[0], bn_lws[1], bn_lws[2]), nullptr, nullptr);
      if (!checkSuccess(error_num))
      {
          LOGE("Failed enqueuing the batchnorm kernel.");
          return -1;
      }
      //run
      std::vector<std::vector<size_t> > gws_list;
      std::vector<std::vector<size_t> > lws_list;
      gws_list.push_back(bn_gws);
      lws_list.push_back(bn_lws);
      uint64_t kwg_size = 0;
      this->rt_param->cl_runtime()->GetKernelMaxWorkGroupSize(cl_kernel, kwg_size);
      this->rt_param->cl_runtime()->tuner().TunerArry(kwg_size, this->output_height, this->output_width,
              bn_gws, bn_lws, gws_list, lws_list);
      double opt_time = std::numeric_limits<double>::max();
      int min_tune = -1;
      for (int j = 0; j < gws_list.size(); j++)
      {
          this->rt_param->command_queue().finish();
          timespec tpstart, tpend;
          clock_gettime(CLOCK_MONOTONIC, &tpstart);
          int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                              cl_kernel, cl::NullRange, cl::NDRange(gws_list[j][0], gws_list[j][1], gws_list[j][2]),
                              cl::NDRange(lws_list[j][0], lws_list[j][1], lws_list[j][2]), nullptr, nullptr);
          if (!checkSuccess(error_num))
          {
              LOGE("Failed enqueuing the batchnorm kernel.");
              return -1;
          }

          this->rt_param->command_queue().finish();
          clock_gettime(CLOCK_MONOTONIC, &tpend);
          double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
          timedif /= 1000.0;
          if (timedif < opt_time)
          {
              opt_time = timedif;
              min_tune = j;
          }
      }

      this->rt_param->cl_runtime()->tuner().set_layer_kernel_wks(key_gws, gws_list[min_tune], opt_time);
      this->rt_param->cl_runtime()->tuner().set_layer_kernel_wks(key_lws, lws_list[min_tune], opt_time);
  }
  else if (clhpp_feather::IsTunned())
  {
      std::vector<size_t> tmp_gws;
      std::vector<size_t> tmp_lws;
      this->rt_param->cl_runtime()->tuner().get_layer_kernel_wks(key_gws, tmp_gws);
      this->rt_param->cl_runtime()->tuner().get_layer_kernel_wks(key_lws, tmp_lws);
      int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                          cl_kernel, cl::NullRange, cl::NDRange(tmp_gws[0], tmp_gws[1], tmp_gws[2]),
                          cl::NDRange(tmp_lws[0], tmp_lws[1], tmp_lws[2]), nullptr, nullptr);
      if (!checkSuccess(error_num))
      {
          LOGE("Failed enqueuing the batchnorm kernel.");
          return -1;
      }
  }
  else if (clhpp_feather::IsTunerInProcess())
  {
      //run
      std::vector<std::vector<size_t> > gws_list;
      std::vector<std::vector<size_t> > lws_list;
      gws_list.push_back(bn_gws);
      lws_list.push_back(bn_lws);
      uint64_t kwg_size = 0;
      this->rt_param->cl_runtime()->GetKernelMaxWorkGroupSize(cl_kernel, kwg_size);
      this->rt_param->cl_runtime()->tuner().IsTunerInProcess(kwg_size, this->output_height, this->output_width,
              bn_gws, bn_lws, gws_list, lws_list);
      this->rt_param->command_queue().finish();
      timespec tpstart, tpend;
      clock_gettime(CLOCK_MONOTONIC, &tpstart);
      int j = 0;
      int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                          cl_kernel, cl::NullRange, cl::NDRange(gws_list[j][0], gws_list[j][1], gws_list[j][2]),
                          cl::NDRange(lws_list[j][0], lws_list[j][1], lws_list[j][2]), nullptr, nullptr);
      if (!checkSuccess(error_num))
      {
          LOGE("Failed enqueuing the batchnorm kernel.");
          return -1;
      }
      this->rt_param->command_queue().finish();
      clock_gettime(CLOCK_MONOTONIC, &tpend);
      double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
      timedif /= 1000.0;
      this->rt_param->cl_runtime()->tuner().set_layer_kernel_wks(key_gws, gws_list[j], timedif);
      this->rt_param->cl_runtime()->tuner().set_layer_kernel_wks(key_lws, lws_list[j], timedif);
  }
  else
  {
      int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                          cl_kernel, cl::NullRange, cl::NDRange(bn_gws[0], bn_gws[1], bn_gws[2]),
                          cl::NDRange(bn_lws[0], bn_lws[1], bn_lws[2]), nullptr, nullptr);
      if (!checkSuccess(error_num))
      {
          LOGE("Failed enqueuing the batchnorm kernel.");
          return -1;
      }
  }
#endif
    return 0;
}

template <class Dtype>
int BatchNormLayerCL<Dtype>::ForwardReshapeCL()
{
    if (this->output_height == this->_bottom_blobs[this->_bottom[0]]->height() &&
            this->output_width == this->_bottom_blobs[this->_bottom[0]]->width())
        return this->ForwardCL();

    bool set_kernel_arg_success = true;
    clhpp_feather::CLKernelInfo& batchnorm_kernel_info = this->cl_kernel_info_map["batchnorm"];
    std::vector<size_t>& batchnorm_gws = batchnorm_kernel_info.gws;
    std::vector<size_t>& batchnorm_lws = batchnorm_kernel_info.lws;
    cl::Kernel& cl_kernel = batchnorm_kernel_info.kernel;


    this->output_height = this->_bottom_blobs[this->_bottom[0]]->height();
    this->output_width = this->_bottom_blobs[this->_bottom[0]]->width();
    if (this->_top_blobs[this->_top[0]]->ReshapeWithReallocDevice(this->rt_param->context(),
            this->_top_blobs[this->_top[0]]->num(),
            this->_top_blobs[this->_top[0]]->channels(),
            this->output_height, this->output_width) == 2)
    {
        cl::Buffer* output_mem = this->_top_blobs[this->_top[0]]->data_cl();
        set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(6, *output_mem));
    }
    cl::Buffer* input_mem = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(0, *input_mem));
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(3, this->output_height));
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(4, this->output_width));
    if (!set_kernel_arg_success)
    {
        LOGE("Failed setting batchnorm reshape cl_kernels arguments.");
        return 1;
    }

    this->ResetWorkSize("batchnorm", this->output_height, this->output_width);
    this->rt_param->cl_runtime()->FineTuneGroupSize(cl_kernel, this->output_height, this->output_width, batchnorm_gws.data(), batchnorm_lws.data());
    return this->ForwardCL();
}

template <class Dtype>
int BatchNormLayerCL<Dtype>::GenerateTopBlobs()
{
    if (this->_top.size() != 1 || this->_bottom.size() != 1)
        return -1;
    Blob<Dtype>* p_blob = new Blob<Dtype>();
    p_blob->CopyShape(this->_bottom_blobs[this->_bottom[0]]);
    this->output_height = p_blob->height();
    this->output_width = p_blob->width();
    this->output_channels = p_blob->channels();
    MEMPOOL_CHECK_RETURN(this->private_mempool.Alloc(&alpha, this->output_channels * sizeof(Dtype)));
    MEMPOOL_CHECK_RETURN(this->private_mempool.Alloc(&beta, this->output_channels * sizeof(Dtype)));
    p_blob->AllocDevice(this->rt_param->context(), p_blob->data_size_padded_c());
    this->_top_blobs[this->_top[0]] = p_blob;
    this->SetWorkSize("batchnorm", this->output_height, this->output_width, this->channel_block_size);
    return 0;
}

template <class Dtype>
int BatchNormLayerCL<Dtype>::Fuse(Layer<Dtype> *next_layer)
{
    if (next_layer->type().compare("ReLU") == 0)
    {
        fuse_relu = true;
        return 1;
    }
    else if (next_layer->type().compare("Scale") == 0)
    {
        LOGI("BN %s fuse Scale layer %s\n", this->name().c_str(), next_layer->name().c_str());
        this->fuse_scale = true;
        this->scale_bias_term = ((ScaleLayerCL<Dtype>*) next_layer)->bias_term();
        // MEMPOOL_CHECK_RETURN(this->private_mempool.Alloc(&this->scale_data, this->output_channels * sizeof(float)));
        this->scale_data = (float *)malloc(sizeof(float) * this->output_channels);
        if (this->scale_bias_term)
        {
            // MEMPOOL_CHECK_RETURN(this->private_mempool.Alloc(&this->scale_bias_data, this->output_channels * sizeof(float)));
            this->scale_bias_data = (float *)malloc(sizeof(float) * this->output_channels);
        }
        memcpy(this->scale_data, next_layer->weight_blob(0)->data_float(), sizeof(float) * this->output_channels);
        memcpy(this->scale_bias_data, next_layer->weight_blob(1)->data_float(), sizeof(float) * this->output_channels);

        LOGI("BN fuse scale done...");
        return 1;
    }
    else
    {
        return 0;
    }
}

template class BatchNormLayerCL<float>;
template class BatchNormLayerCL<uint16_t>;

};
