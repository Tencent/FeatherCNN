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

#include "scale_layer_cl.h"

namespace feather
{
template <class Dtype>
ScaleLayerCL<Dtype>::ScaleLayerCL(const LayerParameter* layer_param, RuntimeParameter<Dtype>* rt_param)
      : output_channels(0),
        output_width(0),
        output_height(0),
        _bias_term(false),
        fuse_relu(false),
        Layer<Dtype>(layer_param, rt_param)
{
    this->_fusible = true;
    this->_bias_term = layer_param->scale_param()->bias_term();
    this->InitKernelInfo("scale", "scale_buffer");
}


template <class Dtype>
int ScaleLayerCL<Dtype>::SetBuildOptions()
{

    std::ostringstream ss;
    clhpp_feather::CLKernelInfo& scale_kernel_info = this->cl_kernel_info_map["scale"];
    std::vector<std::string>& build_options = scale_kernel_info.build_options;
    ss << this->channel_block_size;
    build_options.push_back("-DN=" + ss.str());
    if (std::is_same<Dtype, uint16_t>::value)
        build_options.push_back("-DDATA_TYPE=half");
    else
        build_options.push_back("-DDATA_TYPE=float");

    if (this->_bias_term)
    {
        build_options.push_back("-DBIAS");
    }
    if (fuse_relu)
    {
        build_options.push_back("-DUSE_RELU");
    }
    return 0;
}

template <class Dtype>
void ScaleLayerCL<Dtype>::PadParamsDevice(Blob<Dtype>* blob, Dtype* data)
{
    size_t data_size = blob->data_size_padded_n();
    blob->AllocDevice(this->rt_param->context(), data_size);
    std::vector<Dtype> padded(data_size, 0);
    memcpy(padded.data(), data, data_size * sizeof(Dtype));
    blob->WriteToDevice(this->rt_param->command_queue(), padded.data(), data_size);
    blob->Free();
}

template <class Dtype>
int ScaleLayerCL<Dtype>::SetKernelParameters()
{
    int error_num;
    bool set_kernel_arguments_success = true;
    int param_idx = 0;
    this->rt_param->cl_runtime()->BuildKernel("scale", this->cl_kernel_info_map);
    clhpp_feather::CLKernelInfo& scale_kernel_info = this->cl_kernel_info_map["scale"];
    std::vector<size_t>& scale_gws = scale_kernel_info.gws;
    std::vector<size_t>& scale_lws = scale_kernel_info.lws;
    cl::Kernel& cl_kernel = scale_kernel_info.kernel;
    cl::Buffer* input_mem = this->_bottom_blobs[this->_bottom[0]]->data_cl_buffer();
    cl::Buffer* output_mem = this->_top_blobs[this->_top[0]]->data_cl_buffer();

    PadParamsDevice(this->_weight_blobs[0], this->_weight_blobs[0]->data());
    cl::Buffer* scale_mem = this->_weight_blobs[0]->data_cl_buffer();
    cl::Buffer* bias_mem = NULL;
    if(this->_bias_term){
        PadParamsDevice(this->_weight_blobs[1], this->_weight_blobs[1]->data());
        bias_mem = this->_weight_blobs[1]->data_cl_buffer();
    }

    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, *input_mem));
    set_kernel_arguments_success &=  checkSuccess(cl_kernel.setArg(param_idx++, *scale_mem));
    if (this->_bias_term)
    {
        set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(param_idx++, *bias_mem));
    }
    set_kernel_arguments_success &=
    checkSuccess(cl_kernel.setArg(param_idx++, this->output_height));
    set_kernel_arguments_success &=
    checkSuccess(cl_kernel.setArg(param_idx++, this->output_width));
    set_kernel_arguments_success &=
    checkSuccess(cl_kernel.setArg(param_idx++, this->output_channels));
    set_kernel_arguments_success &=
    checkSuccess(cl_kernel.setArg(param_idx++, *output_mem));
    this->rt_param->cl_runtime()->FineTuneGroupSize(cl_kernel, this->output_height, this->output_width, scale_gws.data(), scale_lws.data());
    return 0;
}
template <class Dtype>
int ScaleLayerCL<Dtype>::ForwardCL()
{
    clhpp_feather::CLKernelInfo& scale_kernel_info = this->cl_kernel_info_map["scale"];
    std::vector<size_t>& scale_gws = scale_kernel_info.gws;
    std::vector<size_t>& scale_lws = scale_kernel_info.lws;
    cl::Kernel& cl_kernel = scale_kernel_info.kernel;

#ifdef TIMING_CL
  this->rt_param->command_queue().finish();
  std::string cl_program_name = scale_kernel_info.program_name;
  timespec tpstart, tpend;
  cl::Event event;
  clock_gettime(CLOCK_MONOTONIC, &tpstart);
  int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                      cl_kernel, cl::NullRange, cl::NDRange(scale_gws[0], scale_gws[1], scale_gws[2]),
                      cl::NDRange(scale_lws[0], scale_lws[1], scale_lws[2]), nullptr, &event);
  if (!checkSuccess(error_num))
  {
      LOGE("Failed enqueuing the scale kernel.");
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
  std::string key_gws = this->name() + "_" + "scale" + "_gws";
  std::string key_lws = this->name() + "_" + "scale" + "_lws";
  if (clhpp_feather::IsTuning())
  {
      //warm up
      int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                          cl_kernel, cl::NullRange, cl::NDRange(scale_gws[0], scale_gws[1], scale_gws[2]),
                          cl::NDRange(scale_lws[0], scale_lws[1], scale_lws[2]), nullptr, nullptr);
      if (!checkSuccess(error_num))
      {
          LOGE("Failed enqueuing the scale kernel.");
          return -1;
      }
      //run
      std::vector<std::vector<size_t> > gws_list;
      std::vector<std::vector<size_t> > lws_list;
      gws_list.push_back(scale_gws);
      lws_list.push_back(scale_lws);
      uint64_t kwg_size = 0;
      this->rt_param->cl_runtime()->GetKernelMaxWorkGroupSize(cl_kernel, kwg_size);
      this->rt_param->cl_runtime()->tuner().TunerArry(kwg_size, this->output_height, this->output_width,
              scale_gws, scale_lws, gws_list, lws_list);
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
              LOGE("Failed enqueuing the scale kernel.");
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
          LOGE("Failed enqueuing the scale kernel.");
          return -1;
      }
  }
  else if (clhpp_feather::IsTunerInProcess())
  {
      //run
      std::vector<std::vector<size_t> > gws_list;
      std::vector<std::vector<size_t> > lws_list;
      gws_list.push_back(scale_gws);
      lws_list.push_back(scale_lws);
      uint64_t kwg_size = 0;
      this->rt_param->cl_runtime()->GetKernelMaxWorkGroupSize(cl_kernel, kwg_size);
      this->rt_param->cl_runtime()->tuner().IsTunerInProcess(kwg_size, this->output_height, this->output_width,
              scale_gws, scale_lws, gws_list, lws_list);
      this->rt_param->command_queue().finish();
      timespec tpstart, tpend;
      clock_gettime(CLOCK_MONOTONIC, &tpstart);
      int j = 0;
      int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                          cl_kernel, cl::NullRange, cl::NDRange(gws_list[j][0], gws_list[j][1], gws_list[j][2]),
                          cl::NDRange(lws_list[j][0], lws_list[j][1], lws_list[j][2]), nullptr, nullptr);
      if (!checkSuccess(error_num))
      {
          LOGE("Failed enqueuing the scale kernel.");
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
                          cl_kernel, cl::NullRange, cl::NDRange(scale_gws[0], scale_gws[1], scale_gws[2]),
                          cl::NDRange(scale_lws[0], scale_lws[1], scale_lws[2]), nullptr, nullptr);
      if (!checkSuccess(error_num))
      {
          LOGE("Failed enqueuing the scale kernel.");
          return -1;
      }
  }
#endif
    return 0;
}

template <class Dtype>
int ScaleLayerCL<Dtype>::ForwardReshapeCL()
{
    if (this->output_height == this->_bottom_blobs[this->_bottom[0]]->height() &&
            this->output_width == this->_bottom_blobs[this->_bottom[0]]->width())
        return this->ForwardCL();

    bool set_kernel_arg_success = true;
    clhpp_feather::CLKernelInfo& scale_kernel_info = this->cl_kernel_info_map["scale"];
    std::vector<size_t>& scale_gws = scale_kernel_info.gws;
    std::vector<size_t>& scale_lws = scale_kernel_info.lws;
    cl::Kernel& cl_kernel = scale_kernel_info.kernel;


    this->output_height = this->_bottom_blobs[this->_bottom[0]]->height();
    this->output_width = this->_bottom_blobs[this->_bottom[0]]->width();
    if (this->_top_blobs[this->_top[0]]->ReshapeWithReallocDevice(this->rt_param->context(),
            this->_top_blobs[this->_top[0]]->num(),
            this->_top_blobs[this->_top[0]]->channels(),
            this->output_height, this->output_width) == 2)
    {
        cl::Buffer* output_mem = this->_top_blobs[this->_top[0]]->data_cl_buffer();
        set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(6, *output_mem));
    }
    cl::Buffer* input_mem = this->_bottom_blobs[this->_bottom[0]]->data_cl_buffer();
    int param_idx = this->_bias_term ? 3 : 2;

    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(0, *input_mem));
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->output_height));
    set_kernel_arg_success &= checkSuccess(cl_kernel.setArg(param_idx++, this->output_width));
    if (!set_kernel_arg_success)
    {
        LOGE("Failed setting scale reshape cl_kernels arguments.");
        return 1;
    }

    this->ResetWorkSize("scale", this->output_height, this->output_width);
    this->rt_param->cl_runtime()->FineTuneGroupSize(cl_kernel, this->output_height, this->output_width, scale_gws.data(), scale_lws.data());
    return this->ForwardCL();
}

template <class Dtype>
int ScaleLayerCL<Dtype>::GenerateTopBlobs()
{
    if (this->_top.size() != 1 || this->_bottom.size() != 1)
        return -1;
    Blob<Dtype>* p_blob = new Blob<Dtype>();
    p_blob->CopyShape(this->_bottom_blobs[this->_bottom[0]]);
    this->output_height = p_blob->height();
    this->output_width = p_blob->width();
    this->output_channels = p_blob->get_channels_padding();

    p_blob->AllocDevice(this->rt_param->context(), p_blob->data_size_padded_c());
    this->_top_blobs[this->_top[0]] = p_blob;
    this->SetWorkSize("scale", this->output_height, this->output_width, this->channel_block_size);
    return 0;
}

template <class Dtype>
int ScaleLayerCL<Dtype>::Fuse(Layer<Dtype> *next_layer)
{
    if (next_layer->type().compare("ReLU") == 0)
    {
        fuse_relu = true;
        return 1;
    }
    else
    {
        return 0;
    }
}

template class ScaleLayerCL<float>;
template class ScaleLayerCL<uint16_t>;

};
