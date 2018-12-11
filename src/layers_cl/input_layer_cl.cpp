#include "input_layer_cl.h"

namespace feather {

template <class Dtype>
int InputLayerCL<Dtype>::InitCL()
{
    std::string func_name1  = "init1O4";
    std::string kernel_name1 = "clNormalInit";
    auto it_source1 = booster::opencl_kernel_string_map.find("input_buffer");
    std::string kernel_str1(it_source1->second.begin(),it_source1->second.end());

    std::string func_name2 = "init1O42D";
    std::string kernel_name2 = "clNormalInit2D";
    auto it_source2 = booster::opencl_kernel_string_map.find("inputImage");
    std::string kernel_str2(it_source2->second.begin(),it_source2->second.end());

    this->cl_kernel_functions.push_back(func_name1);
    this->cl_kernel_functions.push_back(func_name2);
    this->cl_kernel_names.push_back(kernel_name1);
    this->cl_kernel_names.push_back(kernel_name2);
    this->cl_kernel_symbols.push_back(kernel_str1);
    this->cl_kernel_symbols.push_back(kernel_str2);

    cl::Kernel kernel1;
    cl::Kernel kernel2;
    this->kernels.push_back(kernel1);
    this->kernels.push_back(kernel2);
    cl::Event event;
    this->events.push_back(event);

    return 0;
}

template <class Dtype>
int InputLayerCL<Dtype>::SetWorkSize() {
    Blob<Dtype>* layer_blob = this->_top_blobs[this->_top[0]];
    size_t output_channels = layer_blob->channels();

    if (this->output_height >= 32) this->group_size_h = 16;
    if (this->output_width >= 32) this->group_size_w = 16;

    //layer_blob->_groupChannel = 4; //HWC4
    this->global_work_size[0] = (this->output_height / this->group_size_h + !!(this->output_height % this->group_size_h)) * this->group_size_h;
    this->global_work_size[1] = (this->output_width  / this->group_size_w + !!(this->output_width  % this->group_size_w)) * this->group_size_w;
    this->global_work_size[2] = output_channels / layer_blob->channel_grp() + !!(output_channels % layer_blob->channel_grp());

    this->local_work_size[0] = this->group_size_h;
    this->local_work_size[1] = this->group_size_w;
    if (this->global_work_size[2] > layer_blob->channel_grp() && this->global_work_size[2] % layer_blob->channel_grp() == 0) {
      this->local_work_size[2] = 4;
    } else {
      this->local_work_size[2] = 1;
    }

    return 0;
}

template <class Dtype>
int InputLayerCL<Dtype>::ResetInputAndArgs(size_t data_size) {
    bool flag = false;
    if (data_size > this->input_data_size)
    {
        cl_int error_num;

        this->_cl_fimage = cl::Buffer(this->rt_param->context(),
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  data_size * sizeof(float), nullptr, &error_num);
        if (!checkSuccess(error_num)) {
          LOGE("Failed to create OpenCL buffers[%d]", error_num);
          return -1;
        }

        flag = true;

    }
    this->input_data_size = data_size;
    return flag ? 2 : 0;

}

template <class Dtype>
int InputLayerCL<Dtype>::SetBuildOptions() {
    Blob<Dtype>* layer_blob = this->_top_blobs[this->_top[0]];
    size_t input_channels = layer_blob->channels();
    std::ostringstream ss;
    ss << input_channels;
    this->build_options.push_back("-DINPUT_CHANNELS=" + ss.str());
    if(std::is_same<Dtype, uint16_t>::value)
      this->build_options.push_back("-DDATA_TYPE=half");
    else
      this->build_options.push_back("-DDATA_TYPE=float");

    return 0;
}

template <class Dtype>
int InputLayerCL<Dtype>::SetKernelParameters() {
  int error_num;
  size_t data_size;
  bool set_kernel_arguments_success = true;
  int param_idx = 0;
  Blob<Dtype>* layer_blob = this->_top_blobs[this->_top[0]];


  SetWorkSize();

  this->kernels[0] = cl::Kernel(this->cl_programs[0], this->cl_kernel_functions[0].c_str(), &error_num);
  if (!checkSuccess(error_num)) {
    LOGE("Failed to create normalinit OpenCL kernels[0]. ");
    return -1;
  }
  this->kernels[1] = cl::Kernel(this->cl_programs[1], this->cl_kernel_functions[1].c_str(), &error_num);
  if (!checkSuccess(error_num)) {
    LOGE("Failed to create normalinit OpenCL kernels[1]. ");
    return -1;
  }

  data_size = layer_blob->data_size_padded_c();
  layer_blob->AllocDevice(this->rt_param->context(), data_size);

  data_size = layer_blob->data_size();
  this->_cl_fimage = cl::Buffer(this->rt_param->context(),
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  data_size * sizeof(float), nullptr, &error_num);
  if (!checkSuccess(error_num)) {
    LOGE("Failed to create OpenCL buffers[%d]", error_num);
    return -1;
  }

  this->input_data_size = data_size;

  cl::Buffer* layer_data_cl = layer_blob->data_cl();
  set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, this->_cl_fimage));
  set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, *layer_data_cl));
  set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, output_height));
  set_kernel_arguments_success &= checkSuccess(this->kernels[0].setArg(param_idx++, output_width));

  if (!set_kernel_arguments_success) {
    LOGE("Failed setting normalinit OpenCL kernels[0] arguments. %s: %s", __FILE__, __LINE__);
    return -1;
  }

  cl::ImageFormat img_format(CL_RGBA, CL_UNORM_INT8);
  _cl_img2d = cl::Image2D(this->rt_param->context(),
                      CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, img_format,
                      output_width, output_height, 0, nullptr, &error_num);

  if (error_num != CL_SUCCESS) {
    LOGE("Failed to create OpenCL Image2D[%d]. %s: %s", __FILE__, __LINE__, error_num);
    return -1;
  }

  param_idx = 0;
  set_kernel_arguments_success &= checkSuccess(this->kernels[1].setArg(param_idx++, this->_cl_img2d));
  set_kernel_arguments_success &= checkSuccess(this->kernels[1].setArg(param_idx++, *layer_data_cl));
  set_kernel_arguments_success &= checkSuccess(this->kernels[1].setArg(param_idx++, output_height));
  set_kernel_arguments_success &= checkSuccess(this->kernels[1].setArg(param_idx++, output_width));
  if (!set_kernel_arguments_success) {
    LOGE("Failed setting normalinit OpenCL kernels[1] arguments. %s: %s", __FILE__, __LINE__);
    return -1;
  }
  this->FineTuneGroupSize(this->kernels[0], this->output_height, this->output_width);
  this->FineTuneGroupSize(this->kernels[1], this->output_height, this->output_width);
  return 0;
}

template <class Dtype>
int InputLayerCL<Dtype>::FloatToDevice(const float* input_data) {
#ifdef TIMING_CL
  this->rt_param->command_queue().finish();
  timespec tpstart, tpend;
  clock_gettime(CLOCK_MONOTONIC, &tpstart);

  cl_int error_num;
  float* map_data =
    (float*)this->rt_param->command_queue().enqueueMapBuffer(this->_cl_fimage, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                           0, this->input_data_size * sizeof(float), nullptr, nullptr, &error_num);
  if (!checkSuccess(error_num)) {
    LOGE("fatal error: WriteBuffer Mapping memory objects failed [%d].  %s: %d", error_num, __FILE__, __LINE__);
    return -1;
  }

  memcpy(map_data, input_data, this->input_data_size * sizeof(float));

  error_num = this->rt_param->command_queue().enqueueUnmapMemObject(this->_cl_fimage, map_data,
                                           nullptr, nullptr);
  if (!checkSuccess(error_num)){
    LOGE("fatal error: Unmapping memory objects failed. %s: %s", __FILE__, __LINE__);
  }

  this->rt_param->command_queue().finish();
  clock_gettime(CLOCK_MONOTONIC, &tpend);
  double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
  LOGI("[%s] FloatToDevice Execution time in %lf ms\n", this->name().c_str(), timedif / 1000.0);

#else
  cl_int error_num;
  float* map_data =
    (float*)this->rt_param->command_queue().enqueueMapBuffer(this->_cl_fimage, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                           0, this->input_data_size * sizeof(float), nullptr, nullptr, &error_num);
  if (!checkSuccess(error_num)) {
    LOGE("fatal error: WriteBuffer Mapping memory objects failed [%d].  %s: %d", error_num, __FILE__, __LINE__);
    return -1;
  }

  memcpy(map_data, input_data, this->input_data_size * sizeof(float));

  error_num = this->rt_param->command_queue().enqueueUnmapMemObject(this->_cl_fimage, map_data,
                                           nullptr, nullptr);
  if (!checkSuccess(error_num)){
    LOGE("fatal error: Unmapping memory objects failed. %s: %s", __FILE__, __LINE__);
  }

#endif
  return 0;
}

template <class Dtype>
int InputLayerCL<Dtype>::UintToDevice(const uint8_t* src_bgra) {
#ifdef TIMING_CL
  this->rt_param->command_queue().finish();
  timespec tpstart, tpend;
  clock_gettime(CLOCK_MONOTONIC, &tpstart);

  int error_num;
  std::vector<size_t> mapped_image_pitch(2);
  std::array<size_t, 3> origin = {0, 0, 0};
  std::array<size_t, 3> region = { static_cast<size_t>(this->output_width), static_cast<size_t>(this->output_height), 1 };
  uint8_t* map_data = reinterpret_cast<uint8_t*>(this->rt_param->command_queue().enqueueMapImage(
      this->_cl_img2d, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, origin, region,
      mapped_image_pitch.data(), mapped_image_pitch.data() + 1, nullptr,
      nullptr, &error_num));
  if (!checkSuccess(error_num)) {
    LOGE("fatal error: mapping _cl_img2d objects failed. %s: %d", error_num, __FILE__, __LINE__);
  }

  memcpy(map_data, src_bgra, this->input_data_size);

  error_num = this->rt_param->command_queue().enqueueUnmapMemObject(this->_cl_img2d, map_data, nullptr, nullptr);
  if (!checkSuccess(error_num)) {
    LOGE("fatal error: Deconstructor Unmapping _cl_img2d objects failed.");
  }

  this->rt_param->command_queue().finish();
  clock_gettime(CLOCK_MONOTONIC, &tpend);
  double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
  LOGI("[%s] UintToDevice Execution time in %lf ms\n", this->name().c_str(), timedif / 1000.0);

#else
  int error_num;
  std::vector<size_t> mapped_image_pitch(2);
  std::array<size_t, 3> origin = {0, 0, 0};
  std::array<size_t, 3> region = { static_cast<size_t>(this->output_width), static_cast<size_t>(this->output_height), 1 };
  uint8_t* map_data = reinterpret_cast<uint8_t*>(this->rt_param->command_queue().enqueueMapImage(
      this->_cl_img2d, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, origin, region,
      mapped_image_pitch.data(), mapped_image_pitch.data() + 1, nullptr,
      nullptr, &error_num));
  if (!checkSuccess(error_num)) {
    LOGE("fatal error: mapping _cl_img2d objects failed. %s: %d", error_num, __FILE__, __LINE__);
  }

  memcpy(map_data, src_bgra, this->input_data_size);

  error_num = this->rt_param->command_queue().enqueueUnmapMemObject(this->_cl_img2d, map_data, nullptr, nullptr);
  if (!checkSuccess(error_num)) {
    LOGE("fatal error: Deconstructor Unmapping _cl_img2d objects failed.");
  }

#endif
  return 0;
}

template <class Dtype>
int InputLayerCL<Dtype>::RunKernel(int type) {
#ifdef TIMING_CL
    this->rt_param->command_queue().finish();
    timespec tpstart, tpend;
    clock_gettime(CLOCK_MONOTONIC, &tpstart);

  // int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
  //       kernels[type], cl::NullRange, cl::NDRange(global_work_size[0], global_work_size[1], global_work_size[2]),
  //       cl::NDRange(local_work_size[0], local_work_size[1], local_work_size[2]), nullptr, &events[0]);
  int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
        this->kernels[type], cl::NullRange, cl::NDRange(this->global_work_size[0], this->global_work_size[1], this->global_work_size[2]),
        cl::NDRange(this->local_work_size[0], this->local_work_size[1], this->local_work_size[2]), nullptr, &this->events[0]);

  if (!checkSuccess(error_num)) {
    LOGE("Failed enqueuing the normalinit kernel.");
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
        this->kernels[type], cl::NullRange, cl::NDRange(this->global_work_size[0], this->global_work_size[1], this->global_work_size[2]),
        cl::NDRange(this->local_work_size[0], this->local_work_size[1], this->local_work_size[2]), nullptr, nullptr);
  if (!checkSuccess(error_num)) {
    LOGE("Failed enqueuing the normalinit kernel.");
    return -1;
  }

#endif

    return 0;
}

template class InputLayerCL<float>;
template class InputLayerCL<uint16_t>;

}; // namespace feather
