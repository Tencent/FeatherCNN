#include "input_layer_cl.h"

namespace feather {

int InputLayerCL::InitCL()
{
    std::string kernel_name1 = "clNormalInit";
    std::string func_name1  = "init1O4";
    auto it_source1 = booster::opencl_kernel_string_map.find("inputBufferFloat");
    std::string kernel_str1(it_source1->second.begin(),it_source1->second.end());
    std::string func_name2 = "init1O42D";
    std::string kernel_name2 = "clNormalInit2D";
    auto it_source2 = booster::opencl_kernel_string_map.find("inputImage");
    std::string kernel_str2(it_source2->second.begin(),it_source2->second.end());

    Blob<uint16_t>* layer_blob = this->_top_blobs[this->_top[0]];
    size_t channels = layer_blob->channels();
    if (channels < 1 || channels > 4) {
        LOGE("unsupported channel size: %d", channels);
        return -1;
    }
    std::ostringstream ss;
    ss << channels;
    this->build_options.push_back("-DINPUT_CHANNELS=" + ss.str());


    this->cl_kernel_functions.push_back(func_name1);
    this->cl_kernel_functions.push_back(func_name2);
    this->cl_kernel_names.push_back(kernel_name1);
    this->cl_kernel_names.push_back(kernel_name2);
    this->cl_kernel_symbols.push_back(kernel_str1);
    this->cl_kernel_symbols.push_back(kernel_str2);

    cl_kernel kernel1;
    cl_kernel kernel2;
    this->kernels.push_back(kernel1);
    this->kernels.push_back(kernel2);
    cl_event event;
    this->events.push_back(event);

    return 0;
}

int InputLayerCL::SetWorkSize() {
    Blob<uint16_t>* layer_blob = this->_top_blobs[this->_top[0]];
    size_t output_channels = layer_blob->channels();

    if (this->output_height >= 32) this->group_size_h = 16;
    if (this->output_width >= 32) this->group_size_w = 16;

    //layer_blob->_groupChannel = 4; //HWC4
    this->global_work_size[0] = (this->output_height / this->group_size_h + !!(this->output_height % this->group_size_h)) * this->group_size_h;
    this->global_work_size[1] = (this->output_width  / this->group_size_w + !!(this->output_width  % this->group_size_w)) * this->group_size_w;
    this->global_work_size[2] = output_channels / layer_blob->channel_grp() + !!(output_channels % layer_blob->channel_grp());

    this->local_work_size[0] = group_size_h;
    this->local_work_size[1] = group_size_w;
    if (this->global_work_size[2] > layer_blob->channel_grp() && this->global_work_size[2] % layer_blob->channel_grp() == 0) {
      this->local_work_size[2] = 4;
    } else {
      this->local_work_size[2] = 1;
    }

    return 0;
}

int InputLayerCL::ResetInputAndArgs(size_t data_size) {
    bool flag = false;
    if (data_size > this->input_data_size)
    {
        int error_num;
        error_num = clEnqueueUnmapMemObject(this->rt_param->command_queue(), this->_cl_fimage, this->_map_fdata, 0, NULL, NULL);
        if (!checkSuccess(error_num)) {
          LOGE("fatal error: Deconstructor Unmapping _map_fdata objects failed.");
          return 1;
        }

        error_num = clReleaseMemObject(this->_cl_fimage);
        if (!checkSuccess(error_num))
        {
            LOGE("Failed to release mem object. %s: %s", __FILE__, __LINE__);
            return 1;
        }

        this->_cl_fimage = clCreateBuffer(this->rt_param->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        data_size * sizeof(float), NULL, &error_num);
        if (!checkSuccess(error_num)) {
          LOGE("Failed to create OpenCL buffers[%d]", error_num);
          return -1;
        }

        this->_map_fdata = (float*)clEnqueueMapBuffer(this->rt_param->command_queue(), this->_cl_fimage, CL_TRUE, CL_MAP_WRITE,
        0, data_size * sizeof(float), 0, NULL, NULL, &error_num);
        if (!checkSuccess(error_num)) {
          LOGE("fatal error: WriteBuffer Mapping memory objects failed [%d].  %s: %s", error_num, __FILE__, __LINE__);
          return -1;
        }
        flag = true;

    }
    this->input_data_size = data_size;
    return flag ? 2 : 0;

}

int InputLayerCL::SetKernelParameters() {

    int error_num;
    size_t data_size;
    bool set_kernel_arguments_success = true;
    int param_idx = 0;
    Blob<uint16_t>* layer_blob = this->_top_blobs[this->_top[0]];

    SetWorkSize();

    this->kernels[0] = clCreateKernel(this->cl_programs[0], this->cl_kernel_functions[0].c_str(), &error_num);
    if (!checkSuccess(error_num)) {
      LOGE("Failed to create normalinit OpenCL kernels[0]. ");
      return -1;
    }
    this->kernels[1] = clCreateKernel(this->cl_programs[1], this->cl_kernel_functions[1].c_str(), &error_num);
    if (!checkSuccess(error_num)) {
      LOGE("Failed to create normalinit OpenCL kernels[1]. ");
      return -1;
    }

    data_size = layer_blob->data_size_padded_c();
    layer_blob->AllocDevice(this->rt_param->context(), data_size);

    data_size = layer_blob->data_size();
    this->_cl_fimage = clCreateBuffer(this->rt_param->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                      data_size * sizeof(float), NULL, &error_num);
    if (!checkSuccess(error_num)) {
      LOGE("Failed to create OpenCL buffers[%d]", error_num);
      return -1;
    }

    this->_map_fdata = (float*)clEnqueueMapBuffer(this->rt_param->command_queue(), this->_cl_fimage, CL_TRUE, CL_MAP_WRITE,
                                                  0, data_size * sizeof(float), 0, NULL, NULL, &error_num);
    if (!checkSuccess(error_num)) {
      LOGE("fatal error: WriteBuffer Mapping memory objects failed [%d].  %s: %s", error_num, __FILE__, __LINE__);
      return -1;
    }
    this->input_data_size = data_size;

    cl_mem layer_data_cl = layer_blob->data_cl();

    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_mem), &this->_cl_fimage));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_mem), &layer_data_cl));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->output_height));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &this->output_width));


    if (!set_kernel_arguments_success) {
      LOGE("Failed setting normalinit OpenCL kernels[0] arguments. %s: %s", __FILE__, __LINE__);
      return -1;
    }

    cl_image_format cl_img_fmt;
    cl_img_fmt.image_channel_order = CL_RGBA;
    cl_img_fmt.image_channel_data_type = CL_UNORM_INT8;
    _cl_img2d = clCreateImage2D(this->rt_param->context(),
                                    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                    &cl_img_fmt,
                                    this->output_width,
                                    this->output_height,
                                    0,
                                    NULL,
                                    &error_num);
    if (error_num != CL_SUCCESS) {
      LOGE("Failed to create OpenCL Image2D[%d]. %s: %s", __FILE__, __LINE__, error_num);
      return -1;
    }

    size_t row_pitch;
    const size_t origin[3] = { 0, 0, 0 };
    const size_t region[3] = { static_cast<size_t>(output_width), static_cast<size_t>(output_height), 1 };
    this->_map_img = reinterpret_cast<uint8_t*>(clEnqueueMapImage(this->rt_param->command_queue(), this->_cl_img2d, CL_TRUE,
                                                                  CL_MAP_WRITE, origin, region, &row_pitch,
                                                                  NULL, 0, NULL, NULL, &error_num));
    if (!checkSuccess(error_num)) {
      LOGE("fatal error: mapping _cl_img2d objects failed. %s: %s", error_num, __FILE__, __LINE__);
    }

    param_idx = 0;
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[1], param_idx++, sizeof(cl_mem), &this->_cl_img2d));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[1], param_idx++, sizeof(cl_mem), &layer_data_cl));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[1], param_idx++, sizeof(cl_int), &this->output_height));
    set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[1], param_idx++, sizeof(cl_int), &this->output_width));
    if (!set_kernel_arguments_success) {
      LOGE("Failed setting normalinit OpenCL kernels[1] arguments. %s: %s", __FILE__, __LINE__);
      return -1;
    }

    FineTuneGroupSize(this->kernels[0], this->output_height, this->output_width);
    FineTuneGroupSize(this->kernels[1], this->output_height, this->output_width);

    return 0;
}

int InputLayerCL::FloatToDevice(const float* input_data) {
  // Blob<uint16_t>* layer_blob = this->_top_blobs[this->_top[0]];
  // size_t data_size = layer_blob->data_size();
  memcpy(this->_map_fdata, input_data, this->input_data_size * sizeof(float));
  return 0;
}

int InputLayerCL::UintToDevice(const uint8_t* src_bgra) {
  // Blob<uint16_t>* layer_blob = this->_top_blobs[this->_top[0]];
  // size_t data_size = layer_blob->data_size();
  memcpy(this->_map_img, src_bgra, this->input_data_size);
  return 0;
}

int InputLayerCL::RunKernel(int type) {
#ifdef TIMING_CL
    clFinish(commandQueue);
    timespec tpstart, tpend;
    clock_gettime(CLOCK_MONOTONIC, &tpstart);

    int error_num = clEnqueueNDRangeKernel(this->rt_param->command_queue(), kernels[type], 3,
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
    int error_num = clEnqueueNDRangeKernel(this->rt_param->command_queue(), kernels[type], 3,
                    NULL, this->global_work_size, this->local_work_size, 0, NULL, NULL);
    if (!checkSuccess(error_num)) {
      LOGE("Failed enqueuing the inner product kernel. %s", errorNumberToString(error_num).c_str());
      return -1;
    }
#endif

    return 0;
}
}; // namespace feather
