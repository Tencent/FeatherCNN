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


int InputLayerCL::SetKernelParameters() {

  int error_num;
  size_t data_size;
  bool set_kernel_arguments_success = true;
  int param_idx = 0;
  Blob<uint16_t>* layer_blob = this->_top_blobs[this->_top[0]];
  size_t output_height = layer_blob->height();
  size_t output_width = layer_blob->width();
  size_t output_channels = layer_blob->channels();

  if (output_height >= 32) group_size_h = 16;
  if (output_width >= 32) group_size_w = 16;

  //layer_blob->_groupChannel = 4; //HWC4
  this->global_work_size[0] = (output_height / group_size_h + !!(output_height % group_size_h)) * group_size_h;
  this->global_work_size[1] = (output_width  / group_size_w + !!(output_width  % group_size_w)) * group_size_w;
  this->global_work_size[2] = output_channels / layer_blob->channel_grp() + !!(output_channels % layer_blob->channel_grp());

  this->local_work_size[0] = group_size_h;
  this->local_work_size[1] = group_size_w;
  if (this->global_work_size[2] > layer_blob->channel_grp() && this->global_work_size[2] % layer_blob->channel_grp() == 0) {
    this->local_work_size[2] = 4;
  } else {
    this->local_work_size[2] = 1;
  }


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

  cl_mem layer_data_cl = layer_blob->data_cl();

  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_mem), &this->_cl_fimage));
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_mem), &layer_data_cl));
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &output_height));
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &output_width));
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], param_idx++, sizeof(cl_int), &output_channels));

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
                                  output_width,
                                  output_height,
                                  0,
                                  NULL,
                                  &error_num);
  if (error_num != CL_SUCCESS) {
    LOGE("Failed to create OpenCL Image2D[%d]. %s: %s", __FILE__, __LINE__, error_num);
    return -1;
  }

  param_idx = 0;
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[1], param_idx++, sizeof(cl_mem), &this->_cl_img2d));
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[1], param_idx++, sizeof(cl_mem), &layer_data_cl));
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[1], param_idx++, sizeof(cl_int), &output_height));
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[1], param_idx++, sizeof(cl_int), &output_width));
  if (!set_kernel_arguments_success) {
    LOGE("Failed setting normalinit OpenCL kernels[1] arguments. %s: %s", __FILE__, __LINE__);
    return -1;
  }

  this->_map_fdata = (float*)clEnqueueMapBuffer(this->rt_param->command_queue(), this->_cl_fimage, CL_TRUE, CL_MAP_WRITE,
                                                0, data_size * sizeof(float), 0, NULL, NULL, &error_num);
  if (!checkSuccess(error_num)) {
    LOGE("fatal error: WriteBuffer Mapping memory objects failed [%d].  %s: %s", error_num, __FILE__, __LINE__);
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


  FineTuneGroupSize(this->kernels[0], layer_blob->height(), layer_blob->width());
  FineTuneGroupSize(this->kernels[1], layer_blob->height(), layer_blob->width());

  return 0;
}

int InputLayerCL::FloatToDevice(const float* input_data) {
  Blob<uint16_t>* layer_blob = this->_top_blobs[this->_top[0]];
  size_t data_size = layer_blob->data_size();

  memcpy(this->_map_fdata, input_data, data_size * sizeof(float));

  return 0;
}

int InputLayerCL::UintToDevice(const uint8_t* src_bgra) {
  Blob<uint16_t>* layer_blob = this->_top_blobs[this->_top[0]];
  size_t data_size = layer_blob->data_size();

  memcpy(this->_map_img, src_bgra, data_size);

  return 0;
}

int InputLayerCL::RunKernel(int type) {
  int error_num = clEnqueueNDRangeKernel(this->rt_param->command_queue(), kernels[type], 3, NULL,
                                         global_work_size, local_work_size, 0, NULL, &events[0]);
  if (!checkSuccess(error_num)) {
    LOGE("Failed enqueuing the normalinit kernel.");
    return -1;
  }

  /* if we wanna do something for event in future */
  error_num = clReleaseEvent(events[0]);
  if (!checkSuccess(error_num)) {
    LOGE("Failed release event.");
    return -1;
  }

  return 0;
}
}; // namespace feather
