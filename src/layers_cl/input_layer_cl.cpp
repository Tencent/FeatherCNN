
#include "cl_input_layer.h"

namespace feather {
int CLInputLayer::InitCl() {
  string kernel_name1 = "clNormalInit";
  string func_name1  = "init1O4";
  auto it_source1 = kernel_string_map.find("inputBufferFloat");
  std::string kernel_str1(it_source1->second.begin(),it_source1->second.end());

  string func_name2 = "init1O42D";
  string kernel_name2 = "clNormalInit2D";
  auto it_source2 = kernel_string_map.find("inputImage");
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

int CLInputLayer::build() {
  Blob<float>* layer_blob = this->_top_blobs[this->_top[0]];
  size_t output_height = layer_blob->height();
  size_t output_width = layer_blob->width();
  size_t output_channels = layer_blob->channels();

  if (output_height >= 32) groupSizeH = 16;
  if (output_width >= 32) groupSizeW = 16;

  layer_blob->_groupChannel = 4; //HWC4
  this->global_work_size[0] = (output_height / group_size_h + !!(output_height % group_size_h)) * groupSizeH;
  this->global_work_size[1] = (output_width  / group_size_w + !!(output_width  % group_size_w)) * groupSizeW;
  this->global_work_size[2] = output_channels / layer_blob->_groupChannel + !!(output_channels % layer_blob->_groupChannel);

  this->local_work_size[0] = group_size_h;
  this->local_work_size[1] = group_size_w;
  if (this->global_work_size[2] > layer_blob->_groupChannel && this->global_work_size[2] % layer_blob->_groupChannel == 0) {
    this->local_work_size[2] = 4;
  } else {
    this->local_work_size[2] = 1;
  }

  int error_num;
  size_t mem_size;

  this->kernels[0] = clCreateKernel(programs[0], this->cl_kernel_functions[0].c_str(), &error_num);
  if (!checkSuccess(error_num)) {
    LOGE("Failed to create normalinit OpenCL kernels[0]. ");
    return -1;
  }
  this->kernels[1] = clCreateKernel(programs[1], this->cl_kernel_functions[1].c_str(), &error_num);
  if (!checkSuccess(error_num)) {
    LOGE("Failed to create normalinit OpenCL kernels[1]. ");
    return -1;
  }

  mem_size = output_height * output_width * layer_blob->get_channels_padding();
  // layer_blob->_data_cl = clCreateBuffer(this->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
  //                                 mem_size * sizeof(half), NULL, &error_num);
  // if (!checkSuccess(error_num)) {
  //     LOGE("Failed to create OpenCL buffers[%d]", error_num);
  //     exit(-1);
  // }
  layer_blob->Alloc_cl(this->context, mem_size);

  mem_size = layer_blob->data_size();
  this->_cl_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                   mem_size * sizeof(unsigned short), NULL, &error_num);
  if (!checkSuccess(error_num)) {
    LOGE("Failed to create OpenCL buffers[%d]", error_num);
    exit(-1);
  }

  this->_cl_fimage = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                    mem_size * sizeof(float), NULL, &error_num);
  if (!checkSuccess(error_num)) {
    LOGE("Failed to create OpenCL buffers[%d]", error_num);
    exit(-1);
  }

  bool set_kernel_arguments_success = true;
  cl_mem layer_data_cl = layer_blob->data_cl();
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], 0, sizeof(cl_mem), &this->_cl_fimage));
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], 1, sizeof(cl_mem), &layer_data_cl));
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], 2, sizeof(cl_int), &output_height));
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], 3, sizeof(cl_int), &output_width));
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[0], 4, sizeof(cl_int), &output_channels));

  if (!set_kernel_arguments_success) {
    LOGE("Failed setting normalinit OpenCL kernels[0] arguments.");
    return -1;
  }
  fineTuneGroupSize(this->kernels[0], layer_blob->height(), layer_blob->width());

  cl_image_format cl_img_fmt;
  cl_img_fmt.image_channel_order = CL_RGBA;
  cl_img_fmt.image_channel_data_type = CL_UNORM_INT8;
  _cl_img2d = clCreateImage2D(this->context,
                                  CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                  &cl_img_fmt,
                                  output_width,
                                  output_height,
                                  0,
                                  NULL,
                                  &error_num);
  if (error_num != CL_SUCCESS) {
    LOGE("Failed to create OpenCL Image2D[%d]", error_num);
    return -1;
  }

  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[1], 0, sizeof(cl_mem), &this->_cl_img2d));
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[1], 1, sizeof(cl_mem), &layer_data_cl));
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[1], 2, sizeof(cl_int), &output_height));
  set_kernel_arguments_success &= checkSuccess(clSetKernelArg(kernels[1], 3, sizeof(cl_int), &output_width));
  // setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernels[1], 4, sizeof(cl_int), &output_channels ));
  if (!set_kernel_arguments_success) {
    LOGE("Failed setting normalinit OpenCL kernels[1] arguments. ");
    return -1;
  }

  fineTuneGroupSize(this->kernels[1], layer_blob->height(), layer_blob->width());

  this->_map_data = (half*)clEnqueueMapBuffer(this->commandQueue, this->_cl_image, CL_TRUE, CL_MAP_WRITE,
                                              0, mem_size * sizeof(half), 0, NULL, NULL, &error_num);
  if (!checkSuccess(error_num)) {
    LOGE("fatal error: WriteBuffer Mapping memory objects failed. %d", error_num);
    return -1;
  }

  this->_map_fdata = (float*)clEnqueueMapBuffer(this->commandQueue, this->_cl_fimage, CL_TRUE, CL_MAP_WRITE,
                                                0, mem_size * sizeof(float), 0, NULL, NULL, &error_num);
  if (!checkSuccess(error_num)) {
    LOGE("fatal error: WriteBuffer Mapping memory objects failed. %d", error_num);
    return -1;
  }

  size_t row_pitch;
  const size_t origin[3] = { 0, 0, 0 };
  const size_t region[3] = { static_cast<size_t>(output_width), static_cast<size_t>(output_height), 1 };
  this->_map_img = reinterpret_cast<uint8_t*>(clEnqueueMapImage(this->commandQueue,
                                                                this->_cl_img2d,
                                                                CL_TRUE,
                                                                CL_MAP_WRITE,
                                                                origin,
                                                                region,
                                                                &row_pitch,
                                                                NULL,
                                                                0,
                                                                NULL,
                                                                NULL,
                                                                &error_num));
  if (!checkSuccess(error_num)) {
    LOGE("fatal error: mapping _cl_img2d objects failed.");
  }

  return 0;
}

int CLInputLayer::FloatToDevice() {
  Blob<float>* layer_blob = this->_top_blobs[this->_top[0]];
  size_t output_height = layer_blob->height();
  size_t output_width = layer_blob->width();
  size_t output_channels = layer_blob->channels();

  int error_num;
  size_t mem_size;
  mem_size = layer_blob->data_size();

  // for (int i = 0; i < output_height * output_width; ++i) {
  //   for (int j = 0; j < output_channels; ++j) {
  //     _map_data[i * output_channels + j] = hs_floatToHalf(this->_input_data[j * output_height * output_width + i]);
  //   }
  // }
  memcpy(_map_fdata, _input_data, mem_size * sizeof(float));

  return 0;
}

int CLInputLayer::UintToDevice() {
  Blob<float>* layer_blob = this->_top_blobs[this->_top[0]];
  size_t output_height = layer_blob->height();
  size_t output_width = layer_blob->width();
  size_t output_channels = layer_blob->channels();
  // size_t row_pitch;
  // int error_num;
  // const size_t origin[3] = { 0, 0, 0 };
  // const size_t region[3] = { static_cast<size_t>(output_width), static_cast<size_t>(output_height), 1 };
  // this->_map_img = reinterpret_cast<uint8_t*>(clEnqueueMapImage(this->commandQueue,
  //                                                                      this->_cl_img2d,
  //                                                                      CL_TRUE,
  //                                                                      CL_MAP_WRITE,
  //                                                                      origin,
  //                                                                      region,
  //                                                                      &row_pitch,
  //                                                                      NULL,
  //                                                                      0,
  //                                                                      NULL,
  //                                                                      NULL,
  //                                                                      &error_num));
  //
  // if (!checkSuccess(error_num)) {
  //   LOGE("fatal error: mapping _cl_img2d objects failed.");
  // }

  //clEnqueueWriteImage(this->commandQueue, this->_cl_img2d, CL_TRUE, origin, region, 0, 0, _input_image, 0, 0, 0);

  memcpy(this->_map_img, _input_image, output_height * output_width * output_channels);

  // error_num = clEnqueueUnmapMemObject(this->commandQueue, this->_cl_img2d, this->_map_img, 0, NULL, NULL);
  // if (!checkSuccess(error_num)) {
  //   LOGE("fatal error: Deconstructor Unmapping _cl_img2d objects failed.");
  // }

  return 0;
}

int CLInputLayer::RunKernel(int type) {
  int error_num = clEnqueueNDRangeKernel(commandQueue, kernels[type], 3, NULL,
                                         globalWorkSize, localWorkSize, 0, NULL, &events[0]);
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
