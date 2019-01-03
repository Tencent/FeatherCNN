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
#include "input_layer_cl.h"

namespace feather
{

template <class Dtype>
InputLayerCL<Dtype>::InputLayerCL(const LayerParameter *layer_param, RuntimeParameter<Dtype>* rt_param): Layer<Dtype>(layer_param, rt_param), _cl_fimage(NULL), _cl_img2d(NULL)
{
    //From proto
    const InputParameter *input_param = layer_param->input_param();
    size_t input_num = VectorLength(input_param->name());
    size_t input_dim_num = VectorLength(input_param->dim());
    assert(input_num > 0);
    assert(input_dim_num == input_num * 4);
    for (int i = 0; i < input_num; ++i)
    {
        size_t num = input_param->dim()->Get(i * 4);
        size_t channels = input_param->dim()->Get(i * 4 + 1);
        size_t height = input_param->dim()->Get(i * 4 + 2);
        size_t width = input_param->dim()->Get(i * 4 + 3);

        std::string input_name = input_param->name()->Get(i)->str();
        this->_top.push_back(input_name);
        this->_top_blobs[input_name] = new Blob<Dtype>(num, channels, height, width);


        this->output_height = height;
        this->output_width = width;
        this->input_channels = channels;

        //_top_blobs[input_name]->PrintBlobInfo();
        LOGI("input_name cl %s (n c h w)=(%ld %ld %ld %ld)\n", input_name.c_str(), num, channels, height, width);
        this->InitCL();
        this->SetWorkSize();
    }
}

template <class Dtype>
int InputLayerCL<Dtype>::InitCL()
{

    std::string program_name_float = "input_buffer";
    std::string kernel_name_float = "float_chw_to_hwc";
    auto it_source_float = booster::opencl_kernel_string_map.find(program_name_float);
    if (it_source_float != booster::opencl_kernel_string_map.end())
    {
        this->cl_kernel_info_map[kernel_name_float].program_name = program_name_float;
        this->cl_kernel_info_map[kernel_name_float].kernel_name = kernel_name_float;
        this->cl_kernel_info_map[kernel_name_float].kernel_source = std::string(it_source_float->second.begin(), it_source_float->second.end());
    }
    else
    {
        LOGE("can't find program %s!", program_name_float.c_str());
        return -1;
    }

    std::string program_name_uint8 = "input_image";
    std::string kernel_name_uint8 = "uint8_hwc_to_hwc";
    auto it_source_uint8 = booster::opencl_kernel_string_map.find(program_name_uint8);
    if (it_source_uint8 != booster::opencl_kernel_string_map.end())
    {
        this->cl_kernel_info_map[kernel_name_uint8].program_name = program_name_uint8;
        this->cl_kernel_info_map[kernel_name_uint8].kernel_name = kernel_name_uint8;
        this->cl_kernel_info_map[kernel_name_uint8].kernel_source = std::string(it_source_uint8->second.begin(), it_source_uint8->second.end());
    }
    else
    {
        LOGE("can't find program %s!", program_name_uint8.c_str());
        return -1;
    }

    return 0;
}

template <class Dtype>
int InputLayerCL<Dtype>::SetWorkSize()
{
    clhpp_feather::CLKernelInfo& float_kernel_info = this->cl_kernel_info_map["float_chw_to_hwc"];
    std::vector<size_t>& float_gws = float_kernel_info.gws;
    std::vector<size_t>& float_lws = float_kernel_info.lws;

    clhpp_feather::CLKernelInfo& uint8_kernel_info = this->cl_kernel_info_map["uint8_hwc_to_hwc"];
    std::vector<size_t>& uint8_gws = uint8_kernel_info.gws;
    std::vector<size_t>& uint8_lws = uint8_kernel_info.lws;

    size_t padded_output_c = this->_top_blobs[this->_top[0]]->get_channels_padding();

    if (float_gws.size() != 0 || float_gws.size() != 0)
    {
        float_gws.clear();
        float_gws.clear();
    }
    int h_lws = this->output_height > 32 ? 16 : 8;
    int w_lws = this->output_width > 32 ? 16 : 8;

    int c_blk_size = 4;
    if (padded_output_c % 16 == 0)
    {
        c_blk_size = 16;
    }
    else if (padded_output_c % 8 == 0)
    {
        c_blk_size = 8;
    }
    this->channel_grp_size = c_blk_size;

    size_t float_gws_dim0 = (this->output_height / h_lws + !!(this->output_height % h_lws)) * h_lws;
    size_t float_gws_dim1 = (this->output_width / w_lws  + !!(this->output_width % w_lws)) * w_lws;
    size_t float_gws_dim2 = padded_output_c / c_blk_size;

    size_t float_lws_dim0 = h_lws;
    size_t float_lws_dim1 = w_lws;
    size_t float_lws_dim2 = (float_gws_dim2 > 4 && float_gws_dim2 % 4 == 0) ? 4 : 1;

    float_gws.push_back(float_gws_dim0);
    float_gws.push_back(float_gws_dim1);
    float_gws.push_back(float_gws_dim2);
    float_lws.push_back(float_lws_dim0);
    float_lws.push_back(float_lws_dim1);
    float_lws.push_back(float_lws_dim2);
    uint8_gws.assign(float_gws.begin(), float_gws.end());
    uint8_lws.assign(float_lws.begin(), float_lws.end());

    return 0;
}
template <class Dtype>
int InputLayerCL<Dtype>::ResetWorkSizeFloat()
{
    clhpp_feather::CLKernelInfo& float_kernel_info = this->cl_kernel_info_map["float_chw_to_hwc"];
    std::vector<size_t>& float_gws = float_kernel_info.gws;
    std::vector<size_t>& float_lws = float_kernel_info.lws;

    int h_lws = this->output_height > 32 ? 16 : 8;
    int w_lws = this->output_width > 32 ? 16 : 8;

    size_t float_gws_dim0 = (this->output_height / h_lws + !!(this->output_height % h_lws)) * h_lws;
    size_t float_gws_dim1 = (this->output_width / w_lws  + !!(this->output_width % w_lws)) * w_lws;

    size_t float_lws_dim0 = h_lws;
    size_t float_lws_dim1 = w_lws;

    float_gws[0] = float_gws_dim0;
    float_gws[1] = float_gws_dim1;
    float_lws[0] = float_lws_dim0;
    float_lws[1] = float_lws_dim1;

    return 0;
}

template <class Dtype>
int InputLayerCL<Dtype>::ResetInputAndArgs(size_t data_size)
{
    if (data_size > this->input_data_size)
    {
        cl_int error_num;

        this->_cl_fimage = cl::Buffer(this->rt_param->context(),
                                      CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                      data_size * sizeof(float), nullptr, &error_num);
        if (!checkSuccess(error_num))
        {
            LOGE("Failed to create OpenCL buffers[%d]", error_num);
            return -1;
        }
    }
    this->input_data_size = data_size;
    return 0;

}

template <class Dtype>
int InputLayerCL<Dtype>::SetBuildOptions()
{
    clhpp_feather::CLKernelInfo& float_kernel_info = this->cl_kernel_info_map["float_chw_to_hwc"];
    std::vector<std::string>& float_build_options = float_kernel_info.build_options;

    clhpp_feather::CLKernelInfo& uint8_kernel_info = this->cl_kernel_info_map["uint8_hwc_to_hwc"];
    std::vector<std::string>& uint8_build_options = uint8_kernel_info.build_options;

    std::ostringstream ss0;
    ss0 << this->input_channels;
    float_build_options.push_back("-DIN_CHANNELS=" + ss0.str());
    std::ostringstream ss1;
    ss1 << this->channel_grp_size;
    float_build_options.push_back("-DN=" + ss1.str());
    float_build_options.push_back("-DIN_DATA_TYPE=float");
    if (std::is_same<Dtype, uint16_t>::value)
        float_build_options.push_back("-DDATA_TYPE=half");
    else
        float_build_options.push_back("-DDATA_TYPE=float");

    return 0;
}

template <class Dtype>
int InputLayerCL<Dtype>::SetKernelParameters()
{
    int error_num;
    size_t data_size;
    bool set_kernel_arguments_success = true;
    int param_idx = 0;

    this->rt_param->cl_runtime()->BuildKernel("float_chw_to_hwc", this->cl_kernel_info_map);
    clhpp_feather::CLKernelInfo& float_kernel_info = this->cl_kernel_info_map["float_chw_to_hwc"];
    std::vector<size_t>& float_gws = float_kernel_info.gws;
    std::vector<size_t>& float_lws = float_kernel_info.lws;
    cl::Kernel& float_cl_kernel = float_kernel_info.kernel;

    this->rt_param->cl_runtime()->BuildKernel("uint8_hwc_to_hwc", this->cl_kernel_info_map);
    clhpp_feather::CLKernelInfo& uint8_kernel_info = this->cl_kernel_info_map["uint8_hwc_to_hwc"];
    std::vector<size_t>& uint8_gws = uint8_kernel_info.gws;
    std::vector<size_t>& uint8_lws = uint8_kernel_info.lws;
    cl::Kernel& uint8_cl_kernel = uint8_kernel_info.kernel;

    Blob<Dtype>* layer_blob = this->_top_blobs[this->_top[0]];
    data_size = layer_blob->data_size_padded_c();
    layer_blob->AllocDevice(this->rt_param->context(), data_size);
    data_size = layer_blob->data_size();
    this->_cl_fimage = cl::Buffer(this->rt_param->context(),
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  data_size * sizeof(float), nullptr, &error_num);
    if (!checkSuccess(error_num))
    {
        LOGE("Failed to create OpenCL buffers[%d]", error_num);
        return -1;
    }
    this->input_data_size = data_size;

    cl::Buffer* layer_data_cl = layer_blob->data_cl();
    set_kernel_arguments_success &= checkSuccess(float_cl_kernel.setArg(param_idx++, this->_cl_fimage));
    set_kernel_arguments_success &= checkSuccess(float_cl_kernel.setArg(param_idx++, *layer_data_cl));
    set_kernel_arguments_success &= checkSuccess(float_cl_kernel.setArg(param_idx++, this->output_height));
    set_kernel_arguments_success &= checkSuccess(float_cl_kernel.setArg(param_idx++, this->output_width));

    if (!set_kernel_arguments_success)
    {
        LOGE("Failed setting normalinit OpenCL cl_kernels[0] arguments. %s: %s", __FILE__, __LINE__);
        return -1;
    }

    cl::ImageFormat img_format(CL_RGBA, CL_UNORM_INT8);
    this->_cl_img2d = cl::Image2D(this->rt_param->context(),
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, img_format,
                                  this->output_width, this->output_height, 0, nullptr, &error_num);

    if (error_num != CL_SUCCESS)
    {
        LOGE("Failed to create OpenCL Image2D[%d]. %s: %s", __FILE__, __LINE__, error_num);
        return -1;
    }
    param_idx = 0;
    set_kernel_arguments_success &= checkSuccess(uint8_cl_kernel.setArg(param_idx++, this->_cl_img2d));
    set_kernel_arguments_success &= checkSuccess(uint8_cl_kernel.setArg(param_idx++, *layer_data_cl));
    set_kernel_arguments_success &= checkSuccess(uint8_cl_kernel.setArg(param_idx++, this->output_height));
    set_kernel_arguments_success &= checkSuccess(uint8_cl_kernel.setArg(param_idx++, this->output_width));
    if (!set_kernel_arguments_success)
    {
        LOGE("Failed setting normalinit OpenCL cl_kernels[1] arguments. %s: %s", __FILE__, __LINE__);
        return -1;
    }
    this->rt_param->cl_runtime()->FineTuneGroupSize(float_cl_kernel, this->output_height, this->output_width, float_gws.data(), float_lws.data());


    this->rt_param->cl_runtime()->FineTuneGroupSize(uint8_cl_kernel, this->output_height, this->output_width, uint8_gws.data(), uint8_lws.data());
    return 0;
}

template <class Dtype>
int InputLayerCL<Dtype>::FloatToDevice(const float* input_data)
{
#ifdef TIMING_CL
    this->rt_param->command_queue().finish();
    timespec tpstart, tpend;
    clock_gettime(CLOCK_MONOTONIC, &tpstart);

    cl_int error_num;
    float* map_data =
        (float*)this->rt_param->command_queue().enqueueMapBuffer(this->_cl_fimage, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                0, this->input_data_size * sizeof(float), nullptr, nullptr, &error_num);
    if (!checkSuccess(error_num))
    {
        LOGE("fatal error: WriteBuffer Mapping memory objects failed [%d].  %s: %d", error_num, __FILE__, __LINE__);
        return -1;
    }

    memcpy(map_data, input_data, this->input_data_size * sizeof(float));

    error_num = this->rt_param->command_queue().enqueueUnmapMemObject(this->_cl_fimage, map_data,
                nullptr, nullptr);
    if (!checkSuccess(error_num))
    {
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
    if (!checkSuccess(error_num))
    {
        LOGE("fatal error: WriteBuffer Mapping memory objects failed [%d].  %s: %d", error_num, __FILE__, __LINE__);
        return -1;
    }

    memcpy(map_data, input_data, this->input_data_size * sizeof(float));

    error_num = this->rt_param->command_queue().enqueueUnmapMemObject(this->_cl_fimage, map_data,
                nullptr, nullptr);
    if (!checkSuccess(error_num))
    {
        LOGE("fatal error: Unmapping memory objects failed. %s: %s", __FILE__, __LINE__);
    }

#endif
    return 0;
}

template <class Dtype>
int InputLayerCL<Dtype>::UintToDevice(const uint8_t* src_bgra)
{
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
    if (!checkSuccess(error_num))
    {
        LOGE("fatal error: mapping _cl_img2d objects failed. %s: %d", error_num, __FILE__, __LINE__);
    }

    memcpy(map_data, src_bgra, this->input_data_size);

    error_num = this->rt_param->command_queue().enqueueUnmapMemObject(this->_cl_img2d, map_data, nullptr, nullptr);
    if (!checkSuccess(error_num))
    {
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
    if (!checkSuccess(error_num))
    {
        LOGE("fatal error: mapping _cl_img2d objects failed. %s: %d", error_num, __FILE__, __LINE__);
    }

    memcpy(map_data, src_bgra, this->input_data_size);

    error_num = this->rt_param->command_queue().enqueueUnmapMemObject(this->_cl_img2d, map_data, nullptr, nullptr);
    if (!checkSuccess(error_num))
    {
        LOGE("fatal error: Deconstructor Unmapping _cl_img2d objects failed.");
    }

#endif
    return 0;
}
template <class Dtype>
int InputLayerCL<Dtype>::CopyInput(std::string name, const float *input_data)
{
    this->FloatToDevice(input_data);
    this->RunKernel("float_chw_to_hwc");
    return 0;
}
template <class Dtype>
int InputLayerCL<Dtype>::CopyInput(std::string name, const uint8_t* src_bgra)
{
    this->UintToDevice(src_bgra);
    this->RunKernel("uint8_hwc_to_hwc");
    return 0;
}

template <class Dtype>
int InputLayerCL<Dtype>::ReshapeFloat(std::string name, int height, int width)
{
    if (height == this->output_height && width == this->output_width)
    {
        return 0;
    }
    bool set_kernel_arguments_success = true;
    clhpp_feather::CLKernelInfo& float_kernel_info = this->cl_kernel_info_map["float_chw_to_hwc"];
    std::vector<size_t>& float_gws = float_kernel_info.gws;
    std::vector<size_t>& float_lws = float_kernel_info.lws;
    cl::Kernel& cl_kernel = float_kernel_info.kernel;

    int num = this->_top_blobs[name]->num();
    int channels = this->_top_blobs[name]->channels();
    if (this->_top_blobs[name]->ReshapeWithReallocDevice(this->rt_param->context(), num, channels, height, width) == 2)
    {
        cl::Buffer* layer_data_cl = this->_top_blobs[name]->data_cl();
        set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(1, *layer_data_cl));
    }
    this->output_height = this->_top_blobs[name]->height();
    this->output_width = this->_top_blobs[name]->width();

    ResetInputAndArgs(num * channels * height * width);
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(0, this->_cl_fimage));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(2, this->output_height));
    set_kernel_arguments_success &= checkSuccess(cl_kernel.setArg(3, this->output_width));
    if (!set_kernel_arguments_success)
    {
        LOGE("Failed setting normalinit OpenCL cl_kernels[0] arguments. %s: %s", __FILE__, __LINE__);
        return -1;
    }
    this->ResetWorkSizeFloat();
    this->rt_param->cl_runtime()->FineTuneGroupSize(cl_kernel, this->output_height, this->output_width, float_gws.data(), float_gws.data());
    return 0;
}


template <class Dtype>
int InputLayerCL<Dtype>::RunKernel(std::string kernel_type)
{
    clhpp_feather::CLKernelInfo& input_kernel_info = this->cl_kernel_info_map[kernel_type];
    std::vector<size_t>& input_gws = input_kernel_info.gws;
    std::vector<size_t>& input_lws = input_kernel_info.lws;
    cl::Kernel& cl_kernel = input_kernel_info.kernel;
#ifdef TIMING_CL
    this->rt_param->command_queue().finish();
    timespec tpstart, tpend;
    cl::Event event;
    std::string cl_program_name = input_kernel_info.program_name;
    clock_gettime(CLOCK_MONOTONIC, &tpstart);

    int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                        cl_kernel, cl::NullRange, cl::NDRange(input_gws[0], input_gws[1], input_gws[2]),
                        cl::NDRange(input_lws[0], input_lws[1], input_lws[2]), nullptr, &event);

    if (!checkSuccess(error_num))
    {
        LOGE("Failed enqueuing the normalinit kernel.");
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
    LOGI("[%s] [%s] Execution time in kernel: %0.5f, %0.5f, %0.5f\n", this->name().c_str(), cl_program_name.c_str(), submit_kerel_time, start_kerel_time, stop_kerel_time);

#else
    int error_num = this->rt_param->command_queue().enqueueNDRangeKernel(
                        cl_kernel, cl::NullRange, cl::NDRange(input_gws[0], input_gws[1], input_gws[2]),
                        cl::NDRange(input_lws[0], input_lws[1], input_lws[2]), nullptr, nullptr);

    if (!checkSuccess(error_num))
    {
        LOGE("Failed enqueuing the normalinit kernel.");
        return -1;
    }

#endif

    return 0;
}

template class InputLayerCL<float>;
template class InputLayerCL<uint16_t>;

}; // namespace feather
