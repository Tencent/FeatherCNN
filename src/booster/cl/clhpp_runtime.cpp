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
#include "CLHPP/clhpp_runtime.hpp"

// Adreno extensions
// Adreno performance hints
typedef cl_uint cl_perf_hint;
#define CL_CONTEXT_PERF_HINT_QCOM 0x40C2
#define CL_PERF_HINT_HIGH_QCOM 0x40C3
#define CL_PERF_HINT_NORMAL_QCOM 0x40C4
#define CL_PERF_HINT_LOW_QCOM 0x40C5
// Adreno priority hints
typedef cl_uint cl_priority_hint;
#define CL_PRIORITY_HINT_NONE_QCOM 0
#define CL_CONTEXT_PRIORITY_HINT_QCOM 0x40C9
#define CL_PRIORITY_HINT_HIGH_QCOM 0x40CA
#define CL_PRIORITY_HINT_NORMAL_QCOM 0x40CB
#define CL_PRIORITY_HINT_LOW_QCOM 0x40CC

#define CL_KERNEL_WAVE_SIZE_QCOM 0xAA02


using namespace std;



namespace clhpp_feather
{

GPUType ParseGPUType(const std::string &device_name)
{
    constexpr const char *kQualcommAdrenoGPUStr = "QUALCOMM Adreno(TM)";
    constexpr const char *kMaliGPUStr = "Mali";
    constexpr const char *kPowerVRGPUStr = "PowerVR";

    if (device_name == kQualcommAdrenoGPUStr)
    {
        return GPUType::QUALCOMM_ADRENO;
    }
    else if (device_name.find(kMaliGPUStr) != std::string::npos)
    {
        return GPUType::MALI;
    }
    else if (device_name.find(kPowerVRGPUStr) != std::string::npos)
    {
        return GPUType::PowerVR;
    }
    else
    {
        return GPUType::UNKNOWN;
    }
}

OpenCLVersion ParseDeviceVersion(
    const std::string &device_version)
{
    // OpenCL Device version string format:
    // OpenCL<space><major_version.minor_version><space>
    // <vendor-specific information>
    auto words = Split(device_version, ' ');
    if (words[1] == "2.0")
    {
        return OpenCLVersion::CL_VER_2_0;
    }
    else if (words[1] == "1.2")
    {
        return OpenCLVersion::CL_VER_1_2;
    }
    else if (words[1] == "1.1")
    {
        return OpenCLVersion::CL_VER_1_1;
    }
    else if (words[1] == "1.0")
    {
        return OpenCLVersion::CL_VER_1_0;
    }
    else
    {
        LOGE("Do not support OpenCL version: %s", words[1].c_str());
        return OpenCLVersion::CL_VER_UNKNOWN;
    }
}

OpenCLRuntime::OpenCLRuntime()
{
    _cl_program_map = std::make_shared<std::map<std::string, cl::Program> >();
    _tuner = std::make_shared<clhpp_feather::Tuner<size_t> > ();
    OpenCLProbe();
    PrintOpenCLInfo();

    dirCreate(g_pre_kernel_dir);
}
OpenCLRuntime::~OpenCLRuntime()
{
    _command_queue->finish();
    _command_queue.reset();
    _context.reset();
    _device.reset();
    _cl_program_map.reset();
}

int OpenCLRuntime::OpenCLProbe()
{
    cl_int err;

    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0)
    {
        LOGE(" No platforms found. Check OpenCL installation!");
        return -1;
    }
    cl::Platform default_platform = all_platforms[0];
    LOGI("Using platform: %s", default_platform.getInfo<CL_PLATFORM_NAME>().c_str());

    // get default device (CPUs, GPUs) of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0)
    {
        LOGE("No devices found. Check OpenCL installation!");
        return -1;
    }

    bool gpu_detected = false;
    _device = std::make_shared<cl::Device>();
    for (auto device : all_devices)
    {
        if (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU)
        {
            *_device = device;
            gpu_detected = true;
            _gpu_device_name = device.getInfo<CL_DEVICE_NAME>();
            _gpu_type = ParseGPUType(_gpu_device_name);
            _gpu_device_version = device.getInfo<CL_DEVICE_VERSION>();
            _opencl_version = ParseDeviceVersion(_gpu_device_version);
            if (_opencl_version == OpenCLVersion::CL_VER_UNKNOWN)
            {
                return -1;
            }
            LOGI("Using device_name: [%s], device_version [%s]", _gpu_device_name.c_str(), _gpu_device_version.c_str());
            break;
        }
    }
    if (!gpu_detected)
    {
        LOGE("No Gpu device found");
        return -1;
    }

    //_context
    cl_command_queue_properties properties = 0;
    properties |= CL_QUEUE_PROFILING_ENABLE;

    _context = std::shared_ptr<cl::Context>(
                   new cl::Context({*_device}, nullptr, nullptr, nullptr, &err));
    if (!checkSuccess(err))
    {
        LOGE("new cl::Context error");
        return -1;
    }

    _command_queue = std::make_shared<cl::CommandQueue>(*_context,
                     *_device,
                     properties,
                     &err);
    if (!checkSuccess(err))
    {
        LOGE("new cl::CommandQueue error");
        return -1;
    }

    return 0;

}

int OpenCLRuntime::GetDeviceMaxWorkGroupSize(uint64_t& size)
{
    cl_int error_num = _device->getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);
    if (error_num != CL_SUCCESS)
    {
        LOGE("GetDeviceMaxWorkGroupSize error: %s", OpenCLErrorToString(error_num).c_str());
        return -1;
    }
    return 0;
}

int OpenCLRuntime::GetDeviceMaxMemAllocSize(uint64_t& size)
{
    cl_int error_num = _device->getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &size);
    if (error_num != CL_SUCCESS)
    {
        LOGE("GetDeviceMaxMemAllocSize error: %s", OpenCLErrorToString(error_num).c_str());
        return -1;
    }
    return 0;
}

int OpenCLRuntime::IsImageSupport(cl_bool& res)
{
    cl_int error_num = _device->getInfo(CL_DEVICE_IMAGE_SUPPORT, &res);
    if (error_num != CL_SUCCESS)
    {
        LOGE("IsImageSupport error: %s", OpenCLErrorToString(error_num).c_str());
        return false;
    }
    return res == CL_TRUE;
}

int OpenCLRuntime::GetMaxImage2DSize(size_t& m_height, size_t& m_width)
{
    cl_int error_num = _device->getInfo(CL_DEVICE_IMAGE2D_MAX_HEIGHT, &m_height);
    if (error_num != CL_SUCCESS)
    {
        LOGE("GetMaxImage2DSize height error: %s", OpenCLErrorToString(error_num).c_str());
        return -1;
    }
    error_num = _device->getInfo(CL_DEVICE_IMAGE2D_MAX_WIDTH, &m_width);
    if (error_num != CL_SUCCESS)
    {
        LOGE("GetMaxImage2DSize width error: %s", OpenCLErrorToString(error_num).c_str());
        return -1;
    }
    return 0;
}

int OpenCLRuntime::GetKernelMaxWorkGroupSize(const cl::Kernel& kernel, uint64_t& size)
{
    cl_int error_num = kernel.getWorkGroupInfo(*_device, CL_KERNEL_WORK_GROUP_SIZE,
                       &size);

    if (error_num != CL_SUCCESS)
    {
        LOGE("GetKernelMaxWorkGroupSize error: %s", OpenCLErrorToString(error_num).c_str());
        return -1;
    }
    return 0;
}


int OpenCLRuntime::GetKernelWaveSize(const cl::Kernel &kernel, uint64_t& size)
{
    cl_int error_num = kernel.getWorkGroupInfo(*_device, CL_KERNEL_WAVE_SIZE_QCOM,
                       &size);
    if (error_num != CL_SUCCESS)
    {
        LOGE("GetKernelWaveSize error: %s", OpenCLErrorToString(error_num).c_str());
        return -1;
    }
    return 0;
}



int OpenCLRuntime::FineTuneGroupSize(const cl::Kernel& kernel,
                                     const size_t& height,
                                     const size_t& width,
                                     size_t *gws,
                                     size_t *lws)
{
    //gws HWC
    //lws  HWC
    uint64_t current_work_group_size = 0;
    uint64_t kernel_work_group_size = 0;

    if (this->GetDeviceMaxWorkGroupSize(current_work_group_size))
    {
        LOGE("Get kernel work group info failed.");
        return -1;
    }

    if (this->GetKernelMaxWorkGroupSize(kernel, kernel_work_group_size))
    {
        LOGE("Get kernel work group size failed.");
        return -1;
    }

    uint64_t total_lws = lws[0] * lws[1] * lws[2];
    uint64_t valid_min_wgs = std::min(current_work_group_size,  kernel_work_group_size);
    int flag = 0;
    while (total_lws > valid_min_wgs)
    {
        if (lws[2] > 1 && (flag % 3) == 0)
        {
            lws[2] /= 2;
        }
        else if (lws[1] > 1 && (flag % 3) == 1)
        {
            lws[1] /= 2;
        }
        else if (lws[0] > 1 && (flag % 3) == 2)
        {
            lws[0] /= 2;
        }
        flag++;
        total_lws = lws[0] * lws[1] * lws[2];
    }

    gws[0] = (height / lws[0] + !!(height % lws[0])) * lws[0];
    gws[1] = (width / lws[1]  + !!(width % lws[1])) * lws[1];
    return 0;
}


int OpenCLRuntime::BuildKernel(const std::string& cl_kernel_name, std::map<std::string, clhpp_feather::CLKernelInfo>& cl_kernel_info_map)
{
    clhpp_feather::CLKernelInfo& kernel_info = cl_kernel_info_map[cl_kernel_name];
    const std::string& program_name = kernel_info.program_name;
    const std::string& kernel_name = kernel_info.kernel_name;
    const std::string& kernel_source = kernel_info.kernel_source;
    cl::Program& program = kernel_info.program;
    cl::Kernel& kernel = kernel_info.kernel;
    const std::vector<std::string>& build_options = kernel_info.build_options;

    std::string opt_str = "";
    for (auto &opt : build_options)
    {
        opt_str += " " + opt;
    }
    std::string promap_key = program_name + opt_str;

    if (_cl_program_map->find(promap_key) != _cl_program_map->end())
    {
        program = (*_cl_program_map)[promap_key];
    }
    else
    {
        std::string kernelAddr = promap_key + ".bin";
        StringTool::RelaceString(kernelAddr, "/", "_");
        StringTool::RelaceString(kernelAddr, ":", "_");
        StringTool::RelaceString(kernelAddr, " ", "_");
        StringTool::RelaceString(kernelAddr, "-", "_");
        StringTool::RelaceString(kernelAddr, "=", "_");

        kernelAddr = "no_save";
        if (buildProgram(*_context,
                         *_device,
                         program,
                         kernel_source,
                         opt_str,
                         kernelAddr))
        {
            LOGE("Build program failed.");
            return -1;
        }
        (*_cl_program_map)[promap_key] = program;
    }

    int error_num;
    kernel = cl::Kernel(program, kernel_name.c_str(), &error_num);
    if (error_num != CL_SUCCESS) {
        LOGE("failed to build kernel %s from program %s [%s]", kernel_name.c_str(),
                                                               program_name.c_str(),
                                                               OpenCLErrorToString(error_num).c_str());
        return -1;
    }

    return 0;
}



} //namespace cl_feather
