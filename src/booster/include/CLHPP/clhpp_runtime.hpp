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
#pragma once

#include <cstring>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <map>

#include "common.h"
#include "CLHPP/clhpp_common.hpp"
#include "CLHPP/clhpp_tuner.hpp"

extern const std::string g_pre_kernel_dir;

namespace clhpp_feather
{

struct CLKernelInfo
{
    std::string program_name;
    std::string kernel_name;
    std::string kernel_source;
    cl::Program program;
    cl::Kernel kernel;
    std::vector<std::string> build_options;
    std::vector<size_t> gws;
    std::vector<size_t> lws;
};

enum GPUType
{
    QUALCOMM_ADRENO,
    MALI,
    PowerVR,
    UNKNOWN,
};

enum OpenCLVersion
{
    CL_VER_1_0,
    CL_VER_1_1,
    CL_VER_1_2,
    CL_VER_2_0,
    CL_VER_UNKNOWN,
};

inline std::vector<std::string> Split(const std::string &str, char delims)
{
    std::vector<std::string> result;
    std::string tmp = str;
    while (!tmp.empty())
    {
        size_t next_offset = tmp.find(delims);
        result.push_back(tmp.substr(0, next_offset));
        if (next_offset == std::string::npos)
        {
            break;
        }
        else
        {
            tmp = tmp.substr(next_offset + 1);
        }
    }
    return result;
}

class OpenCLRuntime
{
    public:
        OpenCLRuntime();
        ~OpenCLRuntime();
        int OpenCLProbe();
        int GetDeviceMaxWorkGroupSize(uint64_t& size);
        int GetDeviceMaxMemAllocSize(uint64_t& size);
        int IsImageSupport(cl_bool& res);
        int GetMaxImage2DSize(size_t& m_height, size_t& m_width);
        int GetKernelMaxWorkGroupSize(const cl::Kernel &kernel, uint64_t& size);
        int GetKernelWaveSize(const cl::Kernel &kernel, uint64_t& size);
        int FineTuneGroupSize(const cl::Kernel &kernel,
                              const size_t &height,
                              const size_t &width,
                              size_t *gws,
                              size_t *lws);
        int BuildKernel(const std::string& cl_kernel_name,
                        std::map<std::string, clhpp_feather::CLKernelInfo>& cl_kernel_info_map);

        cl::Context &context() const
        {
            return *_context;
        }
        cl::CommandQueue &command_queue() const
        {
            return *_command_queue;
        }
        cl::Device &device() const
        {
            return *_device;
        }
        std::map<std::string, cl::Program> &cl_program_map() const
        {
            return *_cl_program_map;
        }
        clhpp_feather::Tuner<size_t> &tuner() const
        {
            return *_tuner;
        }


        std::string FeatherOpenclVersion()
        {
            return _feather_opencl_version;
        }
        std::string GpuDeviceName()
        {
            return _gpu_device_name;
        }
        std::string GpuDeviceVersion()
        {
            return _gpu_device_version;
        }
        void PrintOpenCLInfo()
        {
            LOGI("----OpenCLInfo----");
            LOGI("[DeviceName]:           [%s]", this->GpuDeviceName().c_str());
            LOGI("[DeviceVersion]:        [%s]", this->GpuDeviceVersion().c_str());
            LOGI("[FeatherOpenclVersion]: [%s]", this->FeatherOpenclVersion().c_str());
        }


    private:
        OpenCLVersion _opencl_version;
        GPUType _gpu_type;
        std::string _gpu_device_name;
        std::string _gpu_device_version;
        std::string _feather_opencl_version = "1.1.0";

        std::shared_ptr<cl::Context> _context;
        std::shared_ptr<cl::Device> _device;
        std::shared_ptr<cl::CommandQueue> _command_queue;
        std::shared_ptr<std::map<std::string, cl::Program> > _cl_program_map;
        std::shared_ptr<clhpp_feather::Tuner<size_t> > _tuner;
};


}
