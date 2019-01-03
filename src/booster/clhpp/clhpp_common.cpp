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
#include "CLHPP/clhpp_common.hpp"


using namespace std;


const std::string OpenCLErrorToString(cl_int error)
{
    switch (error)
    {
        case CL_SUCCESS:
            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:
            return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_COMPILE_PROGRAM_FAILURE:
            return "CL_COMPILE_PROGRAM_FAILURE";
        case CL_LINKER_NOT_AVAILABLE:
            return "CL_LINKER_NOT_AVAILABLE";
        case CL_LINK_PROGRAM_FAILURE:
            return "CL_LINK_PROGRAM_FAILURE";
        case CL_DEVICE_PARTITION_FAILED:
            return "CL_DEVICE_PARTITION_FAILED";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
            return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case CL_INVALID_VALUE:
            return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:
            return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY:
            return "CL_INVALID_PROPERTY";
        case CL_INVALID_IMAGE_DESCRIPTOR:
            return "CL_INVALID_IMAGE_DESCRIPTOR";
        case CL_INVALID_COMPILER_OPTIONS:
            return "CL_INVALID_COMPILER_OPTIONS";
        case CL_INVALID_LINKER_OPTIONS:
            return "CL_INVALID_LINKER_OPTIONS";
        case CL_INVALID_DEVICE_PARTITION_COUNT:
            return "CL_INVALID_DEVICE_PARTITION_COUNT";
        case CL_INVALID_PIPE_SIZE:
            return "CL_INVALID_PIPE_SIZE";
        case CL_INVALID_DEVICE_QUEUE:
            return "CL_INVALID_DEVICE_QUEUE";
        default:
            return "UNKNOWN";
    }
}

bool fileIsExists(std::string fileAddress)
{
    ifstream f(fileAddress.c_str());
    return f.good();
}

bool dirIsExists(std::string dirAddress)
{
    if (opendir(dirAddress.c_str()) == NULL)
        return false;
    return true;
}

bool dirCreate(std::string dirAddress)
{
    LOGE("dirCreate %s", dirAddress.c_str());
    if (!dirIsExists(dirAddress))
        mkdir(dirAddress.c_str(), 0775);
    return dirIsExists(dirAddress);
}

bool checkSuccess(cl_int error_num)
{
    if (error_num != CL_SUCCESS)
    {
        LOGE("OpenCL error: %s", OpenCLErrorToString(error_num).c_str());
        return false;
    }
    return true;
}

int buildProgramFromSource(const cl::Context& context,
                           const cl::Device & device,
                           cl::Program& program,
                           const std::string& kernel_code,
                           std::string build_opts,
                           std::string kernelAddr)
{
    cl_int error_num;
    cl::Program::Sources sources;
    sources.push_back(kernel_code);
    program = cl::Program(context, sources);
    build_opts += " -cl-mad-enable -cl-fast-relaxed-math";
    error_num = program.build({device}, build_opts.c_str());

    if (!checkSuccess(error_num))
    {
        if (program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) ==
                CL_BUILD_ERROR)
        {
            std::string build_log =
                program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            LOGE("Program build log: %s", build_log.c_str());
        }
        LOGE("Build program from source failed\n");
        return -1;
    }

    // Keep built program binary
    LOGI("keep build program in [%s]", kernelAddr.c_str());
    size_t binarySize;
    error_num = clGetProgramInfo(program(), CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binarySize, NULL);
    if (!checkSuccess(error_num))
    {
        LOGE("Get program info binary size failed");
        return 0;
    }
    unsigned char* binaryPtr = (unsigned char*) malloc(binarySize);
    error_num = clGetProgramInfo(program(), CL_PROGRAM_BINARIES,
                                 binarySize, &binaryPtr, NULL);
    if (!checkSuccess(error_num))
    {
        LOGE("Get program info binary str failed");
        return 0;
    }
    FILE* binaryF = fopen(kernelAddr.c_str(), "w");
    if (binaryF == NULL)
    {
        LOGE("fopen w file %s %d error", kernelAddr.c_str(), binarySize);
        return 0;
    }
    fwrite(binaryPtr, binarySize, 1, binaryF);
    fclose(binaryF);
    free(binaryPtr);
    //LOGE("fopen w file %s %d success", kernelAddr.c_str(), binarySize);
    return 0;
}

int buildProgramFromPrecompiledBinary(const cl::Context& context,
                                      const cl::Device & device,
                                      cl::Program& program,
                                      const std::string& kernel_code,
                                      std::string build_opts,
                                      std::string kernelAddr)
{
    if (!fileIsExists(kernelAddr))
        return -1;
    FILE* binaryF = fopen(kernelAddr.c_str(), "r");
    if (binaryF == NULL)
    {
        LOGE("fopen r file %s error", kernelAddr.c_str());
        return -1;
    }
    fseek(binaryF, 0, SEEK_END);
    size_t kernelSize = ftell(binaryF);
    fseek(binaryF, 0, SEEK_SET);
    unsigned char* kernelCode = (unsigned char *)malloc(kernelSize);
    if (fread(kernelCode, 1, kernelSize, binaryF) < kernelSize)
    {
        LOGE("fread r file %s %d error", kernelAddr.c_str(), kernelSize);
        return -1;
    }
    fclose(binaryF);

    std::vector<unsigned char> content(
        reinterpret_cast<unsigned char const *>(kernelCode),
        reinterpret_cast<unsigned char const *>(kernelCode) + kernelSize);

    cl_int error_num;
    program = cl::Program(context, {device}, {content});

    build_opts += " -cl-mad-enable -cl-fast-relaxed-math";
    error_num = program.build({device}, build_opts.c_str());
    if (!checkSuccess(error_num))
    {
        if (program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) ==
                CL_BUILD_ERROR)
        {
            std::string build_log =
                program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            LOGE("Program Pre build log: %s", build_log.c_str());
        }
        return -1;
    }
    free(kernelCode);
    return 0;
}

int buildProgram(const cl::Context& context,
                 const cl::Device & device,
                 cl::Program& program,
                 const std::string& kernel_code,
                 std::string build_opts,
                 std::string kernelAddr)
{
    if (kernelAddr == "no_save")
        return buildProgramFromSource(context, device, program, kernel_code, build_opts);

    kernelAddr = g_pre_kernel_dir + kernelAddr;
    int ret = 0;
    ret = buildProgramFromPrecompiledBinary(context, device, program, kernel_code, build_opts, kernelAddr);
    if (ret != 0)
    {
        ret = buildProgramFromSource(context, device, program, kernel_code, build_opts, kernelAddr);
        return ret;
    }
    return 0;
}

int buildProgramFromSource(const cl::Context& context,
                           const cl::Device & device,
                           cl::Program& program,
                           const std::string& kernel_code,
                           std::string build_opts)
{
    cl_int error_num;
    cl::Program::Sources sources;
    sources.push_back(kernel_code);
    program = cl::Program(context, sources);
    build_opts += " -cl-mad-enable -cl-fast-relaxed-math";
    error_num = program.build({device}, build_opts.c_str());

    if (!checkSuccess(error_num))
    {
        if (program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) ==
                CL_BUILD_ERROR)
        {
            std::string build_log =
                program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            LOGE("Program build log: %s", build_log.c_str());
        }
        LOGE("Build program from source failed\n");
        return -1;
    }
    return 0;

}
