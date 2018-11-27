#include "CL/cl_runtime.h"

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


using namespace std;

namespace cl_feather{
  OpenCLRuntime::OpenCLRuntime() {
    OpenCLProbe();
  }
  OpenCLRuntime::~OpenCLRuntime() {
    if (!checkSuccess(clFinish(_command_queue))){
      LOGE("Failed waiting for command queue run. ");
    }

    if(_context != 0){
        clReleaseContext(_context);
    }

    if (_command_queue != 0){
        clReleaseCommandQueue(_command_queue);
    }
  }

  int OpenCLRuntime::OpenCLProbe() {
    cl_int error_num;
    cl_platform_id platforms;
    cl_uint num_platforms;
    char buffer[1024];

    error_num = clGetPlatformIDs(1, &platforms, &num_platforms);

    if (error_num){
        LOGE("failed to clGetPlatformIDs: %d", error_num);
        return -1;
    }
    error_num = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_GPU, 1, &_device, &num_platforms);
    if(error_num){
        LOGE("failed to clGetDeviceIDs: %d", error_num);
        return -1;
    } else {
        LOGD("number of platforms is: %d", num_platforms);
    }

    error_num = clGetDeviceInfo(_device, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
    if (error_num) {
        LOGE("failed to clGetDeviceInfo CL_DEVICE_NAME: %d", error_num);
        return -1;
    }
    else {
        LOGD("CL_DEVICE_NAME = %s", buffer);
    }

    string tmp_device_name(buffer);

    error_num = clGetDeviceInfo(_device, CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
    if (error_num) {
        LOGE("failed to clGetDeviceInfo CL_DEVICE_VERSION: %d", error_num);
        return -1;
    }
    else {
        LOGD("CL_DEVICE_VERSION = %s", buffer);
    }
    string tmp_device_version(buffer);

    error_num = clGetDeviceInfo(_device, CL_DEVICE_EXTENSIONS, sizeof(buffer), buffer, NULL);
    if (error_num) {
        LOGE("failed to clGetDeviceInfo CL_DEVICE_EXTENSIONS: %d", error_num);
        return -1;
    }
    else {
        LOGD("CL_DEVICE_EXTENSIONS = %s", buffer);
    }

    LOGD("Gpu:[%s][%s]", tmp_device_name.c_str(), tmp_device_version.c_str());
    if(tmp_device_name == "QUALCOMM Adreno(TM)" && tmp_device_version.find("2.0") != std::string::npos)
    {
        
        LOGD("QUALCOMM 2.0 opt High");
        std::vector<cl_context_properties> context_properties;
        context_properties.reserve(5);
        context_properties.push_back(CL_CONTEXT_PERF_HINT_QCOM);
        context_properties.push_back(CL_PERF_HINT_HIGH_QCOM);
        context_properties.push_back(CL_CONTEXT_PRIORITY_HINT_QCOM);
        context_properties.push_back(CL_PRIORITY_HINT_HIGH_QCOM);

        _context = clCreateContext(context_properties.data(), 1, &_device, NULL, NULL, &error_num);

    }
    else {
        _context = clCreateContext(NULL, 1, &_device, NULL, NULL, &error_num);
    }
    
    if (error_num) {
        LOGE("failed to clCreateContext: %d", error_num);
        return -1;
    }

    if (!createCommandQueue(_context, &_command_queue, &_device)){
        LOGE("failed to createCommandQueue");
        return -1;
    }
    return 0;
  }


} //namespace cl_feather