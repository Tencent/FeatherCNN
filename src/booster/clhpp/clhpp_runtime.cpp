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


using namespace std;


namespace clhpp_feather{

  GPUType ParseGPUType(const std::string &device_name) {
    constexpr const char *kQualcommAdrenoGPUStr = "QUALCOMM Adreno(TM)";
    constexpr const char *kMaliGPUStr = "Mali";
    constexpr const char *kPowerVRGPUStr = "PowerVR";

    if (device_name == kQualcommAdrenoGPUStr) {
      return GPUType::QUALCOMM_ADRENO;
    } else if (device_name.find(kMaliGPUStr) != std::string::npos) {
      return GPUType::MALI;
    } else if (device_name.find(kPowerVRGPUStr) != std::string::npos) {
      return GPUType::PowerVR;
    } else {
      return GPUType::UNKNOWN;
    }
  }

  OpenCLVersion ParseDeviceVersion(
      const std::string &device_version) {
    // OpenCL Device version string format:
    // OpenCL<space><major_version.minor_version><space>
    // <vendor-specific information>
    auto words = Split(device_version, ' ');
    if (words[1] == "2.0") {
      return OpenCLVersion::CL_VER_2_0;
    } else if (words[1] == "1.2") {
      return OpenCLVersion::CL_VER_1_2;
    } else if (words[1] == "1.1") {
      return OpenCLVersion::CL_VER_1_1;
    } else if (words[1] == "1.0") {
      return OpenCLVersion::CL_VER_1_0;
    } else {
      LOGE("Do not support OpenCL version: %s", words[1].c_str());
      return OpenCLVersion::CL_VER_UNKNOWN;
    }
  }

  OpenCLRuntime::OpenCLRuntime() {
    OpenCLProbe();
  }
  OpenCLRuntime::~OpenCLRuntime() {
    _command_queue->finish();
    _command_queue.reset();
    _context.reset();
    _device.reset();
  }

  int OpenCLRuntime::OpenCLProbe() {
    cl_int err;
    
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size()==0) {
        LOGE(" No platforms found. Check OpenCL installation!");
        return -1;
    }
    cl::Platform default_platform=all_platforms[0];
    LOGI("Using platform: %s", default_platform.getInfo<CL_PLATFORM_NAME>().c_str());

    // get default device (CPUs, GPUs) of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        LOGE("No devices found. Check OpenCL installation!");
        return -1;
    }

    bool gpu_detected = false;
    _device = std::make_shared<cl::Device>();
    for (auto device : all_devices) {
      if (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
        *_device = device;
        gpu_detected = true;
        _gpu_device_name = device.getInfo<CL_DEVICE_NAME>();
        _gpu_type = ParseGPUType(_gpu_device_name);
        _gpu_device_version = device.getInfo<CL_DEVICE_VERSION>();
        _opencl_version = ParseDeviceVersion(_gpu_device_version);
        if (_opencl_version == OpenCLVersion::CL_VER_UNKNOWN) {
          return -1;
        }
        LOGI("Using device_name: [%s], device_version [%s]", _gpu_device_name.c_str(), _gpu_device_version.c_str());
        break;
      }
    }
    if(!gpu_detected) {
      LOGE("No Gpu device found");
      return -1;
    }

    //_context
    cl_command_queue_properties properties = 0;
    properties |= CL_QUEUE_PROFILING_ENABLE;

    _context = std::shared_ptr<cl::Context>(
          new cl::Context({*_device}, nullptr, nullptr, nullptr, &err));
    if(!checkSuccess(err)) {
      LOGE("new cl::Context error");
      return -1;
    }

    _command_queue = std::make_shared<cl::CommandQueue>(*_context,
                                                      *_device,
                                                      properties,
                                                      &err);
    if(!checkSuccess(err)){
      LOGE("new cl::CommandQueue error");
      return -1;
    }

    return 0;

  }


} //namespace cl_feather