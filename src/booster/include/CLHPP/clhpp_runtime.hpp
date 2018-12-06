#ifndef CLHPP_RUNTIME_HPP
#define CLHPP_RUNTIME_HPP

#include <cstring>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include "CLHPP/clhpp_common.hpp"

namespace clhpp_feather {

enum GPUType {
  QUALCOMM_ADRENO,
  MALI,
  PowerVR,
  UNKNOWN,
};

enum OpenCLVersion {
  CL_VER_1_0,
  CL_VER_1_1, 
  CL_VER_1_2,
  CL_VER_2_0,
  CL_VER_UNKNOWN,
};

inline std::vector<std::string> Split(const std::string &str, char delims) {
  std::vector<std::string> result;
  std::string tmp = str;
  while (!tmp.empty()) {
    size_t next_offset = tmp.find(delims);
    result.push_back(tmp.substr(0, next_offset));
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp = tmp.substr(next_offset + 1);
    }
  }
  return result;
}

class OpenCLRuntime {
public:
  OpenCLRuntime();
  ~OpenCLRuntime();
  int OpenCLProbe();

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

  std::string FeatherOpenclVersion() {
    return _feather_opencl_version; 
  }
  std::string GpuDeviceName() {
    return _gpu_device_name;
  }
  std::string GpuDeviceVersion() {
    return _gpu_device_version;
  }
  void PrintOpenCLInfo() {
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

};


}

#endif