#ifndef CL_RUNTIME_H
#define CL_RUNTIME_H

#include <cstring>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include "CL/cl_common.h"

namespace cl_feather {

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

class OpenCLRuntime {
public:
  OpenCLRuntime();
  ~OpenCLRuntime();
  int OpenCLProbe();

  cl_context context() const
  {
      return _context;
  }
  cl_command_queue command_queue() const
  {
      return _command_queue;
  }
  cl_device_id device() const
  {
      return _device;
  }

private:
  OpenCLVersion _opencl_version;
  GPUType _gpu_type;
  
  cl_context _context;
  cl_command_queue _command_queue;
  cl_device_id _device;

};


}

#endif