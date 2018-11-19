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

/*
 * For runtime parameters
 */

#pragma once

#include "mempool.h"

// #define FEATHER_OPENCL
#ifdef FEATHER_OPENCL
#include <CL/cl.h>
#endif

enum DeviceType
{
    CPU = 0,
    GPU_CL = 1,
    GPU_GL = 2
};

template<typename Dtype>
class RuntimeParameter
{
    public:
      RuntimeParameter() : _common_mempool(NULL),
#ifdef FEATHER_OPENCL
                           _device_type(DeviceType::CPU), 
                           _context(NULL), 
                           _commandQueue(NULL), _device(NULL),
#endif
                           _num_threads(1)
      {
      }
      RuntimeParameter(CommonMemPool<Dtype> *common_mempool, size_t num_threads)
          : _common_mempool(common_mempool), 
          _num_threads(num_threads)
      {
      }

#ifdef FEATHER_OPENCL
      int SetupOpenCLEnv(
          const cl_context &context,
          const cl_command_queue &command_queue,
          const cl_device_id &device)
      {
          _device_type = DeviceType::GPU_CL;
          _context = context;
          _command_queue = command_queue;
          _device = device;
      }
#endif

      CommonMemPool<Dtype> *common_mempool() const
      {
          return _common_mempool;
      }
      size_t num_threads() const
      {
          return _num_threads;
      }


        DeviceType device_type() const
        {
            return _device_type;
        }
#ifdef FEATHER_OPENCL
        cl_context context() const
        {
            return _context;
        }
        cl_command_queue commandQueue() const
        {
            return _commandQueue;
        }
        cl_device_id device() const
        {
            return _device;
        }
#endif

      private:
        CommonMemPool<Dtype> *_common_mempool;
        size_t _num_threads;

        DeviceType _device_type;
#ifdef FEATHER_OPENCL
        cl_context _context;
        cl_command_queue _command_queue;
        cl_device_id _device;
#endif
};
