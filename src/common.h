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

#include <string>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <pthread.h>
// #include <android/log.h>
#include <stdio.h>
#if 0
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "FeatherLib", __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "FeatherLib", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "FeatherLib", __VA_ARGS__)
#else
#include <stdio.h>
#define LOGI(...) fprintf(stdout, __VA_ARGS__);fprintf(stdout,"\n");
#define LOGD(...) fprintf(stdout, __VA_ARGS__);fprintf(stdout,"\n");
#define LOGE(...) fprintf(stderr, __VA_ARGS__);fprintf(stderr,"\n");
#endif
class StringTool
{
    public:
        static void SplitString(const std::string &input, const std::string &delim, std::vector<std::string> &parts);
};


int min(int a, int b);
void* _mm_malloc(size_t sz, size_t align);
void _mm_free(void* ptr);

#ifdef FEATHER_OPENCL
#include <CL/cl.h>
/**
 * \brief Create an OpenCL command queue for a given context.
 * \param[in] context The OpenCL context to use.
 * \param[out] commandQueue The created OpenCL command queue.
 * \param[out] device The device in which the command queue is created.
 * \return False if an error occurred, otherwise true.
 */
bool createCommandQueue(cl_context context, cl_command_queue* commandQueue, cl_device_id* device);
/**
 * \brief Convert OpenCL error numbers to their string form.
 * \details Uses the error number definitions from cl.h.
 * \param[in] errorNumber The error number returned from an OpenCL command.
 * \return A name of the error.
 */
std::string errorNumberToString(cl_int errorNumber);
/**
 * \brief Check an OpenCL error number for errors.
 * \details If errorNumber is not CL_SUCESS, the function will print the string form of the error number.
 * \param[in] errorNumber The error number returned from an OpenCL command.
 * \return False if errorNumber != CL_SUCCESS, true otherwise.
 */
bool checkSuccess(cl_int errorNumber);

int buildProgramFromSource(const cl_context& context, const cl_device_id& device, cl_program& program,
                            const std::string& kernel_code, std::string build_opts);
#endif
