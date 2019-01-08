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

#pragma once

#include <string>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <pthread.h>
#include <sys/system_properties.h>

#ifdef FEATHER_OPENCL
#include "CLHPP/clhpp_common.hpp"
#endif

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


typedef unsigned short half;

class StringTool
{
    public:
        static void SplitString(const std::string &input, const std::string &delim, std::vector<std::string> &parts);
        static void RelaceString(std::string &input, const std::string &delim, const std::string& repstr);
};


int min(int a, int b);

#if defined(__linux__) || defined(__APPLE_CC__)
#include <mm_malloc.h>
#else
void* _mm_malloc(size_t sz, size_t align);
void _mm_free(void* ptr);
#endif

bool judge_android7_opencl();
unsigned short hs_floatToHalf(float f);
int hs_halfToFloatRep(unsigned short c);
float hs_halfToFloat(unsigned short c);
