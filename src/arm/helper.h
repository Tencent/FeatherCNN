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

#if __ARM_NEON
#include <arm_neon.h>

#ifdef FEATHER_ANDROID_LOG
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "FeatherLib", __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "FeatherLib", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "FeatherLib", __VA_ARGS__)
#else
#include <stdio.h>
#define LOGI(...) fprintf(stdout, __VA_ARGS__)
#define LOGD(...) fprintf(stdout, __VA_ARGS__)
#define LOGE(...) fprintf(stderr, __VA_ARGS__)
#endif


void print_vec2(float32x4_t* vp);
void print_vec3(float32x4_t* vp);
void print_vec(float32x4_t* vp, const char* comment);
void print_vec(float32x4_t* vp);
void print_arr(float* vp);
void print_floats(const float* arr, const int len);

void print_floats(const float* arr, const int dimX, const int dimY);

void diff(float* arr1, float* arr2, int len);
void diff(float* arr1, float* arr2, int M, int N);

//Thanks nihui for this code snippet!
#ifndef __aarch64__

#ifndef __APPLE__
static inline float32x4_t vfmaq_f32(float32x4_t _s, float32x4_t _a, float32x4_t _b)
{
	return vmlaq_f32(_s, _a, _b);
}
#endif
static inline float32x4_t vfmaq_laneq_f32(float32x4_t _s, float32x4_t _a, float32x4_t _b, int lane)
{
	if(lane == 0)      return vmlaq_lane_f32(_s, _a, vget_low_f32(_b), 0);
	else if(lane == 1) return vmlaq_lane_f32(_s, _a, vget_low_f32(_b), 1);
	else if(lane == 2) return vmlaq_lane_f32(_s, _a, vget_high_f32(_b), 0);
	else if(lane == 3) return vmlaq_lane_f32(_s, _a, vget_high_f32(_b), 1);
	else return vdupq_n_f32(0.f);
}
#endif
#endif

#include <time.h>
class Timer
{
    public:
        Timer() {}
        virtual ~Timer() {}
        void startBench();
        void endBench(const char *commets);
        void endBench(const char *commets, double fold);
    private:
        timespec start, stop;
};
