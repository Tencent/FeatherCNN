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

#include <arm_neon.h>

void print_vec2(float32x4_t* vp);
void print_vec3(float32x4_t* vp);
void print_vec(float32x4_t* vp, const char* comment);
void print_vec(float32x4_t* vp);
void print_arr(float* vp);
void print_floats(const float* arr, const int len);

void print_floats(const float* arr, const int dimX, const int dimY);

void diff(float* arr1, float* arr2, int len);
void diff(float* arr1, float* arr2, int M, int N);

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
