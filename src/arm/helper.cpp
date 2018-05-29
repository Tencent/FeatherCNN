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

#include "helper.h"

#include <stdio.h>
#include <stdio.h>
#include <arm_neon.h>
#include <math.h>

void print_vec2(float32x4_t* vp)
{
    float* ep = (float *) vp;
    printf("input %.1f, %.1f, %.1f, %.1f\n", *(ep), *(ep + 1), *(ep + 2), *(ep + 3));
}

void print_vec3(float32x4_t* vp)
{
    float* ep = (float *) vp;
    printf("transformed %.1f, %.1f, %.1f, %.1f\n", *(ep), *(ep + 1), *(ep + 2), *(ep + 3));
}

void print_vec(float32x4_t* vp, const char* comment)
{
    float* ep = (float *) vp;
    printf("%s %.3f, %.3f, %.3f, %.3f\n", comment, *(ep), *(ep + 1), *(ep + 2), *(ep + 3));
}


void print_vec(float32x4_t* vp)
{
    float* ep = (float *) vp;
    printf("vec %.1f, %.1f, %.1f, %.1f\n", *(ep), *(ep + 1), *(ep + 2), *(ep + 3));
}

void print_arr(float* vp)
{
    float* ep = (float *) vp;
    printf("arr %.1f, %.1f, %.1f, %.1f\n", *(ep), *(ep + 1), *(ep + 2), *(ep + 3));
}

void print_floats(const float* arr, const int len)
{
    for (int i = 0; i < len; ++i)
    {
        printf("%.2f ", arr[i]);
    }
    printf("\n\n");
}

void print_floats(const float* arr, const int dimX, const int dimY)
{
    for (int i = 0; i < dimX; ++i)
    {
        for (int j = 0; j < dimY; ++j)
            printf("%.2f ", arr[i * dimY + j]);
        printf("\n");
    }
    printf("\n\n");
}


void diff(float* arr1, float* arr2, int len)
{
    float dif = 0.0f;
    for (int i = 0; i < len; ++i)
    {
        float err = fabsf(arr1[i] - arr2[i]);
        if (err > 1.0f)
        {
            dif += err;
        }
    }
    printf("The difference is %.2f\n", dif);
}
void diff(float* arr1, float* arr2, int M, int N)
{
    float dif = 0.0f;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float err = fabsf(arr1[i * N + j] - arr2[i * N + j]);
            if (err > 1.0f)
            {
                dif += err;
                printf("Error position (%d, %d), value %.2f, %.2f\n", i, j, arr1[i * N + j], arr2[i * N + j]);
            }
        }
    }
    printf("The difference is %.2f\n", dif);
}

#include <time.h>

void Timer::startBench()
{
    clock_gettime(CLOCK_MONOTONIC, &start);
}

void Timer::endBench(const char* comment)
{
    clock_gettime(CLOCK_MONOTONIC, &stop);
    double elapsedTime = (stop.tv_sec - start.tv_sec) * 1000.0 + (stop.tv_nsec - start.tv_nsec) / 1000000.0;
    printf("%s %lfms\n", comment, elapsedTime);
}

void Timer::endBench(const char* comment, double fold)
{
    clock_gettime(CLOCK_MONOTONIC, &stop);
    double elapsedTime = (stop.tv_sec - start.tv_sec) * 1000.0 + (stop.tv_nsec - start.tv_nsec) / 1000000.0;
    printf("%s %lfms\n", comment, elapsedTime / fold);
}
