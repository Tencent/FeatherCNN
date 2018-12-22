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

#include <net.h>

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <type_traits>

using namespace std;
using namespace feather;

template<typename Dtype>
void PrintBlobData(feather::Net<Dtype> *forward_net, const std::string& blob_name, size_t n)
{
    size_t data_size;
    forward_net->GetBlobDataSize(&data_size, blob_name);
    float *arr = (float*)malloc(sizeof(float) * data_size);
    forward_net->ExtractBlob(arr, blob_name);
    size_t len = std::min(data_size, n);

    for (int i = 0; i < len; ++i)
    {
        printf("%f ", arr[i]);
    }
    puts("");
    free(arr);
}

template <typename Dtype>
void DiffBlobData(feather::Net<float>* cpu_net, feather::Net<Dtype>* gpu_net, const std::string& cpu_blob_name, const std::string& gpu_blob_name)
{
    size_t cpu_data_size = 0, gpu_data_size = 0;
    cpu_net->GetBlobDataSize(&cpu_data_size, cpu_blob_name);
    gpu_net->GetBlobDataSize(&gpu_data_size, gpu_blob_name);
    assert(cpu_data_size == gpu_data_size);
    size_t data_size = cpu_data_size;
    
    float* cpu_data = (float*)malloc(sizeof(float) * data_size);
    cpu_net->ExtractBlob(cpu_data, cpu_blob_name);
    float* gpu_data = (float*)malloc(sizeof(float) * data_size);
    gpu_net->ExtractBlob(gpu_data, gpu_blob_name);
   

    float threshold = std::is_same<Dtype, uint16_t>::value ? 1.0f : 0.001f;
    float diff_sum = 0.f;
    for(int i = 0; i < data_size; ++i)
    {
        float cur_diff = fabs(cpu_data[i] - gpu_data[i]);
        if (cur_diff > threshold)
        {
            printf("Diff %d %f %f\n", i, cpu_data[i], gpu_data[i]);
	    diff_sum += cur_diff;
        }
    }
    printf("diff_sum %f\n", diff_sum);
}

template<typename Dtype>
void testPerf(const std::string& model_path, int input_size, const std::string& output_name, int loop_count, DeviceType type)
{
    std::string device_str = type == DeviceType::CPU ? "CPU" : "GPU";
    std::string type_str = std::is_same<Dtype, uint16_t>::value ? "half" : "float";
    printf("-------------------------test%sPerf<%s>--------------------------\n", device_str.c_str(), type_str.c_str());
    feather::Net<Dtype> forward_net(1, type);
    forward_net.InitFromPath(model_path.c_str());
    float* input = (float*)malloc(sizeof(float) * input_size);
    for(int i = 0; i < input_size; i++)
    {
       input[i] = (i % 256 - 127.5) / 128;
    }
    
    double total_time = 0;
    int timing_count = 0;
    timespec tpstart, tpend;
    for(int i = 0; i < loop_count; i++)
    {
        clock_gettime(CLOCK_MONOTONIC, &tpstart);
        forward_net.Forward(input);
        PrintBlobData(&forward_net, output_name, 10);
        clock_gettime(CLOCK_MONOTONIC, &tpend);
        double time = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
        time = time / 1000.0;
        printf("Prediction costs %lfms\n", time);
	if (time > 400) continue;
        if (i > 0) {
            total_time += time;
	    ++timing_count;
	}
    }

    if (timing_count > 0) {
   	printf("Average %s runtime %lfms\n", device_str.c_str(), total_time / timing_count);
    } else {
	printf("Average runtime > 400ms!\n");
    }

    free(input);
}

template <typename Dtype>
void testDiff(const std::string& model_path, int input_size, const std::string& cpu_blob_name, const std::string& gpu_blob_name)
{
    std::string type_str = std::is_same<Dtype, uint16_t>::value ? "half" : "float";
    printf("\n--------------------------testDiff<%s>---------------------------\n", type_str.c_str());
    feather::Net<float> forward_net_cpu(1, DeviceType::CPU);
    feather::Net<Dtype> forward_net_gpu(1, DeviceType::GPU_CL);

    forward_net_cpu.InitFromPath(model_path.c_str());
    forward_net_gpu.InitFromPath(model_path.c_str());

    float* input = (float*)malloc(sizeof(float) * input_size);
    for(int i = 0; i < input_size; ++i)
    {
      input[i] = (rand() % 256 - 127.5) / 128;
    }
    forward_net_cpu.Forward(input);
    forward_net_gpu.Forward(input);

    DiffBlobData<Dtype>(&forward_net_cpu, &forward_net_gpu, cpu_blob_name, gpu_blob_name);

    free(input);
}

template <typename Dtype>
void testGPU(const std::string& model_path, int input_size, const std::string& cpu_blob_name, const std::string& gpu_blob_name, int loop_count = 2)
{
    testDiff<Dtype>(model_path, input_size, cpu_blob_name, gpu_blob_name);
    testPerf<Dtype>(model_path, input_size, gpu_blob_name, loop_count, DeviceType::GPU_CL);
}

int main(int argc, char* argv[])
{
    if (argc >= 6)
    {
        std::string model_path = std::string(argv[1]);
	int input_channels = atoi(argv[2]);
	int input_height = atoi(argv[3]);
        int input_width = atoi(argv[4]);
        std::string cpu_blob_name = std::string(argv[5]); // ReLU is always fused in the GPU conv/fc layers, 
        std::string gpu_blob_name = std::string(argv[6]); // which is not always the case for CPU conv/fc layers.
        int loop_count = argc > 7 ? atoi(argv[7]) : 2;
	int input_size = input_channels * input_height * input_width;
        testPerf<float>(model_path, input_size, cpu_blob_name, loop_count, DeviceType::CPU);
        testGPU<float>(model_path, input_size, cpu_blob_name, gpu_blob_name, loop_count);
        testGPU<uint16_t>(model_path, input_size, cpu_blob_name, gpu_blob_name, loop_count);
    }
    else
    {
        fprintf(stderr, "Usage: ./feather_test [model_path] [input_channels] [input_height] [input_width] [cpu_blob_name] [gpu_blob_name] [loop_count]\n");
	fprintf(stderr, "Example: ./feather_test nobn.feathermodel 3 224 192 \"fc5_classification_relu\" \"fc5_classification\" 100\n");
        return 0;
    }
    return 0;
}
