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

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <iostream>
#include <fstream>
#include <assert.h>

using namespace std;
using namespace feather;


//#define Dtype float
#define Dtype uint16_t

void PrintBlobData(feather::Net<Dtype> *forward_net, std::string blob_name, int n)
{
    size_t data_size;
    forward_net->GetBlobDataSize(&data_size, blob_name);
    float *arr = (float*) malloc(sizeof(float) * data_size);
    forward_net->ExtractBlob(arr, blob_name);
    size_t len = 0;
    if (n <= 0)
        len = data_size;
    else
        len = n;

    for (int i = 0; i < len; ++i)
    {
        printf("%f ", arr[i]);
    }
    puts("");
    free(arr);
}

void DiffBlobData(feather::Net<uint16_t>* gpu_net, feather::Net<float>* cpu_net, std::string blob_name)
{
    size_t gpu_data_size, cpu_data_size;
    gpu_net->GetBlobDataSize(&gpu_data_size, blob_name);
    cpu_net->GetBlobDataSize(&cpu_data_size, blob_name);
    assert(gpu_data_size == cpu_data_size);
    size_t data_size = cpu_data_size;
    
    float *cpu_data = (float*) malloc(sizeof(float) * data_size);
    cpu_net->ExtractBlob(cpu_data, blob_name);
    float *gpu_data = (float*) malloc(sizeof(float) * data_size);
    gpu_net->ExtractBlob(gpu_data, blob_name);
    
    double diff = 0.f;
    for(int i = 0; i < data_size; ++i)
    {
        double cur_diff = fabs(cpu_data[i] - gpu_data[i]);
        if((cur_diff) > 1.f)
        // if(1)
        {
            printf("Diff %d %f %f\n", i, cpu_data[i], gpu_data[i]);
        }
        diff += (cur_diff > 1.f) ? cur_diff : 0;
    }
    printf("diff %lf\n", diff);
}

void test(std::string model_path, std::string output_name, int loop, DeviceType type = DeviceType::GPU_CL)
{
    printf("++++++Start Loader++++++\n");
    feather::Net<Dtype> forward_net(1, type);

    feather::Net<float> forward_net_cpu(1, DeviceType::CPU);

    forward_net.InitFromPath(model_path.c_str());

    forward_net_cpu.InitFromPath(model_path.c_str());

    //size_t input_size = 224 * 224 * 3 ;
    size_t input_size = 300 * 300 * 3 ;
    float *input_cpu = new float[input_size * 20];
    float *input_gpu = new float[input_size * 20];

    size_t count = 0;
    double time = 0;

    for(int i = 0; i < input_size; i++)
    {
      input_cpu[i] = (float)(count % 256);
      count++;
    }
    
    const int input_img_size = 150 * 150;
    const int input_channels = 3;
    for(int c = 0; c < input_channels; ++c)
    {
        for(int i = 0; i < input_img_size; ++i)
        {
            input_gpu[i * input_channels + c] = input_cpu[c * input_img_size + i];
        } 
    }

    printf("\n\n======   forward begin %s =====\n", type == DeviceType::CPU ? "cpu":"gpu_cl");
    for (int i = 0; i < loop; ++i)
    {

        timespec tpstart, tpend;
        clock_gettime(CLOCK_MONOTONIC, &tpstart);
        forward_net.Forward(input_cpu);

        forward_net_cpu.Forward(input_cpu);


        //PrintBlobData(&forward_net, output_name, 10);

        DiffBlobData(&forward_net, &forward_net_cpu, output_name);



        clock_gettime(CLOCK_MONOTONIC, &tpend);
        double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
        //PrintBlobData(&forward_net, "tx_pose/stage1/branch0/conv5_5_CPM_L1/bias_add:0", 10);
        printf("Prediction costs %lfms\n", timedif / 1000.0);
        if (i > 0)
            time += timedif;
    }
    printf("\n\n======   forward end %s =====\n", type == DeviceType::CPU ? "cpu":"gpu_cl");
    printf("--------Average runtime %lfms------\n", time / (loop) / 1000.0);
}

void gpu_cpu_test(std::string model_path, std::string output_name, int loop)
{
    //test(model_path, output_name, loop, DeviceType::CPU);
    test(model_path, output_name, loop, DeviceType::GPU_CL);
}

int main(int argc, char* argv[])
{
    if (argc == 4)
    {
        size_t num_threads = 1;
        size_t loop = atoi(argv[2]);
        std::string model_path = std::string(argv[1]);
        std::string output_name = std::string(argv[3]);
        gpu_cpu_test(model_path, output_name, loop);
    }
    else
    {
        fprintf(stderr, "Usage: ./testRun [feathermodel] [loop_count] [output_name]\n");
        return 0;
    }
    return 0;
}
