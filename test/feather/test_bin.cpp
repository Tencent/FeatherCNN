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

void test(std::string model_path, std::string output_name, int loop, DeviceType type = DeviceType::GPU_CL)
{
    printf("++++++Start Loader++++++\n");
    feather::Net<Dtype> forward_net(1, type);

    forward_net.InitFromPath(model_path.c_str());

    //size_t input_size = 224 * 224 * 3 ;
    size_t input_size = 300 * 300 * 3 ;
    float *input = new float[input_size * 20];

    size_t count = 0;
    double time = 0;

    for(int i = 0; i < input_size; i++)
    {
      if (count > 255){
        count = 0;
      }
      input[i] = count;
      count++;
    }
    printf("\n\n======   forward begin %s =====\n", type == DeviceType::CPU ? "cpu":"gpu_cl");
    for (int i = 0; i < loop; ++i)
    {

	    timespec tpstart, tpend;
	    clock_gettime(CLOCK_MONOTONIC, &tpstart);
	    forward_net.Forward(input);

        //PrintBlobData(&forward_net, "multibox_head/loc_0/bias_add:0", 10);
        //PrintBlobData(&forward_net, "MobilenetV2/Conv/BatchNorm/Relu:0", 10);
        //mobilev1/conv1/Conv2D:0
        //tx_pose/stage1/branch1/conv5_5_CPM_L2/bias_add:0

        PrintBlobData(&forward_net, output_name, 10);
        
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
