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

//
void PrintBlobData(feather::Net *forward_net, std::string blob_name, int n)
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
        printf("%f\n", arr[i]);
    }
    free(arr);
}

void test(std::string model_path, std::string data_path, int loop, int num_threads)
{
    printf("++++++Start Loader++++++\n");

    feather::Net forward_net(num_threads, DeviceType::GPU_CL);
    //forward_net.test_opencl();
    // printf("done initialization\n");
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
    //
    // // //TODO judge file size
    // // size_t file_size = 0;
    // // FILE* fp = fopen(data_path.c_str(), "rb+");
    // // fseek(fp, 0, SEEK_END);
    // // file_size = ftell(fp);
    // // fseek(fp, 0, SEEK_SET);
    // // if(file_size < input_size)
    // // {
	  // //   fprintf(stderr, "Loading input file smaller than specified size %zu\n", file_size);
	  // //   exit(6);
    // // }
    // // size_t bytes = fread(input, sizeof(float), input_size, fp);
    // // //assert(bytes == input_size * sizeof(float));
    // // if(bytes < input_size)
    // // {
	  // //   fprintf(stderr, "Loading fewer bytes, expected %zu\n", file_size);
	  // //   exit(6);
    // // }
    // // fclose(fp);
    printf("forward begin\n");
    for (int i = 0; i < loop; ++i)
    {

	    timespec tpstart, tpend;
	    clock_gettime(CLOCK_MONOTONIC, &tpstart);
	    forward_net.Forward(input);
	    clock_gettime(CLOCK_MONOTONIC, &tpend);
	    double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
	    printf("Prediction costs %lfms\n", timedif / 1000.0);
	    if (i > 0)
		    time += timedif;
    }
    printf("--------Average runtime %lfms------\n", time / (loop - 1) / 1000.0);
    // //PrintBlobData(forward_net, "fc6", 0);
    PrintBlobData(&forward_net, "tx_pose/stage1/branch1/conv5_5_CPM_L2/bias_add:0", 10);
    // //PrintBlobData(forward_net, "data", 100);
    // //printf("------------------------\n");
    // // PrintBlobData(forward_net, "FeatureExtractor/MobilenetV2/Conv/Conv2D:0", 20);
    // // printf("------------------------\n");
    // // PrintBlobData(forward_net, "FeatureExtractor/MobilenetV2/expanded_conv/depthwise/depthwise:0", 20);
    // //printf("%f, %f\n", input[0], input[1]);
    // //printf("%f, %f\n", input[300], input[301]);
    // //printf("%f, %f\n", input[90000], input[90001]);
    // //printf("%f, %f\n", input[90300], input[90301]);
    // //printf("%f, %f\n", input[180000], input[180001]);
    // //printf("%f, %f\n", input[180300], input[180301]);
    // if (input)
    // {
    //     delete [] input;
    //     input = NULL;
    // }
    // printf("ok so far\n");
    // delete forward_net;
    // printf("done...\n");
}
int main(int argc, char* argv[])
{
    if (argc == 5)
    {
        size_t num_threads = atoi(argv[4]);
        size_t loop = atoi(argv[3]);
        test(std::string(argv[1]), std::string(argv[2]), loop, num_threads);
    }
    else
    {
        fprintf(stderr, "Usage: ./testRun [feathermodel] [input_data] [loop_count] [num_threads]\n");
        return 0;
    }
    return 0;
}
