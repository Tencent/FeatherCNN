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
#include <math.h>
#include <string>
#include <string.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace feather;

void SplitString(const std::string &input, const std::string &delim, std::vector<std::string> &parts)
{
    for (char *s = strtok((char *)input.data(), (char *)delim.data()); s; s = strtok(NULL, (char *)delim.data()))
    {
        if (s != NULL)
        {
            parts.push_back(s);
        }
    }
}

void PrintBlobData(feather::Net<float> *forward_net, std::string blob_name, int n)
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
    printf("\n");
    free(arr);
}


void testReshape(const std::string& model_path)
{

    int height = 2038;
    int width = 1134;
    int input_size = height * width * 3;
    feather::Net<float> forward_net_cpu(1, DeviceType::CPU);

    forward_net_cpu.InitFromPath(model_path.c_str());

    float* input = (float*)malloc(sizeof(float) * input_size);
    for(int i = 0; i < input_size; ++i)
    {
      input[i] = (rand() % 256 - 127.5) / 128;
    }


    float factor = 0.6;
    std::vector<float> scales_;
    int minHW = 1134;
    float s0 = 12 / static_cast<float>(minHW);
    float s = s0;
    while (s < 0.25){
    	scales_.push_back(s);
	s *= 1 / factor;
    }

    for(int i = 0; i < scales_.size(); i++){
	float scale = scales_[i];
	int hs = static_cast<int>(height * scale);
	int ws = static_cast<int>(width * scale);
    	forward_net_cpu.Forward(input, hs, ws);
    	printf("loop count %d/n", i);
    }

    free(input);
}


void test(std::string model_path, std::string data_path, int loop, int num_threads, std::string results=std::string(""))
{
    printf("++++++Start Loader++++++\n");
    feather::Net<float> forward_net(num_threads, DeviceType::CPU);
    forward_net.InitFromPath(model_path.c_str());
    size_t input_size = 576 * 576 * 3 ;
    float *input = new float[input_size * 20];
    std::ifstream in(data_path.c_str());
    std::string line;
    std::string delim = "\t\r\n <>()";
    size_t count = 0;
    double time = 0;

    while (getline(in, line))
    {
        for (int i = 0; i < 1; ++i)
        {
            std::vector<std::string> parts;
            SplitString(line, delim, parts);
            printf("input size %ld parts size %ld\n", input_size, parts.size());
            for (size_t i = 0; i != parts.size(); ++i)
            {
                input[i] = atof(parts[i].c_str());
            }
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
            printf("--------Average runtime %lfmsi------\n", time / (loop - 1) / 1000.0);
	    //if(results.length()>0)
	    //	ResultEvaluate(&forward_net, "prob", 0, results);

            // If you want to dump all the data
            forward_net.DumpBlobMap();
            //PrintBlobData(&forward_net, "v/simplenet/flatten:0", 10);
        }
        break;
    }

    if (input)
    {
        delete [] input;
        input = NULL;
    }
}

int main(int argc, char* argv[])
{
    if (argc == 5)
    {
        size_t num_threads = atoi(argv[4]);
        size_t loop = atoi(argv[3]);
        test(std::string(argv[1]), std::string(argv[2]), loop, num_threads);
        //testReshape(argv[1]); 
    }
    else if(argc == 6)
    {
        size_t num_threads = atoi(argv[4]);
        size_t loop = atoi(argv[3]);
        test(std::string(argv[1]), std::string(argv[2]), loop, num_threads, argv[5]);
    }
    else
    {
        fprintf(stderr, "Usage: ./testRun [feathermodel] [input_data] [loop_count] [num_threads]\n");
        return 0;
    }
    return 0;
}
