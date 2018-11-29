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

#include "layer.h"
#include "rt_param.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <sstream>
#include <sys/system_properties.h>

namespace feather
{

bool judge_android7_opencl()
{
    #include <sys/system_properties.h>
    //libOpenCL.so
    //android7.0 sdk api 24
    char sdk[93] = "";
    __system_property_get("ro.build.version.sdk", sdk);
    //printf("sdk_version [%s]\n", sdk);
    if (std::atoi(sdk) < 24) 
    {
        printf("find sdk [%d] < 24\n", std::atoi(sdk));
        return true;
    }

    std::string libOpenCL_str = "libOpenCL.so";
    std::vector<std::string> libraries_list;
    libraries_list.push_back("/vendor/etc/public.libraries.txt");
    libraries_list.push_back("/system/etc/public.libraries.txt");
    for(int i = 0; i < libraries_list.size(); i++)
    {
        ifstream out;
        std::string line;
        //printf("file [%s]\n", libraries_list[i].c_str());
        out.open(libraries_list[i].c_str());
        while(!out.eof()){
            std::getline(out, line);
            //printf("line [%s]\n", line.c_str());
            if(line.find(libOpenCL_str) != line.npos)
            {
                printf("find line [%s]\n", line.c_str());
                return true;
            }

        }
        out.close();
    }
    return false;
}

class Net
{
    public:
        Net(size_t num_threads, DeviceType device_type);

        ~Net();
        void InitFromPath(const char *model_path);
        void InitFromStringPath(std::string model_path);
        void InitFromFile(FILE *fp);
        void InitFromBuffer(const void *net_buffer);
        bool InitFromBufferCPU(const void *net_buffer);

        int  Forward(float* input);
        int  Forward(float* input, int height, int width);
        int RemoveLayer(Layer<float>* layer);
        int GenLayerTops();
        void TraverseNet();
        int GetBlobDataSize(size_t* data_size, std::string blob_name);
        int PrintBlobData(std::string blob_name);
        int ExtractBlob(float* output_ptr, std::string blob_name);
        std::map<std::string, const Blob<float> *> blob_map;
#ifdef FEATHER_OPENCL
        //int test_opencl();
        bool InitFromBufferCL(const void *net_buffer);
        int RemoveLayer(Layer<uint16_t>* layer);
        std::map<std::string, const Blob<uint16_t> *> blob_map_cl;
#endif
      private:
        std::vector<Layer<float>* > layers;
#ifdef FEATHER_OPENCL
        std::vector<Layer<uint16_t>* > layers_cl;
        // std::map<std::string, const Blob<uint16_t>* >  blob_map_cl;
#endif
        RuntimeParameter<float> *rt_param;
};
};
