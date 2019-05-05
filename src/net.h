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

#include "layer.h"
#include "rt_param.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <sstream>

#include <ncnn/mat.h>

namespace feather
{
class Net
{
    public:
        Net();

        ~Net();

        int LoadParam(const char * param_path);
        int LoadParam(FILE* fp);
        int LoadWeights(const char * weights_path);
        int LoadWeights(FILE* fp);
        
        // int FeedInput(const char* input_name, const int w, const int h, const int c, const float* input_data);
        
        int FeedInput(const char* input_name, ncnn::Mat& in);

        int Forward();
          
        int Extract(std::string blob_name, float** output_ptr, int* n, int *c, int* h, int* w);

        int Extract(std::string blob_name, ncnn::Mat& out);
        
        int BuildBlobMap();
        
        std::map<std::string, Blob<float> *> blob_map;
    
    private:
        int InitLayers();
        int Reshape();
        RuntimeParameter<float> *rt_param;
        std::vector<Layer *> layers;

        /* Flag varibles indicating Net status.
         * 
         * _weights_loaded: if Net has loaded the weights.
         * _net_initialized: if the weights are already initialized.
         */
        int _param_loaded;
        int _weights_loaded;
        int _net_initialized;
};
}; // namespace feather
