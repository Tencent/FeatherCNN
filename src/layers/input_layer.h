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

#include "../feather_simple_generated.h"
#include "../layer.h"

#include <assert.h>
#include <stdio.h>

#include <map>

namespace feather
{
class InputLayer : public Layer
{
    public:
        InputLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
            : Layer(layer_param, rt_param)
        {
            //From proto
            const InputParameter *input_param = layer_param->input_param();
            size_t input_num = VectorLength(input_param->name());
            size_t input_dim_num = VectorLength(input_param->dim());
            assert(input_num > 0);
            assert(input_dim_num == input_num * 4);
            for (int i = 0; i < input_num; ++i)
            {
                size_t num = input_param->dim()->Get(i * 4);
                size_t channels = input_param->dim()->Get(i * 4 + 1);
                size_t height = input_param->dim()->Get(i * 4 + 2);
                size_t width = input_param->dim()->Get(i * 4 + 3);

                std::string input_name = input_param->name()->Get(i)->str();
                _top.push_back(input_name);
                _top_blobs[input_name] = new Blob<float>(num, channels, height, width);
                _top_blobs[input_name]->Alloc();
                //_top_blobs[input_name]->PrintBlobInfo();
                printf("input_name %s (n c h w)=(%ld %ld %ld %ld)\n", input_name.c_str(), num, channels, height, width);
            }
        }

        int Init()
        {
            return 0;
        }

        int CopyInput(std::string name, float *input_data)
        {
            _top_blobs[name]->CopyData(input_data);
            return 0;
        }

        bool IsInputBlob(std::string input_blob_name)
        {
            // return (input_blob_map.find(input_blob_name) == input_blob_name.end()) ? false : true;
            return false;
        }

        size_t input_size()
        {
            return _top_blobs.size();
        }

        const Blob<float> *input_blob(std::string name)
        {
            return _top_blobs[name];
        }

        std::string input_name(int idx)
        {
            auto it = _top_blobs.begin();
            for (int i = 0; i < idx; ++i)
            {
                ++it;
            }
            return it->first;
        }
};
};
