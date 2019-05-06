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

#include "../layer.h"

namespace feather
{
class SplitLayer : public Layer
{
    public:
        SplitLayer(RuntimeParameter<float>* rt_param)
            : Layer(rt_param)
        {

        }

        int Reshape()
        {
            size_t n = bottoms[0]->num();
            size_t c = bottoms[0]->channels();
            size_t h = bottoms[0]->height();
            size_t w = bottoms[0]->width();
            for (int i = 0; i < tops.size(); ++i)
            {
                tops[i]->ReshapeWithRealloc(n, c, h, w);
            }
            return 0;
        }

        int Forward()
        {
            float* src_data = bottoms[0]->data();
            size_t data_size = bottoms[0]->data_size();
            
            for (int i = 0; i < tops.size(); ++i)
            {
                memcpy(tops[i]->data(), src_data, sizeof(float) * data_size);
            }
            return 0;
        }
};
};