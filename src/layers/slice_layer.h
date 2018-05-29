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

namespace feather
{
class SliceLayer : public Layer
{
    public:
        SliceLayer(const LayerParameter* layer_param, const RuntimeParameter<float>* rt_param)
            : axis(0),
              Layer(layer_param, rt_param)
        {
            printf("+++slice layer++\n");
            axis = layer_param->slice_param()->axis();
            if (axis < 0)
            {
                axis = 4 + axis;
            }
            for (int i = 0; i < VectorLength(layer_param->slice_param()->slice_point()); ++i)
            {
                this->slice_point.push_back(layer_param->slice_param()->slice_point()->Get(i));
            }
        }

        int GenerateTopBlobs();
        int Forward();
        int Init();

    private:
        int axis;
        std::vector<size_t> slice_point;
};
};
