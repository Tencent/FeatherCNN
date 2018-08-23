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
class ReshapeLayer : public Layer
{
    public:
        ReshapeLayer(const LayerParameter* layer_param, const RuntimeParameter<float>* rt_param)
              : Layer(layer_param, rt_param)
        {
           dim[0] = layer_param->reshape_param()->shape()->dim()->Get(0);
           dim[1] = layer_param->reshape_param()->shape()->dim()->Get(1);
           dim[2] = layer_param->reshape_param()->shape()->dim()->Get(2);
           dim[3] = layer_param->reshape_param()->shape()->dim()->Get(3);
	printf("dim %d %d %d %d\n", dim[0], dim[1], dim[2], dim[3]);
        }

        int GenerateTopBlobs();
        int Forward();
        int ForwardReshape();
        int Init();

    private:
        int    dim[4];	
	
//        int    num_output;
//        float* select_weights;
};
};
