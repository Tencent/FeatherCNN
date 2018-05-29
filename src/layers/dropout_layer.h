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
class DropoutLayer : public Layer
{
    public:
        DropoutLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
            : Layer(layer_param, rt_param)
        {
        }

        int GenerateTopBlobs()
        {
            //Inplace layer, do nothing
            _top_blobs[_top[0]] = const_cast<Blob<float>*>(_bottom_blobs[_bottom[0]]);
            return 0;
        }

        int Init()
        {
            //Inplace layer, the top points to its own bottom.
            return 0;
        }


        int Forward()
        {
            //Nothing to do at test phase
            return 0;
        }
};
};
