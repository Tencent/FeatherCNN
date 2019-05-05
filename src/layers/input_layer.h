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
#include <assert.h>
#include <stdio.h>

#include <map>

namespace feather
{
class InputLayer : public Layer 
{
    public:
        InputLayer(RuntimeParameter<float>* rt_param)
            : Layer(rt_param)
        {
        }

        int LoadParam(const ncnn::ParamDict& pd)
        {
            int w = pd.get(0, 0);
            int h = pd.get(1, 0);
            int c = pd.get(2, 0);
            // this->tops[0]->ReshapeWithRealloc(1, c, h, w);
            return 0;
        }

        int Reshape()
        {
            // Nothing to do, don't call base class version.
            return 0;
        }

        int Init()
        {
            return 0;
        }
};
};
