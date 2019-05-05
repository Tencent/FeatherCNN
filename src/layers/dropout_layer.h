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
class DropoutLayer : public Layer
{
    public:
        DropoutLayer(RuntimeParameter<float>* rt_param)
            : Layer(rt_param)
        {
            _inplace = false;
        }

        int LoadParam(const ncnn::ParamDict &pd)
        {
            scale = pd.get(0, 1.f);
            return 0;
        }

        int Forward()
        {
            if (scale == 1.f)
            {
                memcpy(tops[0]->data(), bottoms[0]->data(), bottoms[0]->data_size() * sizeof(float));
            }
            else
            {
                int w = bottoms[0]->width();
                int h = bottoms[0]->height();
                int channels = bottoms[0]->channels();
                int size = w * h;

                float* inp = bottoms[0]->data();
                float* outp = tops[0]->data();
                for (int i = 0; i < bottoms[0]->data_size(); ++i)
                {
                        outp[i] = inp[i] * scale;
                }
            }
            return 0;
        }
    private:
        float scale;
};
};