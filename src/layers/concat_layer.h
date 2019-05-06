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
class ConcatLayer : public Layer
{
    public:
        ConcatLayer(RuntimeParameter<float>* rt_param)
            : Layer(rt_param)
        {

        }
        
        int load_param(const ncnn::ParamDict& pd)
        {
            this->axis = pd.get(0, 0);
            return 0;
        }

        int Forward()
        {
            float* top_ptr = tops[0]->data();
            for (int i = 0; i < bottoms.size(); ++i)
            {
                const float* bottom_ptr = bottoms[i]->data();
                memcpy(top_ptr, bottom_ptr, sizeof(float) * bottoms[i]->data_size());
                top_ptr += sizeof(float) * bottoms[i]->data_size();
            }
            return 0;
        }

        int Reshape()
        {
            auto first_blob = this->bottoms[0];
            size_t num = 1;
            size_t channels = first_blob->channels();
            size_t width = first_blob->width();
            size_t height = first_blob->height();

            for (int i = 1; i < bottoms.size(); ++i)
            {
                auto p_blob = bottoms[i];
                if (this->axis == 0)
                {
                    if(!(width == p_blob->width() && height == p_blob->height()))
                    {
                        printf("Images of different shapes cannot be concatenated together\n");
                        return -100;
                    }
                    channels += p_blob->channels();
                }
                else
                {
                    LOGE("FeatherCNN only supports concat at axis = 0.");
                    return -100;
                }
            }
            // printf("Concat output shape %d %d %d\n", channels, height, width);
            tops[0]->ReshapeWithRealloc(1, channels, height, width);
            return 0;
        }
    private:
        int axis;
};
};
