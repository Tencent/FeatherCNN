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

#include "../feather_generated.h"
#include "../layer.h"

namespace feather
{
class LeakyReluLayer : public Layer<float>
{
    public:
        LeakyReluLayer(const LayerParameter* layer_param, RuntimeParameter<float>* rt_param)
            : Layer<float>(layer_param, rt_param)
        {
            float* blob_data = this->weight_blob(0)->data();
            this->alpha = blob_data[0];
        }
        int Forward()
        {
            const Blob<float> *p_bottom = _bottom_blobs[_bottom[0]];
            const float *input = p_bottom->data();
            const size_t data_size = p_bottom->num() * p_bottom->channels() * p_bottom->height() * p_bottom->width();

            float *output = _top_blobs[_top[0]]->data();
            for (size_t i = 0; i < data_size; ++i)
            {
                float mult = this->alpha * input[i];
                output[i] = input[i] > mult ? input[i] : mult;
            }
            return 0;
        }
        float alpha;
};
};
