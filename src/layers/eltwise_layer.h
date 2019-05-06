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
class EltwiseLayer : public Layer
{
    public:
        EltwiseLayer(RuntimeParameter<float>* rt_param)
            : Layer(rt_param),
              op_type(1),
              fuse_relu(0)
        {
        }

        int Reshape()
        {
            size_t n = bottoms[0]->num();
            size_t c = bottoms[0]->channels();
            size_t h = bottoms[0]->height();
            size_t w = bottoms[0]->width();
            // Check bottom shapes
            for (int i = 1; i < bottoms.size(); ++i)
            {
                if ((n != bottoms[i]->num()) || (c != bottoms[i]->channels()) || (h != bottoms[i]->height()) || (w != bottoms[i]->width()))
                {
                    LOGE("Shape mismatch among bottoms of layer %s.", this->name.c_str());
                    return -100;
                }
            }
            for (int i = 0; i < tops.size(); ++i)
            {
                tops[i]->ReshapeWithRealloc(n, c, h, w);
            }
            return 0;
        }

        int LoadParam(const ncnn::ParamDict &pd)
        {
            op_type = pd.get(0, 0);
            ncnn::Mat coeffs = pd.get(1, ncnn::Mat());
            if (!coeffs.empty())
            {
                LOGE("FeatherCNN doesn't support coeffs in eltwise layer. Please refer to ncnn.");
                return -100;
            }
            if (op_type != 1)
            {
                LOGE("FeatherCNN doesn't support ops rather than SUM. Please refer to ncnn.");
                return -100;
            }
            return 0;
        }

        int Forward()
        {
            float* input_alpha = bottoms[0]->data();
            float* input_beta = bottoms[1]->data();
            float* output_data = tops[0]->data();
            size_t data_size = bottoms[0]->data_size();

            if (fuse_relu)
                booster::add_relu<true>(output_data, input_alpha, input_beta, data_size, 1);
            else
                booster::add_relu<false>(output_data, input_alpha, input_beta, data_size, 1);
            return 0;
        }
        
        enum { Operation_PROD = 0, Operation_SUM = 1, Operation_MAX = 2 };

    private:
        int op_type;
        int fuse_relu;

};
};