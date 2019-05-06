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
#include <booster/generic_kernels.h>

namespace feather
{
class ScaleLayer : public Layer
{
    public:
        ScaleLayer(RuntimeParameter<float>* rt_param)
            : channels(0),
              bias_term(0),
              scale_data_size(0),
              Layer(rt_param)
        {
        }

        int LoadParam(const ncnn::ParamDict &pd)
        {
            scale_data_size = pd.get(0, 0);
            bias_term = pd.get(1, 0);
            if (scale_data_size < 0)
            {
                LOGE("feather doesn't accept negative scale data size, please use ncnn to run this model.\n");
                return -100;
            }
            return 0;
        }

        int LoadWeights(const ncnn::ModelBin &mb)
        {
            ncnn::Mat scale_mat;
            ncnn::Mat bias_mat;
            if (scale_data_size == -233)
                return 0;

            scale_mat = mb.load(scale_data_size, 1);
            if (scale_mat.empty())
                return -100;
            channels = scale_data_size;
            Blob<float> *scale_blob = new Blob<float>;
            scale_blob->ReshapeWithRealloc(1, 1, 1, channels);
            scale_blob->CopyDataFromMat(scale_mat);
            weights.push_back(scale_blob);

            if (bias_term)
            {
                bias_mat = mb.load(scale_data_size, 1);
                if (bias_mat.empty())
                    return -100;
                Blob<float> *bias_blob = new Blob<float>;
                bias_blob->ReshapeWithRealloc(1, 1, 1, channels);
                bias_blob->CopyDataFromMat(bias_mat);
                weights.push_back(bias_blob);
            }
            return 0;
        }

        int Forward()
        {
            const float *input = bottoms[0]->data();
            float *output = tops[0]->data();
            size_t stride = bottoms[0]->width() * bottoms[0]->height();
            const float* scale_data = weights[0]->data();
            const float* bias_data = NULL;
            if (bias_term)
                bias_data = weights[1]->data();
            scale_kernel(channels, stride, bias_data, scale_data, input, output, 1);
            return 0;
        }

        int Init()
        {
            // const Blob<float> *p_blob = bottoms[0];
            // input_channels = p_blob->channels();
            // input_height = p_blob->height();
            // input_width = p_blob->width();
            // printf("input %d %d %d", input_channels, input_width, input_height);
            // scale_data = _weight_blobs[0]->data();
            //printf("bias_term %d\n", _bias_term ? 1: 0);
            if (bias_term)
            {
                scale_kernel = booster::scale<true>;
            }
            else
            {
                scale_kernel = booster::scale<false>;
            }
            return 0;
        }

      private:
        size_t channels;
        int bias_term;
        int scale_data_size;

    private:
        void (*scale_kernel)(const size_t channels, const size_t stride, const  float* bias_data, const float* scale_data, const float* input, float* output, const size_t num_threads);
};
};
