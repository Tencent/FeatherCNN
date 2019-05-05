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

#include <math.h>
#include <limits>

#define MAX(a,b) ((a)>(b))?(a):(b)
#define MIN(a,b) ((a)<(b))?(a):(b)

namespace feather
{
class PoolingLayer : public Layer
{
    public:
        PoolingLayer(RuntimeParameter<float>* rt_param)
            : stride_h(1),
              stride_w(1),
              Layer(rt_param)
        {
        }


        int Forward()
        {
            const float *input = bottoms[0]->data();
            float *output = tops[0]->data();
            float *p = output;

            int slot = input_channels * output_h;

            #pragma omp parallel for schedule(static) num_threads(num_threads)
            for (int i = 0; i < input_channels; ++i)
            {
                for (int j = 0; j < output_h; j ++)
                {
                    // int i=slot/output_h,  j=slot%output_h;
                    float *p = output + i * output_h * output_w + j * output_w;
                    for (int l = 0; l < output_w; l++) 
                        p[l] = (this->pooling_type != 0 ? 0 : -1 * std::numeric_limits<float>::max());

                    int tmp_pos = j * stride_h - pad_top - pad_bottom;
                    int x_min = MAX(tmp_pos, 0);
                    int x_max = MIN((int)(tmp_pos + kernel_h), (int) input_h);

                    for (int k = 0; k < output_w; k ++)
                    {
                        int counter = 0;
                        float total = (this->pooling_type != 0 ? 0 : -1 * std::numeric_limits<float>::max());
                        for (int x = x_min; x < x_max; ++x)
                        {
                            int xpos = i * input_h * input_w + x * input_w;
                            int local_pos = k * stride_w - pad_left - pad_right;
                            int y_min     = MAX(local_pos, 0);
                            int y_max     = MIN((int)(local_pos + kernel_w), (int) input_w);

                            for (int y = y_min; y < y_max; ++y)
                            {
                                float value = input[xpos + y];
                                if (this->pooling_type != 0)
                                    total += value, counter++;
                                else
                                total = total > value ? total : value;
                            }
                        }
                        if (this->pooling_type != 0)
                            p[k] += total / (counter);
                        else
                            p[k]  = (p[k] > total) ? p[k] : total;
                    }
                }
            }
            return 0;
        }

        int ForwardReshape()
        {
            const Blob<float> *bottom_blob = bottoms[0];
            input_h = bottom_blob->height();
            input_w = bottom_blob->width();
            input_channels = bottom_blob->channels();
            //printf("layer %s\n", _name.c_str());
            //printf("input %lu %lu %lu\n", input_channels, input_h, input_w);
            if (global_pooling)
            {
                kernel_h = input_h;
                kernel_w = input_w;
                output_h = 1;
                output_w = 1;
                output_channels = input_channels;
            }
            else
            {
                //General pooling.
                output_channels = input_channels;
                output_h = static_cast<int>(ceil(static_cast<float>(input_h + pad_top + pad_bottom - kernel_h) / stride_h)) + 1;
                output_w = static_cast<int>(ceil(static_cast<float>(input_w + pad_left + pad_right - kernel_w) / stride_w)) + 1;
            }
            tops[0]->ReshapeWithRealloc(1, output_channels, output_h, output_w);
            return Forward();
        }

        int LoadParam(const ncnn::ParamDict& pd)
        {
            pooling_type = pd.get(0, 0); //Pooling type?
            kernel_w = pd.get(1, 0);
            kernel_h = pd.get(11, kernel_w);
            stride_w = pd.get(2, 1);
            stride_h = pd.get(12, stride_w);
            pad_left = pd.get(3, 0);
            pad_right = pd.get(14, pad_left);
            pad_top = pd.get(13, pad_left);
            pad_bottom = pd.get(15, pad_top);
            global_pooling = pd.get(4, 0);
            tf_pad_mode = pd.get(5, 0);
            // printf("$$ global_pooling %d\n", global_pooling);
            // printf("$$ padding %d %d %d %d\n", pad_left, pad_bottom, pad_right, pad_top);
            // printf("$$ stride %d %d\n", stride_h, stride_w);
            return 0;
        }

        int Reshape()
        {
            //Only accept a single bottom blob.
            const Blob<float> *bottom_blob = bottoms[0];
            input_h = bottom_blob->height();
            input_w = bottom_blob->width();
            input_channels = bottom_blob->channels();
            // printf("$$ input %d %d %d\n", input_channels, input_h, input_w);
            if (global_pooling)
            {
                kernel_h = input_h;
                kernel_w = input_w;
                output_h = 1;
                output_w = 1;
                output_channels = input_channels;
            }
            else
            {
                //General pooling.
                output_channels = input_channels;
                output_h = static_cast<int>(ceil(static_cast<float>(input_h + pad_top + pad_bottom - kernel_h) / stride_h)) + 1;
                output_w = static_cast<int>(ceil(static_cast<float>(input_w + pad_left + pad_right - kernel_w) / stride_w)) + 1;
            }
            this->tops[0]->ReshapeWithRealloc(1, output_channels, output_h, output_w);
            return 0;
        }

    private:
        int input_h;
        int input_w;
        int input_channels;
        int output_h;
        int output_w;
        int output_channels;
        int pad_left;
        int pad_bottom;
        int pad_right;
        int pad_top;
        int kernel_h;
        int kernel_w;
        int stride_h;
        int stride_w;
        bool global_pooling;
        int pooling_type;
        int tf_pad_mode;
};
};
