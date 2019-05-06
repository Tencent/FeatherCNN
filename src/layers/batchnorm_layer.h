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
class BatchNormLayer : Layer
{
    public:
        BatchNormLayer(RuntimeParameter<float>* rt_param)
            : channels(0),
              scale_bias_term(false),
              fuse_scale(false),
              fuse_relu(false),
              Layer(rt_param)
        {
            _fusible = true;
        }

        int LoadParam(const ncnn::ParamDict &pd)
        {
            this->channels = pd.get(0, 0);
            this->eps = pd.get(1, 0.f);
            return 0;
        }

        int LoadWeights(const ncnn::ModelBin &mb)
        {
            ncnn::Mat slope_data, mean_data, var_data, bias_data;
            slope_data = mb.load(channels, 1);
            if (slope_data.empty())
                return -100;

            mean_data = mb.load(channels, 1);
            if (mean_data.empty())
                return -100;

            var_data = mb.load(channels, 1);
            if (var_data.empty())
                return -100;

            bias_data = mb.load(channels, 1);
            if (bias_data.empty())
                return -100;

            Blob<float> * alpha_blob = new Blob<float>;
            Blob<float> * beta_blob = new Blob<float>;
            this->weights.push_back(alpha_blob);
            this->weights.push_back(beta_blob);
            alpha_blob->ReshapeWithRealloc(1,1,1,channels);
            beta_blob->ReshapeWithRealloc(1,1,1,channels);
            float* alpha_data = alpha_blob->data();
            float* beta_data = beta_blob->data();
            for (int i = 0; i < channels; i++)
            {
                float sqrt_var = sqrt(var_data[i] + this->eps);
                alpha_data[i] = bias_data[i] - slope_data[i] * mean_data[i] / sqrt_var;
                beta_data[i] = slope_data[i] / sqrt_var;
            }
            return 0;
        }

        int Init()
        {
            const Blob<float> *p_blob = this->bottoms[0];
            if (this->channels != p_blob->channels())
            {
                printf("Mismatch channel in layer %s, expected %d but the bottom %s has %d channels.\n", this->name.c_str(), this->channels, p_blob->name.c_str(), p_blob->channels());
                return -100;
            }
            SetKernel();

            return 0;
        }

        int Forward()
        {
            const float *input = bottoms[0]->data();
            float *output = tops[0]->data();
            float* alpha_data = weights[0]->data();
            float* beta_data = weights[1]->data();
            float* scale_data = NULL;
            float* scale_bias_data = NULL;
            if (fuse_scale)
            {
                scale_data = weights[2]->data();
            }
            if (scale_bias_term)
            {
                scale_bias_data = weights[3]->data();
            }
            // memset(output, 0xFF, sizeof(float) * this->top_blob(0)->data_size());
            size_t stride = bottoms[0]->width() * bottoms[0]->height();
            bn_kernel(channels, stride, alpha_data, beta_data, scale_bias_data, scale_data, input, output, 1);
            return 0;
        }

        int Fuse(Layer *next_layer)
        {
            if (next_layer->type.compare("Scale") == 0)
            {
                printf("BN %s fuse Scale layer %s\n", this->name.c_str(), next_layer->name.c_str());
                fuse_scale = true;
                for (int i = 0; i < next_layer->weights.size(); ++i)
                {
                    Blob<float> *p_blob = new Blob<float>();
                    p_blob->Copy(next_layer->weights[i]);
                    // _weight_blobs.push_back(p_blob);
                }
                // scale_bias_term = ((ScaleLayer *)next_layer)->bias_term;
                return 1;
            }
            else if (next_layer->type.compare("ReLU") == 0)
            {
                printf("BN %s fuse ReLU layer %s\n", this->name.c_str(), next_layer->name.c_str());
                fuse_relu = true;
                return 1;
            }
            else
                return 0;
        }

    private:
        int SetKernel()
        {
            int pattern_code = 0;
            pattern_code += (scale_bias_term) ? 0x1 : 0;
            pattern_code += (fuse_scale) ? 0x10 : 0;
            pattern_code += (fuse_relu) ? 0x100 : 0;
            //printf("pat_code %x\n", pat_code);
            switch (pattern_code)
            {
            case 0x000:
                bn_kernel = booster::batchnorm<false, false, false>;
                break;
            case 0x001:
                bn_kernel = booster::batchnorm<true, false, false>;
                break;
            case 0x010:
                bn_kernel = booster::batchnorm<false, true, false>;
                break;
            case 0x011:
                bn_kernel = booster::batchnorm<true, true, false>;
                break;
            case 0x100:
                bn_kernel = booster::batchnorm<false, false, true>;
                break;
            case 0x101:
                bn_kernel = booster::batchnorm<true, false, true>;
                break;
            case 0x110:
                bn_kernel = booster::batchnorm<false, true, true>;
                break;
            case 0x111:
                bn_kernel = booster::batchnorm<true, true, true>;
                break;
            default:
                fprintf(stderr, "Invalid pattern code 0x%x for batchnorm kernel\n", pattern_code);
                return -1;
            }
            return 0;
        }
        void (*bn_kernel)(const size_t channels, const size_t stride, const float* alpha, const float* beta, const float* bias_data, const float* scale_data, const float* input, float* output, const size_t num_threads);

      private:
        // size_t input_channels;
        // size_t input_width;
        // size_t input_height;
        int channels;
        float eps;

        bool fuse_scale;
        bool scale_bias_term;
        bool fuse_relu;
};
};
