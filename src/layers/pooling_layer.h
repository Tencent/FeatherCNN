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

#include <math.h>
#include <limits>

#define MAX(a,b) ((a)>(b))?(a):(b)
#define MIN(a,b) ((a)<(b))?(a):(b)

namespace feather
{
void ave_pool_inner_kernel(float* out, const float* in, const size_t ldin, const size_t kernel_h, const size_t kernel_w)
{
    float total = 0.0;
    for (size_t m = 0; m != kernel_h; ++m)
    {
        for (size_t n = 0; n != kernel_w; ++n)
        {
            size_t pos = m * ldin + n;
            total += in[pos];
        }
    }
    *out = total / kernel_h / kernel_w;
}

void max_pool_inner_kernel(float* out, const float* in, const size_t ldin, const size_t kernel_h, const size_t kernel_w)
{
    float max = 0.0;
    for (size_t m = 0; m != kernel_h; ++m)
    {
        for (size_t n = 0; n != kernel_w; ++n)
        {
            size_t pos = m * ldin + n;
            max = (in[pos] > max) ? in[pos] : max;
        }
    }
    *out = max;
}


class PoolingLayer : public Layer
{
    public:
        PoolingLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
            : stride_height(1),
              stride_width(1),
              Layer(layer_param, rt_param)
        {
            const PoolingParameter *pooling_param = layer_param->pooling_param();
            kernel_height = pooling_param->kernel_h();
            kernel_width = pooling_param->kernel_w();
            pad_height = pooling_param->pad_h();
            pad_width = pooling_param->pad_w();
            stride_height = pooling_param->stride_h();
            stride_width = pooling_param->stride_w();
            stride_height = (stride_height <= 0) ? 1 : stride_height;
            stride_width  = (stride_width  <= 0) ? 1 : stride_width;
            global_pooling = pooling_param->global_pooling();
            this->method = pooling_param->pool();
            switch (this->method)
            {
                case PoolingParameter_::PoolMethod_MAX_:
                    _pool_inner_kernel = max_pool_inner_kernel;
                    break;
                case PoolingParameter_::PoolMethod_AVE:
                    _pool_inner_kernel = ave_pool_inner_kernel;
                    break;
                default:
                    fprintf(stderr, "Unsupported pool method\n");
            }
            //printf("kernel (%ld %ld) pad (%ld %ld) stride (%ld %ld) global_pooling %d\n",
            //     kernel_height, kernel_width, pad_height, pad_width, stride_height, stride_width, global_pooling);

        }


        int Forward()
        {
            fprintf(stderr, "Pooling layer %s\ninput shape %ld %ld %ld kernel shape %ld %ld stride %ld %ld\n", this->name().c_str(), input_channels, input_height, input_width, kernel_height, kernel_width, stride_height, stride_width);
            fprintf(stderr, "output (%d %d)\n", output_height, output_width);
            const float *input = _bottom_blobs[_bottom[0]]->data();
            float *output = _top_blobs[_top[0]]->data();
            float *p = output;

            int slot = input_channels * output_height;

            #pragma omp parallel for schedule(static) num_threads(num_threads)
//  for (int u=0;u<slot;u++)
//  {
            for (int i = 0; i < input_channels; ++i)
            {
                for (int j = 0; j < output_height; j ++)
                {
//    int i=slot/output_height,  j=slot%output_height;
                    float *p = output + i * output_height * output_width + j * output_width;
                    for (int l = 0; l < output_width; l++)  p[l] = (this->method != PoolingParameter_::PoolMethod_MAX_ ? 0 : -1 * std::numeric_limits<float>::max()) ;

                    int tmp_pos = j * (int)stride_height - (int)pad_height;
                    int x_min = MAX(tmp_pos, 0);
                    int x_max = MIN((int)(tmp_pos + kernel_height), (int) input_height);

		    for (int k = 0; k < output_width; k ++)
		    {
			    int counter = 0;
			    float total = (this->method != PoolingParameter_::PoolMethod_MAX_ ? 0 : -1 * std::numeric_limits<float>::max());
			    for (int x = x_min; x < x_max; ++x)
			    {
				    int xpos = i * input_height * input_width + x * input_width;


				    int local_pos = k * (int)stride_width - (int)pad_width;
				    int y_min     = MAX(local_pos, 0);
				    int y_max     = MIN((int)(local_pos + kernel_width), (int) input_width);

				    for (int y = y_min; y < y_max; ++y)
				    {
					    float value = input[xpos + y];
					    if (this->method != PoolingParameter_::PoolMethod_MAX_)        total += value, counter++;
					    else                                          total = total > value ? total : value;
				    }
			    }
			    if (this->method != PoolingParameter_::PoolMethod_MAX_)
				    p[k] += total / (counter);
			    else    p[k]  = (p[k] > total) ? p[k] : total;
		    }
                }
            }
            return 0;
        }
       
        int ForwardReshape()
        {
            const Blob<float> *bottom_blob = _bottom_blobs[_bottom[0]];
            input_height = bottom_blob->height();
            input_width = bottom_blob->width();
            input_channels = bottom_blob->channels();
            //printf("layer %s\n", _name.c_str());
            //printf("input %lu %lu %lu\n", input_channels, input_height, input_width);
            if (global_pooling)
            {
                kernel_height = input_height;
                kernel_width = input_width;
                output_height = 1;
                output_width = 1;
                output_channels = input_channels;
            }
            else
            {
                //General pooling.
                output_channels = input_channels;
                output_height = static_cast<int>(ceil(static_cast<float>(input_height + 2 * pad_height - kernel_height) / stride_height)) + 1;
                output_width = static_cast<int>(ceil(static_cast<float>(input_width + 2 * pad_width - kernel_width) / stride_width)) + 1;
            }
            _top_blobs[_top[0]]->ReshapeWithRealloc(1, output_channels, output_height, output_width);
            return Forward();
        }

        int GenerateTopBlobs()
        {
            //Only accept a single bottom blob.
            const Blob<float> *bottom_blob = _bottom_blobs[_bottom[0]];
            input_height = bottom_blob->height();
            input_width = bottom_blob->width();
            input_channels = bottom_blob->channels();
            //printf("layer %s\n", _name.c_str());
            //printf("input %lu %lu %lu\n", input_channels, input_height, input_width);
            if (global_pooling)
            {
                kernel_height = input_height;
                kernel_width = input_width;
                output_height = 1;
                output_width = 1;
                output_channels = input_channels;
            }
            else
            {
                //General pooling.
                output_channels = input_channels;
                output_height = static_cast<int>(ceil(static_cast<float>(input_height + 2 * pad_height - kernel_height) / stride_height)) + 1;
                output_width = static_cast<int>(ceil(static_cast<float>(input_width + 2 * pad_width - kernel_width) / stride_width)) + 1;
            }
            _top_blobs[_top[0]] = new Blob<float>(1, output_channels, output_height, output_width);
            _top_blobs[_top[0]]->Alloc();
            //_top_blobs[_top[0]]->PrintBlobInfo();
            return 0;
        }

    private:
        size_t input_height;
        size_t input_width;
        size_t input_channels;
        size_t output_height;
        size_t output_width;
        size_t output_channels;
        size_t pad_height;
        size_t pad_width;
        size_t kernel_height;
        size_t kernel_width;
        size_t stride_height;
        size_t stride_width;
        bool global_pooling;
        PoolingParameter_::PoolMethod method;
        void (*_pool_inner_kernel)(float* out, const float* in, const size_t ldin, const size_t kernel_h, const size_t kernel_w);
};
};
