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
class Yolov2ReorgLayer : public Layer<float>
{
    public:
        Yolov2ReorgLayer(const LayerParameter* layer_param, RuntimeParameter<float>* rt_param)
            : Layer<float>(layer_param, rt_param)
        {
        }

        int GenerateTopBlobs()
        {
            if (_top.size() != 1 || _bottom.size() != 1)
                return -1;
            Blob<float> *p_blob = new Blob<float>();
            p_blob->CopyShape(_bottom_blobs[_bottom[0]]);
            p_blob->_channels = p_blob->_channels * 4;
            p_blob->_height = p_blob->_height / 2;
            p_blob->_width = p_blob->_width / 2;
            
            // p_blob->channels = p_blob->channels() * 4;
            p_blob->Alloc();
            _top_blobs[_top[0]] = p_blob;
            return 0;
        }

        void reorg_cpu(float* output, const float *input, const int channels, const int height, const int width, const int stride)
        {
            int out_h = height / stride;
            int out_w = width / stride;
            //printf("input %d %d %d output %d %d\n", channels, height, width, out_h, out_w);
            for (int c = 0; c < channels; c++)
            {
                const float* channel_ptr = input + c * height * width;
                for (int sh = 0; sh < stride; sh++)
                {
                    for (int sw = 0; sw < stride; sw++)
                    {
                        // float *outptr = top_blob.channel(q * stride * stride + sh * stride + sw);
                        // float* outp = output + (c * stride * stride + sh * stride + sw) * out_h * out_w;
                        // float* inp = channel_ptr + sh * stride + sw
                        float* outp = output + (c + (sh * stride + sw) * channels) * out_h * out_w;
                        for (int i = 0; i < out_h; i++)
                        {
                            // const float *sptr = m.row(i * stride + sh) + sw;
                            // const float* sptr = channel_ptr + (i * stride + sh) * width + sw;
                            // const float* sptr = channel_ptr + (i * stride + sh) * width + sw;
                            for (int j = 0; j < out_w; j++)
                            {
                                // outp[i * out_w + j] = sptr[j * stride];
                                outp[i * out_w + j] = channel_ptr[(i * stride + sh) * width + j * stride + sw];
                            }
                        }
                    }
                }
            }
        }

        int Forward()
        {
            const Blob<float> *p_bottom = _bottom_blobs[_bottom[0]];
            const float *input = p_bottom->data();
            const size_t data_size = p_bottom->num() * p_bottom->channels() * p_bottom->height() * p_bottom->width();

            float *output = _top_blobs[_top[0]]->data();
            reorg_cpu(output, input, p_bottom->channels(), p_bottom->height(), p_bottom->width(), 2);
            return 0;
        }
};
};
