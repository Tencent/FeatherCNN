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

#include "slice_layer.h"
#include "arm/generic_kernels.h"

namespace feather
{
int SliceLayer::Forward()
{
    const Blob<float>* bottom_blob = _bottom_blobs[_bottom[0]];
    size_t num = bottom_blob->num();
    size_t channels = bottom_blob->channels();
    size_t height = bottom_blob->height();
    size_t width = bottom_blob->width();

    switch (axis)
    {
        case 0:
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int k = 0; k < slice_point.size() + 1; k++)
            {
                int start = (k == 0) ? 0 : slice_point[k - 1];
                int end   = (k == slice_point.size()) ? num : slice_point[k];

                const float *input = _bottom_blobs[_bottom[0]]->data();
                float *output      = _top_blobs[_top[k]]->data();

                int index = 0;
                for (int i = start; i < end; i++)  for (int j = 0; j < channels; j++)
                        for (int m = 0; m < height; m++)
                        {
                            for (int n = 0; n < width; n++)
                                output[index++] = input[i * channels * height * width + j * height * width + m * width + n];
                        }
            }
            break;
        case 1:
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int k = 0; k < slice_point.size() + 1; k++)
            {
                int start = (k == 0) ? 0 : slice_point[k - 1];
                int end   = (k == slice_point.size()) ? channels : slice_point[k];

                const float *input = _bottom_blobs[_bottom[0]]->data();
                float *output      = _top_blobs[_top[k]]->data();

                int index = 0;
                for (int i = 0; i < num; i++)  for (int j = start; j < end; j++)
                        for (int m = 0; m < height; m++)
                        {
                            for (int n = 0; n < width; n++)
                                output[index++] = input[i * channels * height * width + j * height * width + m * width + n];
                        }
            }
            break;
        case 2:
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int k = 0; k < slice_point.size() + 1; k++)
            {
                int start = (k == 0) ? 0 : slice_point[k - 1];
                int end   = (k == slice_point.size()) ? height : slice_point[k];
                const float *input = _bottom_blobs[_bottom[0]]->data();
                float *output      = _top_blobs[_top[k]]->data();

                int index = 0;
                for (int i = 0; i < num; i++)  for (int j = 0; j < channels; j++)
                        for (int m = start; m < end; m++)
                        {
                            for (int n = 0; n < width; n++)
                                output[index++] = input[i * channels * height * width + j * height * width + m * width + n];
                        }
            }
            break;
        case 3:
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int k = 0; k < slice_point.size() + 1; k++)
            {
                int start = (k == 0) ? 0 : slice_point[k - 1];
                int end   = (k == slice_point.size()) ? width : slice_point[k];

                const float *input = _bottom_blobs[_bottom[0]]->data();
                float *output      = _top_blobs[_top[k]]->data();

                int index = 0;
                for (int i = 0; i < num; i++)  for (int j = 0; j < channels; j++)
                        for (int m = 0; m < height; m++)
                        {
                            for (int n = start; n < end; n++)
                                output[index++] = input[i * channels * height * width + j * height * width + m * width + n];
                        }
            }

            break;
    }

    return 0;
}

int SliceLayer::GenerateTopBlobs()
{
    assert(_bottom.size() == 1);
    assert(_top.size() - 1 == slice_point.size());

    const Blob<float>* bottom_blob = _bottom_blobs[_bottom[0]];
    size_t num = bottom_blob->num();
    size_t channels = bottom_blob->channels();
    size_t height = bottom_blob->height();
    size_t width = bottom_blob->width();

    //Create sliced top blobs.
    switch (axis)
    {
        case 0:
            for (int i = 0; i < _top.size() - 1; ++i)
            {
                _top_blobs[_top[i]] = new Blob<float>(slice_point[i] - ((i == 0) ? 0 : slice_point[i - 1]), channels, height, width);
            }
            _top_blobs[_top[_top.size() - 1]] = new Blob<float>(slice_point[_top.size() - 2], channels, height, width);
            break;
        case 1:
            for (int i = 0; i < _top.size() - 1; ++i)
            {
                _top_blobs[_top[i]] = new Blob<float>(num, slice_point[i] - ((i == 0) ? 0 : slice_point[i - 1]), height, width);
            }
            _top_blobs[_top[_top.size() - 1]] = new Blob<float>(num, slice_point[_top.size() - 2], height, width);
            break;
        case 2:
            for (int i = 0; i < _top.size() - 1; ++i)
            {
                _top_blobs[_top[i]] = new Blob<float>(num, channels, slice_point[i] - ((i == 0) ? 0 : slice_point[i - 1]), width);
            }
            _top_blobs[_top[_top.size() - 1]] = new Blob<float>(num, channels, height - slice_point[_top.size() - 2], width);
            break;
        case 3:
            for (int i = 0; i < _top.size() - 1; ++i)
            {
                _top_blobs[_top[i]] = new Blob<float>(num, channels, height, slice_point[i] - ((i == 0) ? 0 : slice_point[i - 1]));
            }
            _top_blobs[_top[_top.size() - 1]] = new Blob<float>(num, channels, height, width - slice_point[_top.size() - 2]);
            break;
    }

    for (int i = 0; i < _top.size(); ++i)
    {
        _top_blobs[_top[i]]->Alloc();
    }
    return 0;
}

int SliceLayer::Init()
{
    printf("axis %d slice_point num %lu\n", axis, slice_point.size());
    for (int i = 0; i < _top.size(); i++)  printf("%s ", _top[i].c_str());
    printf("\n");
    printf("\n");
    return 0;
}
};
