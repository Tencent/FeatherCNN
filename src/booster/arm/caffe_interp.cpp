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

#include <booster/caffe_interp.h>

// Bi-linear interpolation
// IN : [channels height1 width1] cropped from a bigger [Height1 Width1] image
// OUT: [channels height2 width2] cropped from a bigger [Height2 Width2] image
template <typename Dtype, bool packed>
void caffe_cpu_interp2(const int channels,
                       const Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
                       Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2)
{
    // CHECK(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0);
    // CHECK(Width1 >= width1 + x1 && Height1 >= height1 + y1 && Width2 >= width2 + x2 && Height2 >= height2 + y2);
    // special case: just copy
    if (height1 == height2 && width1 == width2)
    {
        for (int h2 = 0; h2 < height2; ++h2)
        {
            const int h1 = h2;
            for (int w2 = 0; w2 < width2; ++w2)
            {
                const int w1 = w2;
                if (packed)
                {
                    const Dtype *pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
                    Dtype *pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
                    for (int c = 0; c < channels; ++c)
                    {
                        pos2[0] = pos1[0];
                        pos1++;
                        pos2++;
                    }
                }
                else
                {
                    const Dtype *pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
                    Dtype *pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
                    for (int c = 0; c < channels; ++c)
                    {
                        pos2[0] = pos1[0];
                        pos1 += Width1 * Height1;
                        pos2 += Width2 * Height2;
                    }
                }
            }
        }
        return;
    }
    const float rheight = (height2 > 1) ? static_cast<float>(height1) / (height2) : 0.f;
    const float rwidth = (width2 > 1) ? static_cast<float>(width1) / (width2) : 0.f;
    for (int h2 = 0; h2 < height2; ++h2)
    {
        const float h1r = rheight * h2;
        const int h1 = h1r;
        const int h1p = (h1 < height1 - 1) ? 1 : 0;
        const Dtype h1lambda = h1r - h1;
        const Dtype h0lambda = Dtype(1.) - h1lambda;
        for (int w2 = 0; w2 < width2; ++w2)
        {
            const float w1r = rwidth * w2;
            const int w1 = w1r;
            const int w1p = (w1 < width1 - 1) ? 1 : 0;
            const Dtype w1lambda = w1r - w1;
            const Dtype w0lambda = Dtype(1.) - w1lambda;
            if (packed)
            {
                const Dtype *pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
                Dtype *pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
                for (int c = 0; c < channels; ++c)
                {
                    pos2[0] =
                        h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[channels * w1p]) +
                        h1lambda * (w0lambda * pos1[channels * h1p * Width1] + w1lambda * pos1[channels * (h1p * Width1 + w1p)]);
                    pos1++;
                    pos2++;
                }
            }
            else
            {
                const Dtype *pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
                Dtype *pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
                for (int c = 0; c < channels; ++c)
                {
                    pos2[0] =
                        h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p]) +
                        h1lambda * (w0lambda * pos1[h1p * Width1] + w1lambda * pos1[h1p * Width1 + w1p]);
                    pos1 += Width1 * Height1;
                    pos2 += Width2 * Height2;
                }
            }
        }
    }
}

template void caffe_cpu_interp2<float,false>(const int, const float *, const int, const int, const int, const int, const int, const int, float *, const int, const int, const int, const int, const int, const int);
template void caffe_cpu_interp2<float,true>(const int, const float *, const int, const int, const int, const int, const int, const int, float *, const int, const int, const int, const int, const int, const int);
template void caffe_cpu_interp2<double,false>(const int, const double *, const int, const int, const int, const int, const int, const int, double *, const int, const int, const int, const int, const int, const int);
template void caffe_cpu_interp2<double,true>(const int, const double *, const int, const int, const int, const int, const int, const int, double *, const int, const int, const int, const int, const int, const int);
