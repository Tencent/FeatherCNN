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

#include "blob.h"

namespace feather
{
template<class Dtype>
void Blob<Dtype>::Alloc()
{
    size_t dim_byte = _num * _channels * _height * _width * sizeof(Dtype);
    _data = (Dtype*) _mm_malloc(dim_byte, 32);
}
template<class Dtype>
void Blob<Dtype>::Free()
{
    if (this->_data)
    {
        free(this->_data);
        this->_data = NULL;
    }
}

template<class Dtype>
void Blob<Dtype>::ReshapeWithRealloc(const Blob<Dtype> *p_blob)
{
    int num      = p_blob->num();
    int channels = p_blob->channels();
    int height   = p_blob->height();
    int width    = p_blob->width();

    ReshapeWithRealloc(num, channels, height, width);
}

template<class Dtype>
void Blob<Dtype>::ReshapeWithRealloc(int num, int channels, int height, int width)
{
    // LOGI("Reallc: (%d %d %d %d) to (%d %d %d %d)", _num, _channels, _height, _width, num, channels, height, width);
    int elem_size = num * channels * height * width;
    Realloc(elem_size);
    this->_num      = num;
    this->_channels = channels;
    this->_height   = height;
    this->_width    = width;
}

template<class Dtype>
void Blob<Dtype>::Realloc(size_t elem_size)
{
    if (elem_size > this->data_size())
    {
        Free();
        _data = (Dtype*) _mm_malloc(elem_size * sizeof(Dtype), 32);
    }
}

template<class Dtype>
int Blob<Dtype>::CopyFromMat(const ncnn::Mat& mat)
{
    this->ReshapeWithRealloc(1, mat.c, mat.h, mat.w);
    this->CopyDataFromMat(mat);
    return 0;
}

template<class Dtype>
int Blob<Dtype>::CopyDataFromMat(const ncnn::Mat& mat)
{
    if (this->data_size() != mat.c * mat.h * mat.w)
    {
        LOGE("In Blob %s: Mat and target blob shape mismatch. blob shape (%zu %zu %zu %zu), mat shape (%d %d %d)\n", this->name.c_str(), num(), channels(), height(), width(), mat.c, mat.h, mat.w);
        return -500; // BAD DATA DIMENSION
    }
    Dtype* dst_p = (Dtype *) this->_data;
    size_t copy_stride = mat.h * mat.w;
    for (int c = 0; c < mat.c; ++c )
    {
        ncnn::Mat channel_mat = mat.channel(c);
        memcpy(dst_p, channel_mat.data, copy_stride * sizeof(Dtype));
        dst_p += copy_stride;
    }
    return 0;
}

template class Blob<float>;
template class Blob<uint16_t>;
template class Blob<char>;
};
