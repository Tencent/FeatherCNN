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

#include "utils.h"

#include "ncnn/mat.h"

#include <string>

namespace feather
{
template <class Dtype>
class Blob
{
    public:
        Blob()
            : name(), _num(0), _channels(0), _height(0), _width(0), _data(NULL)
        {}

        explicit Blob(std::string name)
            : name(name), _num(0), _channels(0), _height(0), _width(0), _data(NULL)
        {}

        explicit Blob(const size_t num, const size_t channels, const size_t height, const size_t width)
            : name(), _data(NULL), _num(num), _channels(channels), _height(height), _width(width)
        {}

        explicit Blob(Dtype* data, const size_t num, const size_t channels, const size_t height, const size_t width)
            : name(), _data(data), _num(num), _channels(channels), _height(height), _width(width)
        {}

        ~Blob()
        {
            Free();
        }

        void Free();
        void Alloc();

        void ReshapeWithRealloc(const Blob<Dtype> *p_blob);
        void ReshapeWithRealloc(int num, int channels, int height, int width);
        void Realloc(size_t elem_size);

        int CopyFromMat(const ncnn::Mat &src_mat);
        int CopyDataFromMat(const ncnn::Mat &src_mat);

        void CopyData(const Dtype* data)
        {
            size_t size = _num * _channels * _height * _width;
            memcpy(_data, data, sizeof(Dtype) * size);
        }
        void CopyShape(const Blob<Dtype>* p_blob)
        {
            this->_num = p_blob->num();
            this->_channels = p_blob->channels();
            this->_width = p_blob->width();
            this->_height = p_blob->height();
        }
        void Copy(const Blob<Dtype>* p_blob)
        {
            this->Free();
            CopyShape(p_blob);
            this->Alloc();
            CopyData(p_blob->data());
        }

        Dtype* data() const
        {
            return (Dtype*) _data;
        }

        size_t data_size() const
        {
            return _num * _channels * _height * _width;
        }
        size_t num() const
        {
            return _num;
        }
        size_t channels() const
        {
            return _channels;
        }
        size_t height() const
        {
            return _height;
        }
        size_t width() const
        {
            return _width;
        }
        void PrintBlobInfo() const
        {
            printf("----BlobShape----\n");
            printf("NCHW=(%zu %zu %zu %zu)\n", _num, _channels, _height, _width);
            printf("----------------\n");
        }

        std::string name;

        void* _data;
        size_t _elemsize;

        size_t _num;
        size_t _channels;
        size_t _height;
        size_t _width;
};
};
