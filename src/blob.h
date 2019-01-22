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

#include <string>
#include <assert.h>
#include <stdio.h>
#include "common.h"

#ifdef FEATHER_OPENCL
#include "CLHPP/clhpp_common.hpp"
#endif

namespace feather
{
template <class Dtype>
class Blob
{
    public:
        Blob()
            : _num(0), _channels(0), _height(0), _width(0), _data(NULL)
#ifdef FEATHER_OPENCL
            , _data_cl(NULL), _data_float(NULL), _data_im(NULL)
#endif
        {}

        explicit Blob(const size_t num, const size_t channels, const size_t height, const size_t width)
            : _data(NULL), _num(num), _channels(channels), _height(height), _width(width), _name()
#ifdef FEATHER_OPENCL
            , _data_cl(NULL), _data_float(NULL), _data_im(NULL)
#endif
        {}


        explicit Blob(Dtype* data, const size_t num, const size_t channels, const size_t height, const size_t width)
            : _data(data), _num(num), _channels(channels), _height(height), _width(width), _name()
#ifdef FEATHER_OPENCL
            , _data_cl(NULL), _data_float(NULL), _data_im(NULL)
#endif
        {}

        explicit Blob(Dtype* data, size_t num, size_t channels, size_t height, size_t width, std::string name)
            : _data(data), _num(num), _channels(channels), _height(height), _width(width), _name(name)
#ifdef FEATHER_OPENCL
            , _data_cl(NULL), _data_float(NULL), _data_im(NULL)
#endif
        {}

        ~Blob()
        {
            Free();
#ifdef FEATHER_OPENCL
            FreeDevice();
#endif
        }

        void Free();
        void Alloc();

        void ReshapeWithRealloc(const Blob<Dtype> *p_blob);

        void ReshapeWithRealloc(int num, int channels, int height, int width);

        void Realloc(size_t elem_size);

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
            assert(p_blob->data_size() == this->data_size());
            CopyData(p_blob->data());
        }

        void FromProto(const void *proto_in);//proto MUST be of type BlobProto*

        Dtype* data() const
        {
            return _data;
        }

        size_t data_size() const
        {
            return _num * _channels * _height * _width;
        }

        std::string name()
        {
            return _name;
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
            printf("----BlobInfo----\n");
            printf("Shape in nchw (%zu %zu %zu %zu)\n", _num, _channels, _height, _width);
            printf("----------------\n");
        }

#ifdef FEATHER_OPENCL
        cl::Buffer* data_cl() const
        {
            return _data_cl;
        }
        float* data_float() const
        {
            if (std::is_same<Dtype, uint16_t>::value)
                return _data_float;
            else
                return reinterpret_cast<float*>(_data);
        }
        size_t channel_grp() const
        {
            return _channel_grp;
        }
        size_t get_channels_padding() const
        {
            return (_channels / _channel_grp + !!(_channels % _channel_grp)) * _channel_grp;
        }
        size_t get_num_padding() const
        {
            return (_num / _num_grp + !!(_num % _num_grp)) * _num_grp;
        }
        size_t data_size_padded_c() const
        {
            return _num * get_channels_padding() * _height * _width;
        }
        size_t data_size_padded_n() const
        {
            return get_num_padding() * _channels * _height * _width;
        }
        size_t data_size_padded_nc() const
        {
            return get_num_padding() * get_channels_padding() * _height * _width;
        }
        int AllocDevice(cl::Context context, size_t data_size);
        int AllocDeviceImage(cl::Context context, size_t height, size_t width);
        int FreeDevice();
        int ReshapeWithReallocDevice(cl::Context context, size_t num, size_t channels, size_t height, size_t width);

        int WriteToDevice(cl::CommandQueue queue, const Dtype* data, size_t data_size);
        int ReadFromDevice(cl::CommandQueue queue, Dtype* data, size_t data_size) const;
        int ReadFromDeviceCHW(cl::CommandQueue queue, float* data) const;
#endif

        Dtype* _data;

#ifdef FEATHER_OPENCL
        /* Image2D in the near future */
        cl::Buffer *_data_cl;
        cl::Image2D *_data_im;
        float* _data_float;
#endif
        size_t _num;
        size_t _channels;
        size_t _height;
        size_t _width;
        size_t _channel_grp = 4;
        size_t _num_grp = 4;

        std::string _name;
};
};
