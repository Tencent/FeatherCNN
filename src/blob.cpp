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

#include "feather_simple_generated.h"
#include "blob.h"

namespace feather
{
template<class Dtype>
void Blob<Dtype>::Alloc()
{
    size_t dim_byte = _num * _channels * _height * _width * sizeof(Dtype);
    _data = (Dtype*) _mm_malloc(dim_byte, 128);
}

template<class Dtype>
void Blob<Dtype>::FromProto(const void *proto_in)//proto MUST be of type BlobProto*
{
    const BlobProto* proto = (const BlobProto*) proto_in;
    this->_num = proto->num();
    this->_channels = proto->channels();
    this->_height = proto->height();
    this->_width = proto->width();
    size_t data_length;
    data_length = proto->data()->Length();

    if (_num * _channels * _height * _width == data_length)
    {
        this->Alloc();
        for (int i = 0; i < data_length; ++i)
        {
            this->_data[i] = proto->data()->Get(i);
        }
    }
    else
    {
        //Error handling
    }
}

template class Blob<float>;
template class Blob<short>;
template class Blob<char>;
};
