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

#include "feather_generated.h"
#include "blob.h"
#include "fp16/fp16.h"
#include "log.h"

#include "common.h"

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
    if(elem_size > this->data_size())
    {
        Free();
        _data = (Dtype*) _mm_malloc(elem_size * sizeof(Dtype), 32);
    }
}

template<class Dtype>
void Blob<Dtype>::FromProto(const void *proto_in)//proto MUST be of type BlobProto*
{
    bool use_fp16_data = false;
    const BlobProto* proto = (const BlobProto*) proto_in;
    this->_num = proto->num();
    this->_channels = proto->channels();
    this->_height = proto->height();
    this->_width = proto->width();
    size_t data_length;
    data_length = VectorLength(proto->data());
    //printf("data length %d & %d\n", data_length, VectorLength(proto->data_fp16()));
    if(data_length == 0)
    {
	    data_length = VectorLength(proto->data_fp16());
	    //printf("LOADING FROM FP16 DATA LEN %zu\n", data_length);
	    use_fp16_data = true;
    }
    else
    {
	if(VectorLength(proto->data_fp16()) > 0)
	{
    //printf("LOADING FROM FP16 DATA LEN %zu\n", VectorLength(proto->data_fp16()));
		fprintf(stderr, "Fatal error: this model have FP16 and FP32 data in the same blob, aborting...\n");
		exit(-1);
	}
    }


    if (_num * _channels * _height * _width == data_length)
    {
        this->Alloc();
        for (int i = 0; i < data_length; ++i)
        {
            if (use_fp16_data)
            {
                if (std::is_same<Dtype, uint16_t>::value)
                {
                    this->_data[i] = proto->data_fp16()->Get(i);
                }
                else
                {
                    this->_data[i] = fp16_ieee_to_fp32_value(proto->data_fp16()->Get(i));
                }
            }
            else
            {
                if (std::is_same<Dtype, uint16_t>::value)
                {

                    this->_data[i] = hs_floatToHalf(proto->data()->Get(i));
                    // printf("%f, ", hs_halfToFloat(this->_data[i]));
                }
                else
                {
                    this->_data[i] = proto->data()->Get(i);
                }
              //this->_data[i] = proto->data()->Get(i);
            }
        }
    }
    else
    {
        //Error handling
    }
}

#ifdef FEATHER_OPENCL

template<class Dtype>
int Blob<Dtype>::WriteToDevice(cl::CommandQueue queue, const Dtype* data, size_t data_size)
{
    cl_int error_num;
    Dtype *mapped_ptr = (Dtype* )
      queue.enqueueMapBuffer(*_data_cl, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                             0, data_size * sizeof(Dtype), nullptr, nullptr, &error_num);
    if (!checkSuccess(error_num))
    {
      LOGE("fatal error: WriteBuffer Mapping memory objects failed. %s: %s", __FILE__, __LINE__);
      mapped_ptr = nullptr;
      return 1;
    }
    memcpy(mapped_ptr, data, data_size * sizeof(Dtype));

    error_num = queue.enqueueUnmapMemObject(*_data_cl, mapped_ptr,
                                             nullptr, nullptr);
    if (!checkSuccess(error_num)){
      LOGE("fatal error: WriteBuffer Unmapping memory objects failed. %s: %s", __FILE__, __LINE__);
      return 1;
    }
    return 0;

}

template<class Dtype>
int Blob<Dtype>::AllocDevice(cl::Context context, size_t data_size)
{
    if (!this->_data_cl)
    {
        cl_int error_num;
        _data_cl = new cl::Buffer(context,
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  data_size * sizeof(Dtype), nullptr, &error_num);

        if (!checkSuccess(error_num))
        {
            LOGE("Failed to create OpenCL buffers[%d]. %s: %s", error_num, __FILE__, __LINE__);
            return 1;
        }
    }
    return 0;
}

template<class Dtype>
int Blob<Dtype>::ReadFromDevice(cl::CommandQueue queue, Dtype* data, size_t data_size) const
{

    cl_int error_num;
    Dtype*  mapped_ptr = (Dtype* )
      queue.enqueueMapBuffer(*_data_cl, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                             0, data_size * sizeof(Dtype), nullptr, nullptr, &error_num);
    if (!checkSuccess(error_num))
    {
      LOGE("fatal error: ReadFromDevice Mapping memory objects failed. %s: %s", __FILE__, __LINE__);
      mapped_ptr = nullptr;
      return 1;
    }
    memcpy(data, mapped_ptr, data_size * sizeof(Dtype));
    error_num = queue.enqueueUnmapMemObject(*_data_cl, mapped_ptr,
                                             nullptr, nullptr);
    if (!checkSuccess(error_num)){
      LOGE("fatal error: ReadFromDevice Unmapping memory objects failed. %s: %s", __FILE__, __LINE__);
      return 1;
    }
    return 0;

}

template<class Dtype>
int Blob<Dtype>::ReadFromDeviceCHW(cl::CommandQueue queue, float* data) const
{
    cl_int error_num;
    size_t data_size = this->data_size_padded_c();
    Dtype *data_half =
      (Dtype*)(queue.enqueueMapBuffer(*_data_cl, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                             0, data_size * sizeof(Dtype), nullptr, nullptr, &error_num) );
    if (!checkSuccess(error_num))
    {
      LOGE("fatal error: ReadBuffer Mapping memory objects failed. %s: %s", __FILE__, __LINE__);
      data_half = nullptr;
      return 1;
    }

    // for (int i = 0; i < _channels; ++i) {
    //   for (int j = 0; j < _height * _width; ++j) {
    //     int dst_idx = i * _height * _width + j;
    //     int src_idx = j * this->get_channels_padding() + i;
    //     if (std::is_same<Dtype, uint16_t>::value)
    //         data[dst_idx] = hs_halfToFloat(data_half[src_idx]);
    //     else
    //         data[dst_idx] = data_half[src_idx];
    //   }
    // }

    // [(c+N-1)/N, h, w, N] -> [c, h, w]
    auto _height_x_width = _height * _width;
    if (std::is_same<Dtype, uint16_t>::value) {
      for (int i = 0; i < _channels; ++i) {
        auto channel_group_idx = i / this->channel_grp();
        auto channel_idx_within_group = i % this->channel_grp();
        for (int j = 0; j < _height_x_width; ++j) {
          int dst_idx = i * _height_x_width + j;
          int src_idx = (channel_group_idx * _height_x_width + j) * this->channel_grp()
                        + channel_idx_within_group;
          data[dst_idx] = hs_halfToFloat(data_half[src_idx]);
        } 
      }
    } else {
        for (int i = 0; i < _channels; ++i) {
        auto channel_group_idx = i / this->channel_grp();
        auto channel_idx_within_group = i % this->channel_grp();
        for (int j = 0; j < _height_x_width; ++j) {
          int dst_idx = i * _height_x_width + j;
          int src_idx = (channel_group_idx * _height_x_width + j) * this->channel_grp()
                        + channel_idx_within_group;
          data[dst_idx] = data_half[src_idx];
        } 
      }
    }

    error_num = queue.enqueueUnmapMemObject(*_data_cl, data_half,
                                             nullptr, nullptr);
    if (!checkSuccess(error_num)){
      LOGE("fatal error: ReadFromDevice Unmapping memory objects failed. %s: %s", __FILE__, __LINE__);
      return 1;
    }
    return 0;

}

template<class Dtype>
int Blob<Dtype>::FreeDevice()
{
    if(this->_data_cl) {
        delete this->_data_cl;
        this->_data_cl= NULL;
    }
    return 0;
}

template<class Dtype>
int Blob<Dtype>::ReshapeWithReallocDevice(cl::Context context, size_t num, size_t channels, size_t height, size_t width)
{
    size_t old_size = this->height() * this->width();
    size_t new_size = height * width;
    this->_num      = num;
    this->_channels = channels;
    this->_height   = height;
    this->_width    = width;

    if (new_size > old_size)
    {
        this->FreeDevice();
        if(this->AllocDevice(context, this->data_size_padded_c())) {
            LOGE("reallocate cl memory failed.");
            return -1;
        }
        return 2;
    }
    return 0;
}
#endif



template class Blob<float>;
template class Blob<uint16_t>;
template class Blob<char>;
};
