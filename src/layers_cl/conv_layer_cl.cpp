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
#include "conv_layer_cl.h"

namespace feather
{
//#define USE_LEGACY_SGEMM

template <class Dtype>
ConvLayerCL<Dtype>::ConvLayerCL(const LayerParameter *layer_param, RuntimeParameter<Dtype>* rt_param)
    : Layer<Dtype>(layer_param, rt_param), conv_param(),
      conv_booster(),
      bias_data(NULL),
      kernel_data(NULL)
{
    this->_fusible = true;
    const ConvolutionParameter *conv_param_in = layer_param->convolution_param();
    this->conv_param.kernel_h = conv_param_in->kernel_h();
    this->conv_param.kernel_w = conv_param_in->kernel_w();
    this->conv_param.stride_h = conv_param_in->stride_h();
    this->conv_param.stride_w = conv_param_in->stride_w();
    this->conv_param.pad_left = conv_param_in->pad_w();
    this->conv_param.pad_bottom = conv_param_in->pad_h();
    this->conv_param.pad_right = conv_param_in->pad_w();
    this->conv_param.pad_top = conv_param_in->pad_h();
    this->conv_param.group = conv_param_in->group();
    kernel_data = this->_weight_blobs[0]->data();
    this->conv_param.output_channels = this->_weight_blobs[0]->num();
    this->conv_param.bias_term = conv_param_in->bias_term();
    this->conv_param.activation = booster::None;
    assert(this->_weight_blobs.size() > 0);

    if (this->conv_param.stride_w == 0) this->conv_param.stride_w = 1;
    if (this->conv_param.stride_h == 0) this->conv_param.stride_h = 1;
    if (this->conv_param.group == 0) this->conv_param.group = 1;
    if (this->conv_param.bias_term)
    {
        assert(this->_weight_blobs.size() == 2);
        this->bias_data = this->_weight_blobs[1]->data();
    }
}


template <class Dtype>
int ConvLayerCL<Dtype>::SetBuildOptions()
{
    bool is_fp16 = std::is_same<Dtype, uint16_t>::value;
    this->conv_booster.SetBuildOpts(this->conv_param,
                                    is_fp16,
                                    this->conv_booster.GetKernelNames(),
                                    this->cl_kernel_info_map);
    return 0;
}

template <class Dtype>
int ConvLayerCL<Dtype>::SetKernelParameters()
{
    int error_num;
    size_t n_grp_size = this->conv_param.channel_block_size;
    size_t real_weight_size = this->conv_booster.GetWeightSize();
    std::vector<Dtype> weight_reformed(real_weight_size, 0);
    this->conv_booster.WeightReform(this->conv_param,
                                    n_grp_size,
                                    1,
                                    this->kernel_data,
                                    weight_reformed.data());

    this->_weight_blobs[0]->AllocDevice(this->rt_param->context(), real_weight_size);
    this->_weight_blobs[0]->WriteToDevice(this->rt_param->command_queue(), weight_reformed.data(), real_weight_size);
    this->_weight_blobs[0]->Free();

    booster::CLBuffers buffers;
    if (this->conv_param.bias_term)
    {
        this->_weight_blobs[1]->AllocDevice(this->rt_param->context(), this->conv_param.padded_output_channels);
        std::vector<Dtype> bias_padding(this->conv_param.padded_output_channels, 0);
        memcpy(bias_padding.data(), this->bias_data, this->conv_param.output_channels * sizeof(Dtype));
        this->_weight_blobs[1]->WriteToDevice(this->rt_param->command_queue(), bias_padding.data(), this->conv_param.padded_output_channels);
        this->_weight_blobs[1]->Free();
    	buffers.bias_mem = this->_weight_blobs[1]->data_cl();
    }

    this->rt_param->alloc_padded_input();

    buffers.input_mem = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    buffers.padded_input_mem = this->rt_param->padded_input() ? this->rt_param->padded_input()->data_cl() : NULL;
    buffers.weight_mem = this->_weight_blobs[0]->data_cl();
    buffers.output_mem = this->_top_blobs[this->_top[0]]->data_cl();
    buffers.input_trans_mem = NULL;
    buffers.out_trans_mem = NULL;
    this->conv_booster.SetConvKernelParams(this->conv_param, buffers, this->conv_booster.GetKernelNames(), this->cl_kernel_info_map, this->rt_param->cl_runtime(), false);
    this->conv_booster.SetConvWorkSize(this->conv_param, this->cl_kernel_info_map, this->conv_booster.GetKernelNames(), this->rt_param->cl_runtime());

    return 0;
}

template <class Dtype>
int ConvLayerCL<Dtype>::ForwardReshapeCL()
{
    if (this->conv_param.input_h == this->_bottom_blobs[this->_bottom[0]]->height() &&
            this->conv_param.input_w == this->_bottom_blobs[this->_bottom[0]]->width())
        return this->ForwardCL();

    this->conv_param.input_h = this->_bottom_blobs[this->_bottom[0]]->height();
    this->conv_param.input_w = this->_bottom_blobs[this->_bottom[0]]->width();

    this->conv_param.AssignOutputDim();
    if (this->conv_param.output_h < 1 || this->conv_param.output_w < 1)
    {
        LOGE("invalid output size in forward reshape");
        return -1;
    }

    this->_top_blobs[this->_top[0]]->ReshapeWithReallocDevice(this->rt_param->context(),
            this->_top_blobs[this->_top[0]]->num(),
            this->_top_blobs[this->_top[0]]->channels(),
            this->conv_param.output_h, this->conv_param.output_w);

    this->conv_param.AssignCLPaddedDim();
    if (this->conv_param.padding_needed)
    {
        size_t conv_padded_input_size = this->conv_param.padded_input_h * this->conv_param.padded_input_w * this->conv_param.padded_input_channels;
        this->rt_param->realloc_padded_input(conv_padded_input_size);
    }

    booster::CLBuffers buffers;
    buffers.input_mem = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    buffers.padded_input_mem = this->rt_param->padded_input() ? this->rt_param->padded_input()->data_cl() : NULL;
    buffers.output_mem = this->_top_blobs[this->_top[0]]->data_cl();
    this->conv_booster.SetConvKernelParams(this->conv_param, buffers, this->conv_booster.GetKernelNames(), this->cl_kernel_info_map, this->rt_param->cl_runtime(), true);
    this->conv_booster.SetConvWorkSize(this->conv_param, this->cl_kernel_info_map, this->conv_booster.GetKernelNames(), this->rt_param->cl_runtime());
    return this->ForwardCL();
}

template <class Dtype>
int ConvLayerCL<Dtype>::ForwardCL()
{
    //this->conv_booster.Forward(this->rt_param->command_queue(), this->conv_booster.GetKernelNames(), this->cl_kernel_info_map);
    this->conv_booster.Forward(this->rt_param->command_queue(), this->conv_booster.GetKernelNames(),
                               this->cl_kernel_info_map, this->conv_param,
                               this->rt_param->cl_runtime(), this->name());
    return 0;
}


template <class Dtype>
int ConvLayerCL<Dtype>::GenerateTopBlobs()
{
    // Conv layer has and only has one bottom blob.
    const Blob<Dtype> *bottom_blob = this->_bottom_blobs[this->_bottom[0]];

    this->conv_param.input_w = bottom_blob->width();
    this->conv_param.input_h = bottom_blob->height();
    this->conv_param.input_channels = bottom_blob->channels();
    this->conv_param.AssignOutputDim();
    this->_top_blobs[this->_top[0]] = new Blob<Dtype>(1, this->conv_param.output_channels, this->conv_param.output_h, this->conv_param.output_w);
    this->_top_blobs[this->_top[0]]->AllocDevice(this->rt_param->context(), this->_top_blobs[this->_top[0]]->data_size_padded_c());

    this->conv_param.padded_input_channels = this->_bottom_blobs[this->_bottom[0]]->get_channels_padding();
    this->conv_param.padded_output_channels = this->_top_blobs[this->_top[0]]->get_channels_padding();
    this->conv_param.AssignCLPaddedDim();
    if (this->conv_param.padding_needed)
    {
        size_t conv_padded_input_size = this->conv_param.padded_input_h * this->conv_param.padded_input_w * this->conv_param.padded_input_channels;
        this->rt_param->update_padded_input_size(conv_padded_input_size);
    }

    this->conv_booster.SelectAlgo(&this->conv_param);
    this->conv_booster.Init(this->conv_booster.GetProgramNames(), this->conv_booster.GetKernelNames(), this->cl_kernel_info_map);
    return 0;
}


template <class Dtype>
int ConvLayerCL<Dtype>::Fuse(Layer<Dtype> *next_layer)
{
    if (next_layer->type().compare("ReLU") == 0)
    {
        this->conv_param.activation = booster::ReLU;
        return 1;
    }
    else
    {
        return 0;
    }
}

template class ConvLayerCL<float>;
template class ConvLayerCL<uint16_t>;

}; // namespace feather
