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
#include "conv_layer_cl.h"

namespace feather {
//#define USE_LEGACY_SGEMM

template <class Dtype>
ConvLayerCL<Dtype>::ConvLayerCL(const LayerParameter *layer_param, RuntimeParameter<float>* rt_param)
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

    if(this->conv_param.stride_w  == 0) this->conv_param.stride_w  = 1;
    if(this->conv_param.stride_h == 0) this->conv_param.stride_h = 1;
    if(this->conv_param.group == 0) this->conv_param.group = 1;
    if (this->conv_param.bias_term)
    {
        assert(this->_weight_blobs.size() == 2);
        this->bias_data = this->_weight_blobs[1]->data();
    }

    cl::Kernel kernel;
    this->kernels.push_back(kernel);
    cl::Event event;
    this->events.push_back(event);
}


template <class Dtype>
int ConvLayerCL<Dtype>::SetBuildOptions() {
    std::ostringstream ss;
    ss << channel_grp_size;
    this->build_options.push_back("-DN=" + ss.str());
    //this->build_options.push_back("-DDATA_TYPE=half");
    if(std::is_same<Dtype, uint16_t>::value)
      this->build_options.push_back("-DDATA_TYPE=half");
    else
      this->build_options.push_back("-DDATA_TYPE=float");

    if (this->conv_param.bias_term) {
      this->build_options.push_back("-DBIAS");
    }
    switch(this->conv_param.activation) {
      case booster::ReLU:
        this->build_options.push_back("-DUSE_RELU");
        break;
      case booster::None:
        break;
      default:
        break;
    }
    return 0;
}

template <class Dtype>
int ConvLayerCL<Dtype>::SetKernelParameters()
{
    int error_num;
    size_t n_grp_size = this->channel_grp_size;
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

    if (this->conv_param.bias_term) {
      this->_weight_blobs[1]->AllocDevice(this->rt_param->context(), this->conv_param.oc_padded);
      std::vector<Dtype> bias_padding(this->conv_param.oc_padded, 0);
      memcpy(bias_padding.data(), this->bias_data, this->conv_param.output_channels * sizeof(Dtype));
      this->_weight_blobs[1]->WriteToDevice(this->rt_param->command_queue(), bias_padding.data(), this->conv_param.oc_padded);
      this->_weight_blobs[1]->Free();
    }

    this->kernels[0] = cl::Kernel(this->cl_programs[0], this->cl_kernel_functions[0].c_str(), &error_num);
    if (!checkSuccess(error_num)) {
      LOGE("Failed to create conv OpenCL kernels[0]. ");
      return 1;
    }

    booster::CLBuffers buffers;
    buffers.input_mem = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    buffers.weight_mem = this->_weight_blobs[0]->data_cl();
    buffers.output_mem = this->_top_blobs[this->_top[0]]->data_cl();
    buffers.bias_mem = this->_weight_blobs[1]->data_cl();
    buffers.input_trans_mem = nullptr;
    buffers.out_trans_mem = nullptr;
    this->conv_booster.SetConvKernelParams(this->conv_param, buffers, this->kernels, false);
    this->conv_booster.SetConvWorkSize(this->conv_param, conv_gws, conv_lws, this->kernels, this->rt_param->cl_runtime());

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
    this->_top_blobs[this->_top[0]]->ReshapeWithReallocDevice(this->rt_param->context(),
                                      this->_top_blobs[this->_top[0]]->num(),
                                      this->_top_blobs[this->_top[0]]->channels(),
                                      this->conv_param.output_h, this->conv_param.output_w);

    booster::CLBuffers buffers;
    buffers.input_mem = this->_bottom_blobs[this->_bottom[0]]->data_cl();
    buffers.output_mem = this->_top_blobs[this->_top[0]]->data_cl();
    this->conv_booster.SetConvKernelParams(this->conv_param, buffers, this->kernels, true);
    this->conv_booster.SetConvWorkSize(this->conv_param, conv_gws, conv_lws, this->kernels, this->rt_param->cl_runtime());
    return this->ForwardCL();
}

template <class Dtype>
int ConvLayerCL<Dtype>::ForwardCL()
{

    this->conv_booster.Forward(this->rt_param->command_queue(), this->events[0], this->kernels, this->conv_gws, this->conv_lws, this->cl_kernel_names[0]);
    return 0;
}


template <class Dtype>
int ConvLayerCL<Dtype>::GenerateTopBlobs() {
    //Conv layer has and only has one bottom blob.

    const Blob<Dtype> *bottom_blob = this->_bottom_blobs[this->_bottom[0]];

    this->conv_param.input_w = bottom_blob->width();
    this->conv_param.input_h = bottom_blob->height();
    this->conv_param.input_channels = bottom_blob->channels();
    this->conv_param.AssignOutputDim();

    this->_top_blobs[this->_top[0]] = new Blob<Dtype>(1, this->conv_param.output_channels, this->conv_param.output_h, this->conv_param.output_w);
    this->_top_blobs[this->_top[0]]->AllocDevice(this->rt_param->context(), this->_top_blobs[this->_top[0]]->data_size_padded_c());

    this->conv_param.oc_padded = this->_top_blobs[this->_top[0]]->get_channels_padding();
    this->conv_param.ic_padded = this->_bottom_blobs[this->_bottom[0]]->get_channels_padding();
    this->channel_grp_size = 4;
    if (this->conv_param.ic_padded % 8 == 0 && this->conv_param.oc_padded % 8 == 0) {
      this->channel_grp_size = 8;
    }

    this->conv_booster.SelectAlgo(&this->conv_param);
    this->conv_booster.Init(this->cl_kernel_names, this->cl_kernel_symbols, this->cl_kernel_functions);
    return 0;
}


template <class Dtype>
int ConvLayerCL<Dtype>::Fuse(Layer<Dtype> *next_layer)
{
  if (next_layer->type().compare("ReLU") == 0)
  {
    this->conv_param.activation = booster::ReLU;
    return 1;
  } else {
    return 0;
  }
}

template class ConvLayerCL<float>;
template class ConvLayerCL<uint16_t>;

}; // namespace feather
