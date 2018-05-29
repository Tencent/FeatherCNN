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

#include "layer_factory.h"

#include "layer.h"
#include "layers/input_layer.h"
#include "layers/conv_layer.h"
#include "layers/conv_depthwise_layer.h"
#include "layers/conv_im2col_layer.h"
#include "layers/conv_winograd_layer.h"
#include "layers/conv_winogradF63_layer.h"
#include "layers/dropout_layer.h"
#include "layers/batchnorm_layer.h"
#include "layers/lrn_layer.h"
#include "layers/relu_layer.h"
#include "layers/prelu_layer.h"
#include "layers/scale_layer.h"
#include "layers/slice_layer.h"
#include "layers/pooling_layer.h"
#include "layers/eltwise_layer.h"
#include "layers/inner_product_layer.h"
#include "layers/softmax_layer.h"
#include "layers/concat_layer.h"

#include <stdio.h>

namespace feather
{
Layer *GetInputLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new InputLayer(layer_param, rt_param);
}
Layer *GetConvolutionLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    const ConvolutionParameter *conv_param = layer_param->convolution_param();
    size_t group = conv_param->group();
    size_t kernel_height = conv_param->kernel_h();
    size_t kernel_width = conv_param->kernel_w();
    size_t stride_height = conv_param->stride_h();
    size_t stride_width = conv_param->stride_w();
    size_t input_channels = layer_param->blobs()->Get(0)->channels();
    size_t output_channels = layer_param->blobs()->Get(0)->num();
    ConvLayer *conv_layer = NULL;
    if (group == 1 && kernel_height == 3 && kernel_width == 3 && stride_height == 1 && stride_width == 1 && input_channels > 0 && output_channels < 512)
    {
#if 0
        conv_layer = (ConvLayer*) new ConvWinogradLayer(layer_param, rt_param);
#else
        conv_layer = (ConvLayer*) new ConvWinogradF63Layer(layer_param, rt_param);
#endif
    }
    else if (group == 1 && kernel_height == 3 && kernel_width == 3 && stride_height == 1 && stride_width == 1 && input_channels > 4)
    {
        conv_layer = (ConvLayer*) new ConvWinogradLayer(layer_param, rt_param);
    }
    else if (group == 1)
    {
        conv_layer = (ConvLayer*) new ConvIm2colLayer(layer_param, rt_param);
    }
    else//Should be depthwise convolution layer.
    {
        conv_layer = new ConvDepthwiseLayer(layer_param, rt_param);
    }
    return (Layer *) conv_layer;
}
Layer *GetBatchNormLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new BatchNormLayer(layer_param, rt_param);
}
Layer *GetLRNLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new LRNLayer(layer_param, rt_param);
}
Layer *GetConcatLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new ConcatLayer(layer_param, rt_param);
}
Layer *GetDropoutLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new DropoutLayer(layer_param, rt_param);
}
Layer *GetReluLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new ReluLayer(layer_param, rt_param);
}
Layer *GetPReluLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new PReluLayer(layer_param, rt_param);
}
Layer *GetScaleLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new ScaleLayer(layer_param, rt_param);
}
Layer *GetSliceLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new SliceLayer(layer_param, rt_param);
}
Layer *GetPoolingLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new PoolingLayer(layer_param, rt_param);
}
Layer *GetEltwiseLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new EltwiseLayer(layer_param, rt_param);
}
Layer *GetInnerProductLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new InnerProductLayer(layer_param, rt_param);
}
Layer *GetSoftmaxLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new SoftmaxLayer(layer_param, rt_param);
}

void register_layer_creators()
{
    REGISTER_LAYER_CREATOR(Input, GetInputLayer);
    REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer);
    REGISTER_LAYER_CREATOR(BatchNorm, GetBatchNormLayer);
    REGISTER_LAYER_CREATOR(LRN, GetLRNLayer);
    REGISTER_LAYER_CREATOR(Concat, GetConcatLayer);
    REGISTER_LAYER_CREATOR(Dropout, GetDropoutLayer);
    REGISTER_LAYER_CREATOR(ReLU, GetReluLayer);
    REGISTER_LAYER_CREATOR(PReLU, GetPReluLayer);
    REGISTER_LAYER_CREATOR(Scale, GetScaleLayer);
    REGISTER_LAYER_CREATOR(Slice, GetSliceLayer);
    REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayer);
    REGISTER_LAYER_CREATOR(Eltwise, GetEltwiseLayer);
    REGISTER_LAYER_CREATOR(InnerProduct, GetInnerProductLayer);
    REGISTER_LAYER_CREATOR(Softmax, GetSoftmaxLayer);
}
};
