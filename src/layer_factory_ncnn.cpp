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

#include "layer_factory.h"
// #include "layers/input_layer.h"
// #include "layers/conv_layer.h"
// #include "layers/flatten_layer.h"
// #include "layers/dropout_layer.h"
// #include "layers/batchnorm_layer.h"
// #include "layers/lrn_layer.h"
// #include "layers/relu_layer.h"
// #include "layers/leaky_relu_layer.h"
// #include "layers/prelu_layer.h"
// #include "layers/scale_layer.h"
// #include "layers/slice_layer.h"
// #include "layers/pooling_layer.h"
// #include "layers/eltwise_layer.h"
// #include "layers/interp_layer.h"
// #include "layers/inner_product_layer.h"
// #include "layers/reshape_layer.h"
// #include "layers/yolov2_reorg_layer.h"

// #include "layers/softmax_layer.h"
// #include "layers/concat_layer.h"
// #include "layers/filter_layer.h"

#include <stdio.h>

namespace feather
{
// template <class Dtype>
// Layer<Dtype> *GetInputLayer(RuntimeParameter<Dtype> * rt_param)
// {
//     return (Layer<Dtype> *)new InputLayer(layer_param, rt_param);
// }

// template <class Dtype>
// Layer<Dtype> *GetConvolutionLayer(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
// {
//     return (Layer<Dtype> *) new ConvLayer(layer_param, rt_param);
// }

// template <class Dtype>
// Layer<Dtype> *GetDepthwiseConvolutionLayer(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
// {
//     return (Layer<Dtype> *)new ConvLayer(layer_param, rt_param);
// }

// template <class Dtype>
// Layer<Dtype> *GetBatchNormLayer(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
// {
//     return (Layer<Dtype> *)new BatchNormLayer(layer_param, rt_param);
// }

// template <class Dtype>
// Layer<Dtype> *GetLRNLayer(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
// {
//     return (Layer<Dtype> *)new LRNLayer(layer_param, rt_param);
// }

// template <class Dtype>
// Layer<Dtype> *GetConcatLayer(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
// {
//     return (Layer<Dtype> *)new ConcatLayer(layer_param, rt_param);
// }

// template <class Dtype>
// Layer<Dtype> *GetDropoutLayer(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
// {
//     return (Layer<Dtype> *)new DropoutLayer(layer_param, rt_param);
// }

// template <class Dtype>
// Layer<Dtype> *GetReluLayer(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
// {
//     return (Layer<Dtype> *)new ReluLayer(layer_param, rt_param);
// }

// template <class Dtype>
// Layer<Dtype> *GetLeakyReluLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
// {
//     return (Layer<Dtype> *)new LeakyReluLayer(layer_param, rt_param);
// }
// template <class Dtype>
// Layer<Dtype> *GetPReluLayer(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
// {
//     return (Layer<Dtype> *)new PReluLayer(layer_param, rt_param);
// }

// template <class Dtype>
// Layer<Dtype> *GetScaleLayer(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
// {
//     return (Layer<Dtype> *)new ScaleLayer(layer_param, rt_param);
// }

// template <class Dtype>
// Layer<Dtype> *GetSliceLayer(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
// {
//     return (Layer<Dtype> *)new SliceLayer(layer_param, rt_param);
// }

// template <class Dtype>
// Layer<Dtype> *GetPoolingLayer(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
// {
//     return (Layer<Dtype> *)new PoolingLayer(layer_param, rt_param);
// }

// template <class Dtype>
// Layer<Dtype> *GetEltwiseLayer(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
// {
//     return (Layer<Dtype> *)new EltwiseLayer(layer_param, rt_param);
// }

// template <class Dtype>
// Layer<Dtype> *GetInterpLayer(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
// {
//     return (Layer<Dtype> *)new InterpLayer(layer_param, rt_param);
// }
// template <class Dtype>
// Layer<Dtype> *GetInnerProductLayer(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
// {
//     return (Layer<Dtype> *)new InnerProductLayer(layer_param, rt_param);
// }

// template <class Dtype>
// Layer<Dtype> *GetFilterLayer(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
// {
//     return (Layer<Dtype> *)new FilterLayer(layer_param, rt_param);
// }

// template <class Dtype>
// Layer<Dtype> *GetFlattenLayer(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
// {
//     return (Layer<Dtype> *)new FlattenLayer(layer_param, rt_param);
// }

// template <class Dtype>
// Layer<Dtype> *GetSoftmaxLayer(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
// {
//     return (Layer<Dtype> *)new SoftmaxLayer(layer_param, rt_param);
// }

// template <class Dtype>
// Layer<Dtype> *GetReshapeLayer(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
// {
//     return (Layer<Dtype> *)new ReshapeLayer(layer_param, rt_param);
// }

// template <class Dtype>
// Layer<Dtype> *GetYolov2ReorgLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
// {
//     return (Layer<Dtype> *)new Yolov2ReorgLayer(layer_param, rt_param);
// }

void InputLayer()
{
    ;
}
void ConvLayer()
{
    ;
}
void ReluLayer()
{
    ;
}
void PoolingLayer()
{
    ;
}
void InnerProductLayer()
{
    ;
}
void DropoutLayer()
{
    ;
}
void SoftmaxLayer()
{
    ;
}

/* An example to register a layer:
 *
 * feather layer name: ConvLayer, ncnn type name: Convolution
 * 1. Define a layer creator: DEFINE_LAYER_CREATOR(Conv)
 * 2. Register layer in the register_layer_creators() function: REGISTER_LAYER_CREATOR(Convolution, Conv);
 */

DEFINE_LAYER_CREATOR(Input)
DEFINE_LAYER_CREATOR(Conv)
DEFINE_LAYER_CREATOR(Relu)
DEFINE_LAYER_CREATOR(Pooling)
DEFINE_LAYER_CREATOR(InnerProduct)
DEFINE_LAYER_CREATOR(Dropout)
DEFINE_LAYER_CREATOR(Softmax)

void register_layer_creators()
{
    REGISTER_LAYER_CREATOR(Input, Input);
    REGISTER_LAYER_CREATOR(Convolution, Conv);
    REGISTER_LAYER_CREATOR(ReLU, Relu);
    REGISTER_LAYER_CREATOR(Pooling, Pooling);
    REGISTER_LAYER_CREATOR(InnerProduct, InnerProduct);
    REGISTER_LAYER_CREATOR(Dropout, Dropout);
    REGISTER_LAYER_CREATOR(Softmax, Softmax);
    // REGISTER_LAYER_CREATOR(Input, GetInputLayer);
    // REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer);
    // REGISTER_LAYER_CREATOR(DepthwiseConvolution, GetDepthwiseConvolutionLayer);
    // REGISTER_LAYER_CREATOR(BatchNorm, GetBatchNormLayer);
    // REGISTER_LAYER_CREATOR(LRN, GetLRNLayer);
    // REGISTER_LAYER_CREATOR(Concat, GetConcatLayer);
    // REGISTER_LAYER_CREATOR(Dropout, GetDropoutLayer);
    // REGISTER_LAYER_CREATOR(ReLU, GetReluLayer);
    // REGISTER_LAYER_CREATOR(LeakyRelu, GetLeakyReluLayer);
    // REGISTER_LAYER_CREATOR(PReLU, GetPReluLayer);
    // REGISTER_LAYER_CREATOR(Scale, GetScaleLayer);
    // REGISTER_LAYER_CREATOR(Slice, GetSliceLayer);
    // REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayer);
    // REGISTER_LAYER_CREATOR(Eltwise, GetEltwiseLayer);
    // REGISTER_LAYER_CREATOR(Flatten, GetFlattenLayer);
    // REGISTER_LAYER_CREATOR(Interp, GetInterpLayer);
    // REGISTER_LAYER_CREATOR(InnerProduct, GetInnerProductLayer);
    // REGISTER_LAYER_CREATOR(Softmax, GetSoftmaxLayer);
    // REGISTER_LAYER_CREATOR(Filter, GetFilterLayer);
    // REGISTER_LAYER_CREATOR(Reshape, GetReshapeLayer);
    // REGISTER_LAYER_CREATOR(Yolov2Reorg, GetYolov2ReorgLayer);
}
};
