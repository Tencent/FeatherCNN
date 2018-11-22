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

#include "layer_factory.h"

#include "layer.h"
#include "layers/input_layer.h"
#include "layers/conv_layer.h"
#include "layers/flatten_layer.h"
#include "layers/dropout_layer.h"
#include "layers/batchnorm_layer.h"
#include "layers/lrn_layer.h"
#include "layers/relu_layer.h"
#include "layers/prelu_layer.h"
#include "layers/scale_layer.h"
#include "layers/slice_layer.h"
#include "layers/pooling_layer.h"
#include "layers/eltwise_layer.h"
#include "layers/interp_layer.h"
#include "layers/inner_product_layer.h"
#include "layers/reshape_layer.h"

#include "layers/softmax_layer.h"
#include "layers/concat_layer.h"
#include "layers/filter_layer.h"

#include <stdio.h>

namespace feather
{
Layer<float> *GetInputLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<float> *)new InputLayer(layer_param, rt_param);
}
Layer<float> *GetConvolutionLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<float> *) new ConvLayer(layer_param, rt_param);
}
Layer<float> *GetDepthwiseConvolutionLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<float> *)new ConvLayer(layer_param, rt_param);
}
Layer<float> *GetBatchNormLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<float> *)new BatchNormLayer(layer_param, rt_param);
}
Layer<float> *GetLRNLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<float> *)new LRNLayer(layer_param, rt_param);
}
Layer<float> *GetConcatLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<float> *)new ConcatLayer(layer_param, rt_param);
}
Layer<float> *GetDropoutLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<float> *)new DropoutLayer(layer_param, rt_param);
}
Layer<float> *GetReluLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<float> *)new ReluLayer(layer_param, rt_param);
}
Layer<float> *GetPReluLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<float> *)new PReluLayer(layer_param, rt_param);
}
Layer<float> *GetScaleLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<float> *)new ScaleLayer(layer_param, rt_param);
}
Layer<float> *GetSliceLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<float> *)new SliceLayer(layer_param, rt_param);
}
Layer<float> *GetPoolingLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<float> *)new PoolingLayer(layer_param, rt_param);
}
Layer<float> *GetEltwiseLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<float> *)new EltwiseLayer(layer_param, rt_param);
}
Layer<float> *GetInterpLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<float> *)new InterpLayer(layer_param, rt_param);
}

Layer<float> *GetInnerProductLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<float> *)new InnerProductLayer(layer_param, rt_param);
}
Layer<float> *GetFilterLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<float> *)new FilterLayer(layer_param, rt_param);
}
Layer<float> *GetFlattenLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<float> *)new FlattenLayer(layer_param, rt_param);
}
Layer<float> *GetSoftmaxLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<float> *)new SoftmaxLayer(layer_param, rt_param);
}

Layer<float> *GetReshapeLayer(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<float> *)new ReshapeLayer(layer_param, rt_param);
}

void register_layer_creators()
{
    REGISTER_LAYER_CREATOR(Input, GetInputLayer);
    REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer);
    // REGISTER_LAYER_CREATOR(DepthwiseConvolution, GetDepthwiseConvolutionLayer);
    // REGISTER_LAYER_CREATOR(BatchNorm, GetBatchNormLayer);
    // REGISTER_LAYER_CREATOR(LRN, GetLRNLayer);
    // REGISTER_LAYER_CREATOR(Concat, GetConcatLayer);
    // REGISTER_LAYER_CREATOR(Dropout, GetDropoutLayer);
    // REGISTER_LAYER_CREATOR(ReLU, GetReluLayer);
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
}
};
