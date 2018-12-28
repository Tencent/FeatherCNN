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

#ifdef FEATHER_OPENCL

#include <stdio.h>

#include "feather_generated.h"

#include "layer_factory_cl.h"

#include "layer.h"


#include <layers_cl/input_layer_cl.h>
#include <layers_cl/conv_layer_cl.h>
#include <layers_cl/pooling_layer_cl.h>
#include <layers_cl/relu_layer_cl.h>
#include <layers_cl/elewise_layer_cl.h>
#include <layers_cl/inner_product_layer_cl.h>


namespace feather
{

template <class Dtype>
Layer<Dtype> *GetInputLayerCL(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
{
    return (Layer<Dtype> *) new InputLayerCL<Dtype>(layer_param, rt_param);
}

template <class Dtype>
Layer<Dtype> *GetConvolutionLayerCL(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
{
    return (Layer<Dtype> *) new ConvLayerCL<Dtype>(layer_param, rt_param);
}

template <class Dtype>
Layer<Dtype> *GetPoolingLayerCL(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
{
    return (Layer<Dtype> *)new PoolingLayerCL<Dtype>(layer_param, rt_param);
}
template <class Dtype>
Layer<Dtype> *GetReluLayerCL(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
{
    return (Layer<Dtype> *)new ReluLayerCL<Dtype>(layer_param, rt_param);
}
template <class Dtype>
Layer<Dtype> *GetEltwiseLayerCL(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
{
    return (Layer<Dtype> *) new EltwiseLayerCL<Dtype>(layer_param, rt_param);
}

template <class Dtype>
Layer<Dtype> *GetInnerProductLayerCL(const LayerParameter *layer_param, RuntimeParameter<Dtype> * rt_param)
{
    return (Layer<Dtype> *)new InnerProductLayerCL<Dtype>(layer_param, rt_param);
}

void register_layer_creators_cl()
{
    REGISTER_LAYER_CREATOR_CL(Input, GetInputLayerCL);
    REGISTER_LAYER_CREATOR_CL(Convolution, GetConvolutionLayerCL);
    REGISTER_LAYER_CREATOR_CL(Pooling, GetPoolingLayerCL);
    REGISTER_LAYER_CREATOR_CL(ReLU, GetReluLayerCL);
    REGISTER_LAYER_CREATOR_CL(InnerProduct, GetInnerProductLayerCL);
    REGISTER_LAYER_CREATOR_CL(DepthwiseConvolution, GetConvolutionLayerCL);
    REGISTER_LAYER_CREATOR_CL(Eltwise, GetEltwiseLayerCL);

}

};

#endif
