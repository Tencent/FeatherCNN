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
// #include "layers_cl/input_layer.h"
#include <layers_cl/input_layer_cl.h>
#include <layers_cl/direct_conv_layer_cl.h>
#include <layers_cl/pooling_layer_cl.h>
#include <layers_cl/relu_layer_cl.h>
#include <layers_cl/elewise_layer_cl.h>
#include <layers_cl/inner_product_layer_cl.h>


namespace feather
{
Layer<uint16_t> *GetInputLayerCL(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<uint16_t> *) new InputLayerCL(layer_param, rt_param);
}

Layer<uint16_t> *GetConvolutionLayerCL(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<uint16_t> *) new DirectConvLayerCL(layer_param, rt_param);
}
Layer<uint16_t> *GetPoolingLayerCL(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<uint16_t> *)new PoolingLayerCL(layer_param, rt_param);
}

Layer<uint16_t> *GetReluLayerCL(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<uint16_t> *)new ReluLayerCL(layer_param, rt_param);
}

Layer<uint16_t> *GetEltwiseLayerCL(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<uint16_t> *) new EltwiseLayerCL(layer_param, rt_param);
}
Layer<uint16_t> *GetInnerProductLayerCL(const LayerParameter *layer_param, RuntimeParameter<float> * rt_param)
{
    return (Layer<uint16_t> *)new InnerProductLayerCL(layer_param, rt_param);
}

void register_layer_creators_cl()
{
    REGISTER_LAYER_CREATOR_CL(Input, GetInputLayerCL);
    REGISTER_LAYER_CREATOR_CL(Convolution, GetConvolutionLayerCL);
    REGISTER_LAYER_CREATOR_CL(Pooling, GetPoolingLayerCL);
    REGISTER_LAYER_CREATOR_CL(ReLU, GetReluLayerCL);
    REGISTER_LAYER_CREATOR_CL(InnerProduct, GetInnerProductLayerCL);
}

};

#endif
