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
#include "layers/input_layer.h"
#include "layers/conv_layer.h"
#include "layers/pooling_layer.h"
#include "layers/relu_layer.h"
#include "layers/inner_product_layer.h"
#include "layers/dropout_layer.h"
#include "layers/softmax_layer.h"
#include "layers/batchnorm_layer.h"
#include "layers/scale_layer.h"
#include "layers/split_layer.h"
#include "layers/eltwise_layer.h"
#include "layers/concat_layer.h"

#include <stdio.h>

namespace feather
{
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
DEFINE_LAYER_CREATOR(BatchNorm)
DEFINE_LAYER_CREATOR(Scale)
DEFINE_LAYER_CREATOR(Split)
DEFINE_LAYER_CREATOR(Eltwise)
// DEFINE_LAYER_CREATOR(Concat)

void register_layer_creators()
{
    REGISTER_LAYER_CREATOR(Input, Input);
    REGISTER_LAYER_CREATOR(Convolution, Conv);
    REGISTER_LAYER_CREATOR(ReLU, Relu);
    REGISTER_LAYER_CREATOR(Pooling, Pooling);
    REGISTER_LAYER_CREATOR(InnerProduct, InnerProduct);
    REGISTER_LAYER_CREATOR(Dropout, Dropout);
    REGISTER_LAYER_CREATOR(Softmax, Softmax);
    REGISTER_LAYER_CREATOR(BatchNorm, BatchNorm);
    REGISTER_LAYER_CREATOR(Scale, Scale);
    REGISTER_LAYER_CREATOR(Split, Split);
    REGISTER_LAYER_CREATOR(Eltwise, Eltwise);
    // REGISTER_LAYER_CREATOR(Concat, Concat);
}
};
