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

#ifndef _WIN32
#include <unistd.h>
#endif

#include "layer.h"

namespace feather
{
Layer::Layer(RuntimeParameter<float>* rt_param)
    : _fusible(false),
      _inplace(false),
      rt_param(rt_param),
      common_mempool(rt_param->common_mempool())
{
}

Layer::~Layer()
{
    if (!_inplace)
    {
        for (int i = 0; i < tops.size(); ++i)
        {
            delete tops[i];
        }
    }
    for (int i = 0; i < weights.size(); ++i)
    {
        delete weights[i];
    }
}

int Layer::FindBottomIDByName(std::string name)
{
    for( int i = 0; i < this->bottoms.size(); ++i)
    {
        if (this->bottoms[i]->name.compare(name) == 0)
        {
            return i;
        }
    }
    return -1;
}

int Layer::FindTopIDByName(std::string name)
{
    for( int i = 0; i < this->tops.size(); ++i)
    {
        if (this->tops[i]->name.compare(name) == 0)
        {
            return i;
        }
    }
    return -1;
}

int Layer::LoadParam(const ncnn::ParamDict& pd)
{
    // Do nothing.
    return 0;
}

int Layer::LoadWeights(const ncnn::ModelBin& mb)
{
    // Do nothing
    return 0;
}

int Layer::TryFuse(Layer *next_layer)
{
    //Judge if next_layer points to this layer.
    for (int i = 0; i < next_layer->bottoms.size(); ++i)
    {
        for (int j = 0; j < this->tops.size(); ++j)
        {
            if (this->tops[j]->name.compare(next_layer->bottoms[i]->name) == 0)
            {
                return Fuse(next_layer);
            }
        }
    }
    return 0;
}

int Layer::Fuse(Layer* next_layer)
{
    return 0;
}

int Layer::Reshape()
{
    /* GenerateTopBlobs
     * infers top blob shape and allocate space.
     * 
     * The default behavior is allocate a top with the same shape of bottom
     */
    if (tops.size() != 1 || bottoms.size() != 1)
        return -400; //False calling base layer.
    tops[0]->ReshapeWithRealloc(bottoms[0]->num(), bottoms[0]->channels(), bottoms[0]->height(), bottoms[0]->width());
    return 0;
}

int Layer::Init()
{
    return 0;
}

int Layer::Forward()
{
    return false;
}

int Layer::ForwardReshape()
{
    //Default Reshape Assertation:
    //
    //There should be a single top blob as well as bottom blob.
    //The default behaviour is that the top blob is identical to the bottom blob
    //Use default reallocation.
    
    tops[0]->ReshapeWithRealloc(bottoms[0]);
    this->Forward();
    return true;
}

bool Layer::fusible() const
{
    return _fusible;
}
};