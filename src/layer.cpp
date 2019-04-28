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

const Blob<float>* Layer::FindBottomByName(std::string name)
{
    for( int i = 0; i < this->bottoms.size(); ++i)
    {
        if (this->bottoms[i]->name.compare(name) == 0)
        {
            return this->bottoms[i];
        }
    }
    return NULL;
}

Blob<float>* Layer::FindTopByName(std::string name)
{
    for( int i = 0; i < this->tops.size(); ++i)
    {
        if (this->tops[i]->name.compare(name) == 0)
        {
            return tops[i];
        }
    }
    return NULL;
}

// int Layer::SetupBottomBlob(const Blob<float>* p_blob, std::string name)
// {
//     if (std::find(bottoms.begin(), bottoms.end(), name) == bottoms.end())
//         return -1;
//     bottom_blobs[name] = p_blob;
//     return 0;
// }


// int Layer::ReplaceBottomBlob(std::string old_bottom, std::string new_bottom, const Blob<float>* p_blob)
// {
//     //printf("*old bottom %s to new bottom %s\n", old_bottom.c_str(), new_bottom.c_str());
//     std::vector<std::string>::iterator name_iter = _bottom.begin();
//     typedef typename std::map<std::string, const Blob<float>* >::iterator STDMAPITER;
//     STDMAPITER blob_iter = _bottom_blobs.begin();

//     name_iter = std::find(_bottom.begin(), _bottom.end(), old_bottom);
//     blob_iter = _bottom_blobs.find(old_bottom);

//     if (name_iter == _bottom.end() || blob_iter == _bottom_blobs.end())
//         return -1;

//     *name_iter = new_bottom;//should not change order
//     _bottom_blobs.erase(blob_iter);

//     _bottom_blobs[new_bottom] = p_blob;
//     //printf("+old bottom %s to new bottom %s\n", old_bottom.c_str(), new_bottom.c_str());
//     return 0;

// }


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


int Layer::GenerateTopBlobs()
{
    /* GenerateTopBlobs
     * infers top blob shape and allocate space.
     * The default behavior is allocate a top with the same shape of bottom
     */
    if (tops.size() != 1 || bottoms.size() != 1)
        return -1;
    Blob<float>* p_blob = new Blob<float>();
    p_blob->CopyShape(bottoms[0]);
    p_blob->Alloc();
    tops[0] = p_blob;
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
