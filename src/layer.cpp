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

#include "layer.h"
#include "feather_simple_generated.h"//For LayerParameter


namespace feather
{
Layer::Layer(const void* layer_param_in, const RuntimeParameter<float>* rt_param)
    : _fusible(false),
      num_threads(rt_param->num_threads()),
      common_mempool(rt_param->common_mempool())
{
    const LayerParameter* layer_param = (const LayerParameter*)layer_param_in;
    _name = layer_param->name()->str();
    _type = layer_param->type()->str();

    for (int i = 0; i < VectorLength(layer_param->bottom()); ++i)
        _bottom.push_back(layer_param->bottom()->Get(i)->str());

    for (int i = 0; i < VectorLength(layer_param->top()); ++i)
        _top.push_back(layer_param->top()->Get(i)->str());

    size_t blob_num = VectorLength(layer_param->blobs());

    /* Construct weight blobs */
    for (int i = 0; i < blob_num; ++i)
    {
        const BlobProto* proto = (const BlobProto*) layer_param->blobs()->Get(i);
        Blob<float>* p_blob = new Blob<float>();
        p_blob->FromProto(layer_param->blobs()->Get(i));
        _weight_blobs.push_back(p_blob);
    }
}

int Layer::SetupBottomBlob(const Blob<float>* p_blob, std::string name)
{
    if (std::find(_bottom.begin(), _bottom.end(), name) == _bottom.end())
        return -1;
    _bottom_blobs[name] = p_blob;
    return 0;
}

int Layer::ReplaceBottomBlob(std::string old_bottom, std::string new_bottom, const Blob<float>* p_blob)
{
    //printf("*old bottom %s to new bottom %s\n", old_bottom.c_str(), new_bottom.c_str());
    std::vector<std::string>::iterator name_iter = _bottom.begin();
    std::map<std::string, const Blob<float>*>::iterator blob_iter = _bottom_blobs.begin();

    name_iter = std::find(_bottom.begin(), _bottom.end(), old_bottom);
    blob_iter = _bottom_blobs.find(old_bottom);

    if (name_iter == _bottom.end() || blob_iter == _bottom_blobs.end())
        return -1;

    *name_iter = new_bottom;//should not change order
    _bottom_blobs.erase(blob_iter);

    _bottom_blobs[new_bottom] = p_blob;
    //printf("+old bottom %s to new bottom %s\n", old_bottom.c_str(), new_bottom.c_str());
    return 0;

}

int Layer::TryFuse(Layer *next_layer)
{
    //Judge if next_layer points to this layer.
    for (int i = 0; i < next_layer->bottom_size(); ++i)
    {
        for (int j = 0; j < this->top_size(); ++j)
        {
            if (this->top(j).compare(next_layer->bottom(i)) == 0)
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
    if (_top.size() != 1 || _bottom.size() != 1)
        return -1;
    Blob<float>* p_blob = new Blob<float>();
    p_blob->CopyShape(_bottom_blobs[_bottom[0]]);
    p_blob->Alloc();
    _top_blobs[_top[0]] = p_blob;
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
std::string Layer::name()
{
    return _name;
}
std::string Layer::type()
{
    return _type;
}
std::string Layer::bottom(size_t i)
{
    return i >= _bottom.size() ? std::string() : _bottom[i];
}
size_t Layer::bottom_size()
{
    return _bottom.size();
}
std::string Layer::top(size_t i)
{
    return i >= _top.size() ? std::string() : _top[i];
}
size_t Layer::top_size()
{
    return _top.size();
}
size_t Layer::top_blob_size()
{
    return _top_blobs.size();
}
const Blob<float>* Layer::top_blob(std::string name)
{
    if (_top_blobs.find(name) != _top_blobs.end())
        return _top_blobs[name];
    else
        return NULL;
}
const Blob<float>* Layer::top_blob(size_t idx)
{
    std::string name = this->top(idx);
    return top_blob(name);
}
const size_t Layer::weight_blob_num() const
{
    return _weight_blobs.size();
}
const Blob<float>* Layer::weight_blob(size_t i) const
{
    return i > _weight_blobs.size() ? NULL : _weight_blobs[i];
}
bool Layer::fusible() const
{
    return _fusible;
}
};
