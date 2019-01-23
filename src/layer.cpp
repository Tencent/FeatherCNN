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
#include "feather_generated.h"//For LayerParameter


namespace feather
{
template<class Dtype>
Layer<Dtype>::Layer(const void* layer_param_in, RuntimeParameter<Dtype>* rt_param)
    : _fusible(false),
      _inplace(false),
      rt_param(rt_param),
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
        Blob<Dtype>* p_blob = new Blob<Dtype>();
        p_blob->FromProto(layer_param->blobs()->Get(i));
        _weight_blobs.push_back(p_blob);
    }
}

template<class Dtype>
Layer<Dtype>::~Layer()
{
    if (!_inplace)
    {
        for (int i = 0; i < _top_blobs.size(); ++i)
        {
            delete _top_blobs[top(i)];
        }
    }
    for (int i = 0; i < _weight_blobs.size(); ++i)
    {
        delete _weight_blobs[i];
    }
}

template<class Dtype>
int Layer<Dtype>::SetupBottomBlob(const Blob<Dtype>* p_blob, std::string name)
{
    if (std::find(_bottom.begin(), _bottom.end(), name) == _bottom.end())
        return -1;
    _bottom_blobs[name] = p_blob;
    return 0;
}

template<class Dtype>
int Layer<Dtype>::ReplaceBottomBlob(std::string old_bottom, std::string new_bottom, const Blob<Dtype>* p_blob)
{
    //printf("*old bottom %s to new bottom %s\n", old_bottom.c_str(), new_bottom.c_str());
    std::vector<std::string>::iterator name_iter = _bottom.begin();
    typedef typename std::map<std::string, const Blob<Dtype>* >::iterator STDMAPITER;
    STDMAPITER blob_iter = _bottom_blobs.begin();

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

template<class Dtype>
int Layer<Dtype>::TryFuse(Layer *next_layer)
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

template<class Dtype>
int Layer<Dtype>::Fuse(Layer* next_layer)
{
    return 0;
}

template<class Dtype>
int Layer<Dtype>::GenerateTopBlobs()
{
    if (_top.size() != 1 || _bottom.size() != 1)
        return -1;
    Blob<Dtype>* p_blob = new Blob<Dtype>();
    p_blob->CopyShape(_bottom_blobs[_bottom[0]]);
    p_blob->Alloc();

    _top_blobs[_top[0]] = p_blob;
    return 0;
}

template<class Dtype>
int Layer<Dtype>::Init()
{
    return 0;
}

template<class Dtype>
int Layer<Dtype>::Forward()
{
    return false;
}

template<class Dtype>
int Layer<Dtype>::ForwardReshape()
{
    //Default Reshape Assertation:
    //There should be a single top blob as well as bottom blob.
    //The default behaviour is that the top blob is identical to the bottom blob
    //Use default reallocation.
    _top_blobs[top(0)]->ReshapeWithRealloc(_bottom_blobs[bottom(0)]);

    this->Forward();
    return true;
}

template<class Dtype>
std::string Layer<Dtype>::name()
{
    return _name;
}

template<class Dtype>
std::string Layer<Dtype>::type()
{
    return _type;
}

template<class Dtype>
std::string Layer<Dtype>::bottom(size_t i)
{
    return i >= _bottom.size() ? std::string() : _bottom[i];
}

template<class Dtype>
size_t Layer<Dtype>::bottom_size()
{
    return _bottom.size();
}

template<class Dtype>
std::string Layer<Dtype>::top(size_t i)
{
    return i >= _top.size() ? std::string() : _top[i];
}

template<class Dtype>
size_t Layer<Dtype>::top_size()
{
    return _top.size();
}

template<class Dtype>
size_t Layer<Dtype>::top_blob_size()
{
    return _top_blobs.size();
}

template<class Dtype>
const Blob<Dtype>* Layer<Dtype>::top_blob(std::string name)
{
    if (_top_blobs.find(name) != _top_blobs.end())
        return _top_blobs[name];
    else
        return NULL;
}

template<class Dtype>
const Blob<Dtype>* Layer<Dtype>::top_blob(size_t idx)
{
    std::string name = this->top(idx);
    return top_blob(name);
}

template<class Dtype>
const Blob<Dtype>* Layer<Dtype>::bottom_blob(size_t idx)
{
    std::string name = this->bottom(idx);
    return _bottom_blobs[name];
}

template<class Dtype>
const size_t Layer<Dtype>::weight_blob_num() const
{
    return _weight_blobs.size();
}

template<class Dtype>
const Blob<Dtype>* Layer<Dtype>::weight_blob(size_t i) const
{
    return i > _weight_blobs.size() ? NULL : _weight_blobs[i];
}

template<class Dtype>
bool Layer<Dtype>::fusible() const
{
    return _fusible;
}

#ifdef FEATHER_OPENCL

template <class Dtype>
int Layer<Dtype>::InitKernelInfo(std::string kname, std::string pname)
{
    std::string program_name = pname;
    std::string kernel_name = kname;
    auto it_source = booster::opencl_kernel_string_map.find(pname);
    if (it_source != booster::opencl_kernel_string_map.end())
    {
        this->cl_kernel_info_map[kname].program_name = pname;
        this->cl_kernel_info_map[kname].kernel_name = kname;
        this->cl_kernel_info_map[kname].kernel_source = std::string(it_source->second.begin(), it_source->second.end());

    }
    else
    {
        LOGE("can't find program %s!", pname.c_str());
        return -1;
    }
    return 0;
}

template<class Dtype>
int Layer<Dtype>::SetBuildOptions()
{
    //Base layer doesn't know settings.
    return -1;
}

template<class Dtype>
int Layer<Dtype>::SetKernelParameters()
{
    //Base layer doesn't know settings.
    return -1;
}

template<class Dtype>
int Layer<Dtype>::ResetWorkSize(std::string kname, size_t output_height, size_t output_width)
{
    clhpp_feather::CLKernelInfo& kernel_info = this->cl_kernel_info_map[kname];

    int h_lws = output_height > 32 ? 16 : 8;
    int w_lws = output_height > 32 ? 16 : 8;

    size_t gws_dim0 = (output_height / h_lws + !!(output_height % h_lws)) * h_lws;
    size_t gws_dim1 = (output_width / w_lws  + !!(output_width % w_lws)) * w_lws;

    size_t lws_dim0 = h_lws;
    size_t lws_dim1 = w_lws;

    kernel_info.gws[0] = gws_dim0;
    kernel_info.gws[1] = gws_dim0;
    kernel_info.lws[0] = lws_dim0;
    kernel_info.lws[1] = lws_dim0;

    return 0;
}

template <class Dtype>
int Layer<Dtype>::SetWorkSize(std::string kname, size_t output_height, size_t output_width, size_t& channel_block_size)
{
    clhpp_feather::CLKernelInfo& kernel_info = this->cl_kernel_info_map[kname];
    std::vector<size_t>& gws = kernel_info.gws;
    std::vector<size_t>& lws = kernel_info.lws;
    size_t padded_input_c = this->_bottom_blobs[this->_bottom[0]]->get_channels_padding();
    size_t padded_output_c = this->_top_blobs[this->_top[0]]->get_channels_padding();
    if (gws.size() != 0 || lws.size() != 0)
    {
        gws.clear();
        lws.clear();
    }

    int h_lws = output_height > 32 ? 16 : 8;
    int w_lws = output_width > 32 ? 16 : 8;

    int c_blk_size = 4;
    if (padded_input_c % 16 == 0 && padded_output_c % 16 == 0)
    {
        c_blk_size = 16;
    }
    else if (padded_input_c % 8 == 0 && padded_output_c % 8 == 0)
    {
        c_blk_size = 8;
    }
    channel_block_size = c_blk_size;
    size_t gws_dim0 = output_height == 1 ? 1 : (output_height / h_lws + !!(output_height % h_lws)) * h_lws;
    size_t gws_dim1 = output_width == 1 ? 1 : (output_width / w_lws  + !!(output_width % w_lws)) * w_lws;
    size_t gws_dim2 = padded_output_c / c_blk_size;
    size_t lws_dim0 = output_height == 1 ? 1 : h_lws;
    size_t lws_dim1 = output_width == 1 ? 1 : w_lws;
    size_t lws_dim2 = (gws_dim2 > 4 && gws_dim2 % 4 == 0) ? 4 : 1;

    gws.push_back(gws_dim0);
    gws.push_back(gws_dim1);
    gws.push_back(gws_dim2);
    lws.push_back(lws_dim0);
    lws.push_back(lws_dim1);
    lws.push_back(lws_dim2);
    return 0;
}

template<class Dtype>
int Layer<Dtype>::ForwardCL()
{
    return false;
}

template<class Dtype>
int Layer<Dtype>::ForwardReshapeCL()
{
    return false;
}

#endif

template class Layer<float>;

#ifdef FEATHER_OPENCL
template class Layer<uint16_t>;
#endif
};
