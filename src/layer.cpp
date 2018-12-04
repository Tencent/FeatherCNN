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

#include <unistd.h>

#include "layer.h"
#include "feather_generated.h"//For LayerParameter


namespace feather
{
template<class Dtype>
Layer<Dtype>::Layer(const void* layer_param_in, RuntimeParameter<float>* rt_param)
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
    if(!_inplace){
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
template<class Dtype>
int Layer<Dtype>::BuildOpenCLProgram()
{
    std::map<std::string, cl::Program> cl_program_map;
    if (cl_kernel_names.size() != cl_kernel_symbols.size()){
      LOGE("program str and names size not match.");
      return -1;
    }
    int size = cl_kernel_names.size();
    for (int i = 0; i < size; i++){
        //LOGI("current program. %s", cl_kernel_names[i].c_str());
        if (cl_program_map.find(cl_kernel_names[i]) != cl_program_map.end()){
            cl_programs.push_back(cl_program_map[cl_kernel_names[i]]);
            //LOGI("exists current program. %s", cl_kernel_names[i].c_str());
            continue;
        }

        cl::Program cur_program;
        std::string opt_str = "";
        for (auto &opt : this->build_options) {
          opt_str += " " + opt;
        }
        if (buildProgramFromSource(this->rt_param->context(), this->rt_param->device(), cur_program, this->cl_kernel_symbols[i], opt_str)){
            LOGE("Build program from source failed.");
            return -1;
        }

        cl_programs.push_back(cur_program);
        cl_program_map[cl_kernel_names[i]] = cur_program;
    }
    // std::map<std::string, cl_program>::iterator program_iter = cl_program_map.begin();
    // for(; program_iter != cl_program_map.end(); program_iter++){
    //     if(program_iter->second != 0){
    //         clReleaseProgram(program_iter->second);
    //     }
    // }

    return 0;
}

template<class Dtype>
int Layer<Dtype>::SetKernelParameters()
{
    //Base layer doesn't know settings.
    return -1;
}

template<class Dtype>
int Layer<Dtype>::SetWorkSize()
{
    //Base layer doesn't know settings.
    return -1;
}

template<class Dtype>
int Layer<Dtype>::FineTuneGroupSize(const cl::Kernel& kernel, const size_t& height, const size_t& width)
{
    //global_work_size HWC
    //local_work_size  HWC
    uint64_t current_work_group_size = 0;
    cl_int error_num = rt_param->device().getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &current_work_group_size);
    if (!checkSuccess(error_num))
    {
        LOGE("Get kernel work group info failed. %s: %s", __FILE__, __LINE__);
        return -1;
    }
   
    while(local_work_size[0] * local_work_size[1] * local_work_size[2] > current_work_group_size){
        if(local_work_size[0] > 1){
            local_work_size[0] /= 2;
        } else if(local_work_size[1] > 1){
            local_work_size[1] /= 2;
        } else if(local_work_size[2] > 1){
            local_work_size[2] /= 2;
        }
    }
    this->global_work_size[0] = (height / local_work_size[0] + !!(height % local_work_size[0])) * local_work_size[0];
    this->global_work_size[1] = (width / local_work_size[1]  + !!(width % local_work_size[1])) * local_work_size[1];
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
