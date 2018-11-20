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

/*
 * The layer factory modifies from caffe.
 */
#ifdef FEATHER_OPENCL

#pragma once

#include "layer.h"

#include <map>
#include <string>
#include <vector>

using namespace std;

namespace feather
{

class LayerRegistryCL
{
    public:
        typedef Layer<uint16_t>* (*Creator)(const LayerParameter *, RuntimeParameter<float> *);
        typedef std::map<string, Creator> CreatorRegistry;

        static CreatorRegistry &Registry()
        {
            static CreatorRegistry *g_registry_ = new CreatorRegistry();
            return *g_registry_;
        }

        // Adds a creator.
        static void AddCreator(const string &type, Creator creator)
        {
            CreatorRegistry &registry = Registry();
            registry[type] = creator;
        }

        // Get a layer using a LayerParameter.
        static Layer<uint16_t> *CreateLayer(const LayerParameter *param, RuntimeParameter<float> *rt_param)
        {
            const string &type = param->type()->str();
            CreatorRegistry &registry = Registry();
            if (registry.find(type) != registry.end())
            {
                return registry[type](param, rt_param);
            }
            else
            {
                fprintf(stderr, "Layer type %s not registered\n", type.c_str());
                return NULL;
            }
        }

    private:
        // Layer registry should never be instantiated - everything is done with its
        // static variables.
        LayerRegistryCL() {}
};

class LayerRegistererCL
{
    public:
        LayerRegistererCL(const string &type,
                        Layer<uint16_t> * (*creator)(const LayerParameter *, RuntimeParameter<float>*))
        {
            LayerRegistryCL::AddCreator(type, creator);
        }
};

void register_layer_creators_cl();

#define REGISTER_LAYER_CREATOR_CL(type, creator) \
    static LayerRegistererCL g_creator_f_##type(#type, creator);
};

#endif
