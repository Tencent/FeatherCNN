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

/*
 * The layer factory modifies from caffe.
 */
#pragma once

#include "layer.h"

#include <map>
#include <string>
#include <vector>

using namespace std;

namespace feather
{
class Layer;
class LayerRegistry
{
    public:
        typedef Layer* (*Creator)(RuntimeParameter<float> *);
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
        static Layer *CreateLayer(std::string type, RuntimeParameter<float> *rt_param)
        {
            // const string &type = param->type()->str();
            CreatorRegistry &registry = Registry();
            if (registry.find(type) != registry.end())
            {
                return registry[type](rt_param);
            }
            else
            {
                fprintf(stderr, "Layer type %s is not supported in FeatherCNN...Aborting\n", type.c_str());
                return NULL;
            }
        }

    private:
        // Layer registry should never be instantiated - everything is done with its
        // static variables.
        LayerRegistry() {}
};


class LayerRegisterer
{
    public:
        LayerRegisterer(const string &type,
                        Layer * (*creator)(RuntimeParameter<float>* ))
        {
            LayerRegistry::AddCreator(type, creator);
        }
};

void register_layer_creators();

#define DEFINE_LAYER_CREATOR(feather_layer_name) \
    static Layer *GetLayer##feather_layer_name(RuntimeParameter<float> * rt_param) \
    {return (Layer *) feather_layer_name##Layer;}

#define REGISTER_LAYER_CREATOR(ncnn_type_name, feather_layer_name) \
    static LayerRegisterer g_creator_f_##ncnn_type_name(#ncnn_type_name, GetLayer##feather_layer_name);
};
