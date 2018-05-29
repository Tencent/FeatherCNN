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
        typedef Layer *(*Creator)(const LayerParameter *, const RuntimeParameter<float> *);
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
        static Layer *CreateLayer(const LayerParameter *param, const RuntimeParameter<float> *rt_param)
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

        static vector<string> LayerTypeList()
        {
            CreatorRegistry &registry = Registry();
            vector<string> layer_types;
            for (typename CreatorRegistry::iterator iter = registry.begin();
                    iter != registry.end(); ++iter)
            {
                layer_types.push_back(iter->first);
            }
            return layer_types;
        }

    private:
        // Layer registry should never be instantiated - everything is done with its
        // static variables.
        LayerRegistry() {}

        static string LayerTypeListString()
        {
            vector<string> layer_types = LayerTypeList();
            string layer_types_str;
            for (vector<string>::iterator iter = layer_types.begin();
                    iter != layer_types.end(); ++iter)
            {
                if (iter != layer_types.begin())
                {
                    layer_types_str += ", ";
                }
                layer_types_str += *iter;
            }
            return layer_types_str;
        }
};

class LayerRegisterer
{
    public:
        LayerRegisterer(const string &type,
                        Layer * (*creator)(const LayerParameter *, const RuntimeParameter<float>*))
        {
            LayerRegistry::AddCreator(type, creator);
        }
};

void register_layer_creators();

#define REGISTER_LAYER_CREATOR(type, creator) \
    static LayerRegisterer g_creator_f_##type(#type, creator);

#define REGISTER_LAYER_CLASS(type)                            \
    Layer *Creator_##type##Layer(const LayerParameter &param) \
    {                                                         \
        return Layer * (new##type##Layer(param));             \
    }                                                         \
    REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)
};
