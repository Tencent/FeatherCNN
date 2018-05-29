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

#pragma once

#include "blob.h"
#include "mempool.h"
#include "rt_param.h"
#include <common.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include <vector>
#include <map>

namespace feather
{
// template<Dtype>
class Layer
{
    public:
        Layer(const void* layer_param, const RuntimeParameter<float>* rt_param);//Layer param must be LayerParameter type

        int SetupBottomBlob(const Blob<float>* p_blob, std::string name);

        int ReplaceBottomBlob(std::string old_bottom, std::string new_bottom, const Blob<float>* p_blob);

        int TryFuse(Layer *next_layer);

        virtual int Fuse(Layer* next_layer);

        virtual int GenerateTopBlobs();

        //Other initializaiton operations
        virtual int Init();

        virtual int Forward();
        std::string name();
        std::string type();
        std::string bottom(size_t i);
        size_t bottom_size();
        std::string top(size_t i);
        size_t top_size();
        size_t top_blob_size();
        const Blob<float>* top_blob(std::string name);
        const Blob<float>* top_blob(size_t idx);
        //For fusing
        const size_t weight_blob_num() const;
        const Blob<float>* weight_blob(size_t i) const;
        bool fusible() const;
    protected:
        std::string _name;
        std::string _type;

        std::vector<std::string> _bottom;
        std::vector<std::string> _top;

        std::map<std::string, const Blob<float>*> _bottom_blobs; //We don't want to do computation inplace.
        std::map<std::string, Blob<float>*> _top_blobs;

        std::vector<Blob<float>*> _weight_blobs;

        bool _fusible;

        size_t num_threads;

        CommonMemPool<float>    *common_mempool;

        PrivateMemPool<float>   private_mempool;
};
};
