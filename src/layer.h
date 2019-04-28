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

#pragma once

#include "blob.h"
#include "mempool.h"
#include "rt_param.h"
#include "utils.h"
#include <vector>

namespace feather
{
class Layer
{
    public:
        Layer(RuntimeParameter<float>* rt_param);
        ~Layer();

        int SetupBottomBlob(const Blob<float>* p_blob, std::string name);
        int ReplaceBottomBlob(std::string old_bottom, std::string new_bottom, const Blob<float>* p_blob);
        int TryFuse(Layer *next_layer);

        const Blob<float> *FindBottomByName(std::string name);
        Blob<float> *FindTopByName(std::string name);

        virtual int LoadParams();
        virtual int LoadWeights();
        virtual int Fuse(Layer* next_layer);
        virtual int GenerateTopBlobs();
        virtual int Init();
        virtual int Forward();
        virtual int ForwardReshape();

        //For fusing
        bool fusible() const;

        std::string name;
        std::string type;

        std::vector<const Blob<float>* > bottoms; // Don't write bottom blobs.
        std::vector<Blob<float>* > tops;

        std::vector<Blob<float>* > weights;

        bool _fusible;
        bool _inplace;

        CommonMemPool<float>    *common_mempool;
        PrivateMemPool<float>   private_mempool;
        RuntimeParameter<float> *rt_param;
};

};
