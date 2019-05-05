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

#include "ncnn/paramdict.h"
#include "ncnn/modelbin.h"

#include <vector>

namespace feather
{
class Layer
{
    public:
        Layer(RuntimeParameter<float>* rt_param);
        ~Layer();

        /* LoadParam LoadWeights
         * 
         * Load layer specifc paramters and corresponding weight data into memory.
         * The two functions rely on `ncnn` model files.
         */
        virtual int LoadParam(const ncnn::ParamDict& pd); 
        virtual int LoadWeights(const ncnn::ModelBin& mb); 
        // int CopyDataFromMat(Blob<float>* dst, const ncnn::Mat &src);
        
        /* GenerateTopBlobs
         *
         * Infer top blob shape and allocate memory.
         */
        virtual int Reshape(); 

        /* Init
         *
         * Preprocess the weights in order to reduce inference overhead.
         * Common memory pool first memory allocation occurs in this place
         * when specify an initial input.
         */
        virtual int Init();
        
        virtual int Forward();
        virtual int ForwardReshape();

        /* Fusion functions
         * 
         * Layer fusion is an important technique to imporove memory accessing efficiency.
         * We currently support three patterns: Convolutoin-Bias-ReLU, BN-Scale-Relu, InnerProduct-Bias-ReLU
         */
        virtual int Fuse(Layer* next_layer); //Fuse layers when possible.
        int TryFuse(Layer *next_layer);
        bool fusible() const;

        /* Utility functions for blob retrieval by name*/
        int FindBottomIDByName(std::string name);
        int FindTopIDByName(std::string name);

    public: // We just make everything public. Take care when you write a derived layer.
        std::string name;
        std::string type;

        std::vector<Blob<float>* > bottoms;
        std::vector<Blob<float>* > tops;

        std::vector<Blob<float>* > weights;

        bool _fusible;
        bool _inplace;

        CommonMemPool<float>    *common_mempool;
        RuntimeParameter<float> *rt_param;
};
};
