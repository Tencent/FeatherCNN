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
 * For runtime parameters
 */

#pragma once

#include "mempool.h"

template<typename Dtype>
class RuntimeParameter
{
    public:
        RuntimeParameter() : _common_mempool(NULL), _num_threads(1)
        {
        }
        RuntimeParameter(CommonMemPool<Dtype> *common_mempool, size_t num_threads)
            : _common_mempool(common_mempool), _num_threads(num_threads)
        {
        }
        CommonMemPool<Dtype>* common_mempool() const
        {
            return _common_mempool;
        }
        size_t num_threads() const
        {
            return _num_threads;
        }

    private:
        CommonMemPool<Dtype> *_common_mempool;
        size_t _num_threads;
};
