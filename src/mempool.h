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

#include <map>

#define MEMPOOL_CHECK_RETURN(var) {if(!var){fprintf(stderr, "Err in file %s line %d\n", __FILE__, __LINE__);return false;}}

template<typename PTR_TYPE>
class CommonMemPool
{
    public:
        CommonMemPool(): common_size(0), allocated_size(0), common_memory(NULL) {}
        ~CommonMemPool();

        bool Request(size_t size_byte);
        bool GetPtr(PTR_TYPE ** ptr);
        bool Reset();
        bool Free();
        bool Alloc();

    private:
        //Default common memory pool
        size_t common_size;
        size_t allocated_size;
        PTR_TYPE * common_memory;

        //Map common ID to size
        std::map<size_t, size_t> common_size_map;
        //Map common ID to pointer
        std::map<size_t, PTR_TYPE *> common_ptr_map;
};