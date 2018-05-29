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

#ifndef TCNN_MEMORY_POOL_H_
#define TCNN_MEMORY_POOL_H_

#include <map>
#include <common.h>

#define MEMPOOL_CHECK_RETURN(var) {if(!var){fprintf(stderr, "Err in file %s line %d\n", __FILE__, __LINE__);return false;}}

template<typename PTR_TYPE>
class CommonMemPool
{
    public:
        CommonMemPool(): common_size(0), common_memory(NULL) {}
        ~CommonMemPool();

        //Single common memory pool
        bool Request(size_t size_byte);
        bool GetPtr(PTR_TYPE ** ptr);
        bool Free();

        //Multiple common pools by ID
        bool Request(size_t size_byte, size_t id);
        bool GetPtr(PTR_TYPE ** ptr, size_t id);
        bool Free(size_t id);

        bool Alloc();
        bool FreeAll();
        void PrintStats();

    private:
        //Default common memory pool
        size_t common_size;
        PTR_TYPE * common_memory;

        //Map common ID to size
        std::map<size_t, size_t> common_size_map;
        //Map common ID to pointer
        std::map<size_t, PTR_TYPE *> common_ptr_map;
};

template<typename PTR_TYPE>
class PrivateMemPool
{
    public:
        PrivateMemPool();
        ~PrivateMemPool();

        //For private and instant memory allocation
        bool Alloc(PTR_TYPE ** ptr, size_t size_byte);
        bool GetSize(PTR_TYPE * ptr, size_t *size_byte);
        bool Free(PTR_TYPE ** ptr);
        bool FreeAll();
        void PrintStats();
    private:
        //Map private pointer to size
        std::map<PTR_TYPE *, size_t> private_map;
};
#endif
