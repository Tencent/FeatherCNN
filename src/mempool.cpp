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

#include "mempool.h"

#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

template<typename PTR_TYPE>
CommonMemPool<PTR_TYPE>::~CommonMemPool()
{
    if (common_memory || common_size_map.size() || common_ptr_map.size())
    {
        Free();
    }
}

template<typename PTR_TYPE>
bool CommonMemPool<PTR_TYPE>::Alloc()
{
    if (common_memory)
    {
        fprintf(stderr, "Error: common memory already allocated.\n");
        return false;
    }
    if (common_size > 0)
    {
        common_memory = (PTR_TYPE *) _mm_malloc(common_size, 128);
        if (!common_memory)
        {
            fprintf(stderr, "Error: cannot allocate common memory.\n");
            return false;
        }
        allocated_size = common_size;
    }
    if (common_size_map.size())
    {
        std::map<size_t, size_t>::iterator it
            = common_size_map.begin();
        while (it != common_size_map.end())
        {
            PTR_TYPE *wptr = NULL;
            wptr = (PTR_TYPE *) _mm_malloc(it->second, 128);
            if (!wptr)
            {
                fprintf(stderr, "Allocation for size %ld id %ld failed\n", it->second, it->first);
            }
            common_ptr_map[it->first] = wptr;
            ++it;
        }
    }
    return (common_ptr_map.size() == common_size_map.size()) ? true : false;
}

template<typename PTR_TYPE>
bool CommonMemPool<PTR_TYPE>::Free()
{
    if (common_memory)
    {
        free(common_memory);
        allocated_size = 0;
        common_memory = NULL;
    }
    return true;
}

template<typename PTR_TYPE>
bool CommonMemPool<PTR_TYPE>::Reset()
{
    common_size = 0;
    return this->Free();
}

template<typename PTR_TYPE>
bool CommonMemPool<PTR_TYPE>::Request(size_t size_byte)
{
    common_size = (common_size > size_byte) ? common_size : size_byte;
    return true;
}

template<typename PTR_TYPE>
bool CommonMemPool<PTR_TYPE>::GetPtr(PTR_TYPE ** ptr)
{
    if (!common_memory)
    {
        fprintf(stderr, "Common memroy not allocated\n");
        // return false;
    }
    if (this->common_size > allocated_size)
    {
        this->Free();
        this->Alloc();
    }
    *ptr = common_memory;
    return true;
}

template class CommonMemPool<float>;
template class CommonMemPool<int>;
template class CommonMemPool<uint16_t>;
template class CommonMemPool<void>;