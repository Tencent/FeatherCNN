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

#include "mempool.h"

#include <stdio.h>
#include <stdlib.h>

template<typename PTR_TYPE>
CommonMemPool<PTR_TYPE>::~CommonMemPool()
{
    if (common_memory || common_size_map.size() || common_ptr_map.size())
    {
        //fprintf(stderr, "Warning: common memroy not freed before pool desctruction. Proceed with free.\n");
        //PrintStats();
        FreeAll();
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
        common_size = 0;
        common_memory = NULL;
    }
    return true;
}

template<typename PTR_TYPE>
bool CommonMemPool<PTR_TYPE>::Free(size_t id)
{
    std::map<size_t, size_t>::iterator
    it = common_size_map.find(id);
    if (it == common_size_map.end())
    {
        fprintf(stderr, "Error: free common memory id %ld failed: ID doesn't exist\n", id);
        return false;
    }
    free(common_ptr_map[it->first]);
    common_ptr_map.erase(it->first);
    common_size_map.erase(it);
    return true;
}

template<typename PTR_TYPE>
bool CommonMemPool<PTR_TYPE>::FreeAll()
{
    Free();
    std::map<size_t, size_t>::iterator
    it = common_size_map.begin();
    for (; it != common_size_map.end(); ++it)
    {
        free(common_ptr_map[it->first]);
    }
    common_ptr_map.clear();
    common_size_map.clear();
    return true;
}

template<typename PTR_TYPE>
bool CommonMemPool<PTR_TYPE>::Request(size_t size_byte)
{
    common_size = (common_size > size_byte) ? common_size : size_byte;
    return true;
}

template<typename PTR_TYPE>
bool CommonMemPool<PTR_TYPE>::Request(size_t size_byte, size_t id)
{
    std::map<size_t, size_t>::iterator it = common_size_map.find(id);
    if (it != common_size_map.end())
    {
        common_size_map[id] = it->second > size_byte ? it->second : size_byte;
    }
    else
    {
        common_size_map[id] = size_byte;
    }
    return true;
}

template<typename PTR_TYPE>
bool CommonMemPool<PTR_TYPE>::GetPtr(PTR_TYPE ** ptr)
{
    if (!common_memory)
    {
        fprintf(stderr, "Common memroy not allocated\n");
        return false;
    }
    *ptr = common_memory;
    return true;
}

template<typename PTR_TYPE>
bool CommonMemPool<PTR_TYPE>::GetPtr(PTR_TYPE ** ptr, size_t id)
{
    if (common_ptr_map.find(id) == common_ptr_map.end())
    {
        fprintf(stderr, "Error: common ptr for ID %ld not found\n", id);
        *ptr = NULL;
        return false;
    }
    *ptr = common_ptr_map[id];
    return true;
}

template<typename PTR_TYPE>
void CommonMemPool<PTR_TYPE>::PrintStats()
{
    printf("Default common pool stat: size %ld, ptr %lx\n", common_size, (size_t)common_memory);
    std::map<size_t, size_t>::iterator it = common_size_map.begin();
    for (; it != common_size_map.end(); ++it)
    {
        printf("Common pool %ld stat: size %ld, ptr %lx\n", it->first, it->second, (size_t)common_ptr_map[it->first]);
    }
}

template<typename PTR_TYPE>
PrivateMemPool<PTR_TYPE>::PrivateMemPool()
{
    //private_map[NULL] = 0;
}

template<typename PTR_TYPE>
PrivateMemPool<PTR_TYPE>::~PrivateMemPool()
{
    if (private_map.size())
    {
        fprintf(stderr, "Warning: private memories are not freed before memory pool deconstruction. Proceed with free.\n");
        PrintStats();
        FreeAll();
    }
}

template<typename PTR_TYPE>
bool PrivateMemPool<PTR_TYPE>::Alloc(PTR_TYPE ** ptr, size_t size_byte)
{
    PTR_TYPE* wptr = NULL;
    wptr = (PTR_TYPE *) _mm_malloc(size_byte, 128);
    if (!wptr)
    {
        fprintf(stderr, "Allocation of size %ld failed\n", size_byte);
        return false;
    }
    private_map[wptr] = size_byte;
    *ptr = wptr;
    return true;
}

template<typename PTR_TYPE>
bool PrivateMemPool<PTR_TYPE>::GetSize(PTR_TYPE * ptr, size_t * size_byte)
{
    typename std::map<PTR_TYPE *, size_t>::iterator it =
        private_map.find(ptr);
    if (it == private_map.end())
    {
        fprintf(stderr, "Error in free private memory: ptr not found in map\n");
        return false;
    }
    *size_byte = it->second;
    return true;
}

template<typename PTR_TYPE>
bool PrivateMemPool<PTR_TYPE>::Free(PTR_TYPE ** ptr)
{
    typename std::map<PTR_TYPE *, size_t>::iterator it =
        private_map.find(*ptr);
    if (it == private_map.end())
    {
        fprintf(stderr, "Error in free private memory: ptr not found in map\n");
        return false;
    }
    free(it->first);
    private_map.erase(it);
    *ptr = NULL;
    return true;
}

template<typename PTR_TYPE>
bool PrivateMemPool<PTR_TYPE>::FreeAll()
{
    typename std::map<PTR_TYPE *, size_t>::iterator it =
        private_map.begin();
    for (; it != private_map.end(); ++it)
    {
        free(it->first);
    }
    private_map.clear();
    return true;
}

template<typename PTR_TYPE>
void PrivateMemPool<PTR_TYPE>::PrintStats()
{
    size_t total_mem_size = 0;
    typename std::map<PTR_TYPE *, size_t>::iterator it =
        private_map.begin();
    for (; it != private_map.end(); ++it)
    {
        printf("Private memory ptr %lx size %ld\n", (size_t) it->first, it->second);
        total_mem_size += it->second;
    }
    printf("Private memories occupy a total of %ld bytes\n", total_mem_size);
}

template class CommonMemPool<float>;
template class CommonMemPool<int>;
template class CommonMemPool<void>;
template class PrivateMemPool<float>;
template class PrivateMemPool<int>;
template class PrivateMemPool<void>;
