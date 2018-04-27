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

#include "common.h"
#include <cstring>
#include <vector>
#include <cstdlib>

int min(int a, int b)
{
    return (a < b) ? a : b;
}

void* _mm_malloc(size_t sz, size_t align)
{
    void *ptr;
    int alloc_result = posix_memalign(&ptr, align, sz);
    if (alloc_result != 0)
    {
        return NULL;
    }
    
    return ptr;
}

void _mm_free(void* ptr)
{
    if (NULL != ptr)
    {
        free(ptr);
        ptr = NULL;
    }
}

void StringTool::SplitString(const std::string &input, const std::string &delim, std::vector<std::string> &parts)
{
    for(char *s=strtok((char *)input.data(), (char *)delim.data()); s; s=strtok(NULL, (char *)delim.data()))
    {
        if (s != NULL)
        {
            parts.push_back(s);
        }
    }
}
