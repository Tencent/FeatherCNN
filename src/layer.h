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
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include <vector>
#include <map>
#include "common.h"

#ifdef FEATHER_OPENCL
#include "CL/half.h"
#endif

namespace feather
{
template<class Dtype>
class Layer
{
    public:
        Layer(const void* layer_param, RuntimeParameter<float>* rt_param);//Layer param must be LayerParameter type
        ~Layer();
        int SetupBottomBlob(const Blob<Dtype>* p_blob, std::string name);

        int ReplaceBottomBlob(std::string old_bottom, std::string new_bottom, const Blob<Dtype>* p_blob);

        int TryFuse(Layer *next_layer);

#ifdef FEATHER_OPENCL
        int BuildOpenCLProgram();

        virtual int SetKernelParameters();

        int FineTuneGroupSize(const cl_kernel& kernel, const size_t& height, const size_t& width);

        virtual int ForwardCL();
#endif

        virtual int Fuse(Layer* next_layer);

        virtual int GenerateTopBlobs();

        //Other initializaiton operations
        virtual int Init();

        virtual int Forward();

        virtual int ForwardReshape();

        std::string name();
        std::string type();
        std::string bottom(size_t i);
        size_t bottom_size();
        std::string top(size_t i);
        size_t top_size();
        size_t top_blob_size();
        const Blob<Dtype>* top_blob(std::string name);
        const Blob<Dtype>* top_blob(size_t idx);
        const Blob<Dtype>* bottom_blob(size_t idx);
        //For fusing
        const size_t weight_blob_num() const;
        const Blob<Dtype>* weight_blob(size_t i) const;
        bool fusible() const;
    protected:
        std::string _name;
        std::string _type;

        std::vector<std::string> _bottom;
        std::vector<std::string> _top;

        std::map<std::string, const Blob<Dtype>* > _bottom_blobs; //We don't want to do computation inplace.
        std::map<std::string, Blob<Dtype>* > _top_blobs;

        std::vector<Blob<Dtype>* > _weight_blobs;

        bool _fusible;
        bool _inplace;

        size_t num_threads;

        CommonMemPool<float>    *common_mempool;

        PrivateMemPool<float>   private_mempool;

        RuntimeParameter<float> *rt_param;

#ifdef FEATHER_OPENCL
        std::vector<std::string> cl_kernel_names;
        std::vector<std::string> cl_kernel_symbols;
        std::vector<std::string> cl_kernel_functions;
        std::vector<cl_program> cl_programs;
        std::vector<cl_kernel> kernels;
        std::vector<cl_event> events;
        std::vector<std::string> build_options;
        size_t global_work_size[3];
        size_t local_work_size[3];
        int group_size_h = 8;
        int group_size_w = 8;
#endif
};

};
