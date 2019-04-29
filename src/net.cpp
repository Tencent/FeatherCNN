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

#include "layer_factory.h"
#include "net.h"
#include "layer.h"

#include "ncnn/paramdict.h"
#include "ncnn/modelbin.h"

#include <stdio.h>
#include <cstring>

#define LAYER_TIMING
#define LAYER_INIT_TIMING
#define PRINT_SETUP_LOG


namespace feather
{

Net::Net()
{
    register_layer_creators();
    CommonMemPool<float> *mempool = new CommonMemPool<float>();
    rt_param = new RuntimeParameter<float>(mempool, 1);
}

Net::~Net()
{

    for (int i = 0; i < layers.size(); ++i)
    {
        delete layers[i];
        layers[i] = NULL;
    }
    delete rt_param->common_mempool();
    delete rt_param;
    rt_param = NULL;
}

int Net::LoadParam(const char* path)
{
    FILE *fp = NULL;
    fp = fopen(path, "r");
    if (fp == NULL)
    {
        LOGE("Cannot open param file, path: %s\n", path);
        return -1;
    }
    this->LoadParam(fp);
    fclose(fp);
    return 0;
}

int Net::LoadParam(FILE* param_fp)
{
    if(ChkParamHeader(param_fp) == -1)
    {
        return -1;
    }
    // parse
    int layer_count = 0;
    int blob_count = 0;
    int nbr = 0;
    nbr = fscanf(param_fp, "%d %d", &layer_count, &blob_count);
    if (nbr != 2 || layer_count <= 0 || blob_count <= 0)
    {
        fprintf(stderr, "issue with param file\n");
        return -1;
    }
    printf("layer_size %d blob_size %d\n", layer_count, blob_count);
    layers.resize((size_t) layer_count);

    ncnn::ParamDict pd;

    int blob_index = 0;
    for (int i=0; i<layer_count; i++)
    {
        int nscan = 0;

        char layer_type[257];
        char layer_name[257];
        int bottom_count = 0;
        int top_count = 0;
        nscan = fscanf(param_fp, "%256s %256s %d %d", layer_type, layer_name, &bottom_count, &top_count);
        if (nscan != 4)
        {
            continue;
        }

        // fprintf(stderr, "new layer %d %s\n", i, layer_name);
        Layer *layer = LayerRegistry::CreateLayer(layer_type, rt_param);

        if (!layer)
        {
            fprintf(stderr, "layer %s not exists or registered\n", layer_type);
            return -200;
        }

        layer->name = std::string(layer_name);
        layer->type = std::string(layer_type);
        layer->bottoms.resize(bottom_count);

        for (int j=0; j<bottom_count; j++)
        {
            char bottom_name[257];
            nscan = fscanf(param_fp, "%256s", bottom_name);
            if (nscan != 1)
            {
                continue;
            }
	        // printf("bottom name %s\n", bottom_name);
            // layer->bottoms[j] = new Blob<float>(bottom_name);
            std::map<std::string, Blob<float> *>::iterator map_iter = blob_map.find(bottom_name); 
            if (( map_iter == blob_map.end()) && (layer->type.compare("Input") != 0))
            {
                LOGE("Topology error: bottom blob %s of layer %s type %s not found in map.", bottom_name, layer_name, layer_type);
                return -300;
            }
            layer->bottoms[j] = map_iter->second;
            printf("# Bottom name %s\n", layer->bottoms[j]->name.c_str());
        }

        layer->tops.resize(top_count);
        
        for (int j=0; j<top_count; j++)
        {

            char top_name[257];
            nscan = fscanf(param_fp, "%256s", top_name);
            if (nscan != 1)
            {
                continue;
            }
	        // printf("top name %s\n", top_name);
            layer->tops[j] = new Blob<float>(top_name);
            blob_map[top_name] = layer->tops[j];
        }

        // layer specific params
        int pdlr = pd.load_param(param_fp);
        if (pdlr != 0)
        {
            fprintf(stderr, "ParamDict load_param failed\n");
            return pdlr;
        }
        int lr = layer->LoadParam(pd);
        if (lr != 0)
        {
           fprintf(stderr, "Layer %s load_param failed\n", layer_name);
           return lr;
        }
        int gr = layer->GenerateTopBlobs();
        if (gr != 0)
        {
            fprintf(stderr, "Layer %s failed to generate tops\n", layer_name);
            return gr;
        }
        layers[i] = layer;
    }
    return 0;
}

int Net::LoadWeights(const char* path)
{
    FILE *fp = NULL;
    fp = fopen(path, "rb");
    if (fp == NULL)
    {
        LOGE("Cannot open param file, path: %s\n", path);
        return -1;
    }
    this->LoadWeights(fp);
    fclose(fp);
    return 0;   
}

int Net::LoadWeights(FILE* fp)
{
    if (this->layers.empty())
    {
        LOGE("Network has not been loaded. Please load the param file first.\n");
        return -1;
    }
    int ret = 0;
    ncnn::ModelBinFromStdio mb(fp);
    for (size_t i = 0; i < this->layers.size(); i++)
    {
        Layer* layer = layers[i];
        if (!layer){
            fprintf(stderr, "LoadWeights error at layer %d, parameter file has inconsistent content.\n", (int)i);
            ret = -1;
            break;
        }
        int lret = layer->LoadWeights(mb);
        if (lret != 0)
        {
            fprintf(stderr, "Layer %s loading weights failed with exit code %d\n", layer->name.c_str(), lret);
            ret = -1;
            break;
        }
        printf("Layer %zu %s good\n", i, layer->name.c_str());
    }
    return ret;
}

int Net::ExtractBlob(float* output_ptr, std::string name)
{
    if (blob_map.find(std::string(name)) == blob_map.end())
    {
        LOGE("Cannot find blob %s\n", name.c_str());
        return -1;
    }
    const Blob<float> *p_blob = blob_map[name];
       const size_t data_size = p_blob->data_size();
                const float *data = p_blob->data();
                memcpy(output_ptr, data, sizeof(float) * data_size);
    

    return 0;
}

int Net::PrintBlobData(std::string blob_name)
{
    size_t data_size;
    this->GetBlobDataSize(&data_size, blob_name);
    float *arr = (float*) malloc(sizeof(float) * data_size);
    this->ExtractBlob(arr, blob_name);
    size_t len = data_size;

    for (int i = 0; i < len; ++i)
    {
        LOGD("%f\t", arr[i]);
    }
    LOGD("\n");
    free(arr);

    return 0;
}

int Net::GetBlobDataSize(size_t *data_size, std::string name)
{
    if (blob_map.find(std::string(name)) == blob_map.end())
    {
        LOGE("Cannot find blob %s\n", name.c_str());
        return -1;
    }
    const Blob<float> *p_blob = blob_map[name];
    *data_size = p_blob->data_size();
    return 0;
}


int Net::Forward()
{
    // InputLayer *input_layer = (InputLayer *)layers[0];
    // for (int i = 0; i < input_layer->input_size(); ++i)
    // {
    //     input_layer->CopyInput(input_layer->input_name(i), input);
    // }

    int layer_size = layers.size();

    for (int i = 1; i < layer_size; ++i)
    {
        // sleep(2);
#ifdef LAYER_TIMING
        timespec tpstart, tpend;
        clock_gettime(CLOCK_MONOTONIC, &tpstart);
#endif
        //LOGD("Forward layer%d:%s %s\n", i, layers_cl[i]->name().c_str(), layers_cl[i]->type().c_str());
        layers[i]->Forward();

#if 0
        for (size_t j = 0; j < layers[i]->top_blob_size(); j++)
            layers[i]->top_blob(j)->PrintBlobInfo();

        PrintBlobData(layers[i]->name());
#endif

#ifdef LAYER_TIMING
        clock_gettime(CLOCK_MONOTONIC, &tpend);
        double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
        LOGD("Layer %s type %s spent %lfms", layers[i]->name.c_str(), layers[i]->type.c_str(), timedif / 1000.0);
#endif
    }

#ifdef FEATHER_OPENCL
    if (rt_param->device_type() == DeviceType::GPU_CL)
    {
        rt_param->cl_runtime()->tuner().SetTunerPram();
    }
#endif
    return 0;
}

};
