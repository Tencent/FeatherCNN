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

#include "feather_simple_generated.h"
#include "layer_factory.h"
#include "net.h"
#include "layer.h"
#include "layers/input_layer.h"
#include "mempool.h"

#include <stdio.h>
#include <cstring>
//#define LAYER_TIMING

namespace feather
{
Net::Net(size_t num_threads)
{
    register_layer_creators();
    CommonMemPool<float> *mempool = new CommonMemPool<float>();
    rt_param = new RuntimeParameter<float>(mempool, num_threads);
}


Net::~Net()
{
    delete rt_param->common_mempool();
    delete rt_param;
}

int Net::ExtractBlob(float* output_ptr, std::string name)
{
    if (blob_map.find(std::string(name)) == blob_map.end())
    {
        fprintf(stderr, "Cannot find blob %s\n", name.c_str());
        return -1;
    }
    const Blob<float> *p_blob = blob_map[name];
    const size_t data_size = p_blob->data_size();
    const float *data = p_blob->data();

    memcpy(output_ptr, data, sizeof(float) * data_size);
    return 0;
}

int Net::GetBlobDataSize(size_t *data_size, std::string name)
{
    if (blob_map.find(std::string(name)) == blob_map.end())
    {
        fprintf(stderr, "Cannot find blob %s\n", name.c_str());
        return -1;
    }
    const Blob<float> *p_blob = blob_map[name];
    *data_size = p_blob->data_size();
    return 0;
}

int Net::Forward(float *input)
{
    InputLayer *input_layer = (InputLayer *)layers[0];
    for (int i = 0; i < input_layer->input_size(); ++i)
    {
        input_layer->CopyInput(input_layer->input_name(i), input);
    }
    for (int i = 1; i < layers.size(); ++i)
    {
#ifdef LAYER_TIMING
        timespec tpstart, tpend;
        clock_gettime(CLOCK_MONOTONIC, &tpstart);
#endif
        //printf("Forward layer%d:%s %s\n", i, layers[i]->name().c_str(), layers[i]->type().c_str());
        layers[i]->Forward();
#if 0
        for (size_t j = 0; j < layers[i]->top_blob_size(); j++)
            layers[i]->top_blob(j)->PrintBlobInfo();
#endif
#ifdef LAYER_TIMING
        clock_gettime(CLOCK_MONOTONIC, &tpend);
        double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
        printf("layer %s type %s spent %lfms\n", layers[i]->name().c_str(), layers[i]->type().c_str(), timedif / 1000.0);
#endif
    }
    return 0;
}

void Net::TraverseNet()
{
    for (int i = 0; i < layers.size(); ++i)
    {
        printf("Layer %s %s %s\n", layers[i]->name().c_str(),
               layers[i]->bottom(0).c_str(),
               layers[i]->top(0).c_str());
    }
}

void Net::InitFromPath(const char *model_path)
{
    FILE *fp = NULL;
    fp = fopen(model_path, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "Cannot open feather model!\n");
        exit(-1);
    }
    this->InitFromFile(fp);
    fclose(fp);
}
void Net::InitFromFile(FILE* fp)
{
    if (fp == NULL)
    {
        fprintf(stderr, "Cannot open feather model!\n");
        exit(-1);
    }
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t *net_buffer = (uint8_t *) malloc(sizeof(uint8_t) * file_size);
    size_t read_size = fread(net_buffer, sizeof(uint8_t), file_size, fp);
    if (read_size != file_size)
    {
        fprintf(stderr, "Reading model failed! file_size %ld read size %ld\n", file_size, read_size);
        exit(-1);
    }
    printf("Finished loading from file\n");
    this->InitFromBuffer(net_buffer);
    free(net_buffer);
}
bool Net::InitFromBuffer(const void *net_buffer)
{
    //rt_param in the param list just to distinguish.
    const NetParameter *net_param = feather::GetNetParameter(net_buffer);
    size_t layer_num = VectorLength(net_param->layer());
    //Find input layer.
    //printf("Loading %d layers\n", layer_num);
    for (int i = 0; i < layer_num; ++i)
    {
        if (net_param->layer()->Get(i)->type()->str().compare("Input") == 0)
        {
            layers.push_back(LayerRegistry::CreateLayer(net_param->layer()->Get(i), rt_param));
            break;
        }
    }
    for (int i = 1; i < layer_num; ++i)
    {
        const LayerParameter *layer_param = net_param->layer()->Get(i);
        Layer *new_layer = LayerRegistry::CreateLayer(layer_param, rt_param);
        //printf("setup layer %s\n", layer_param->name()->c_str());
        layers.push_back(new_layer);
    }
    //Generate top blobs, will check the dependency.
    for (int i = 0; i < layers.size(); ++i)
    {
        size_t top_num = layers[i]->top_size();
        size_t top_blob_num = layers[i]->top_blob_size();
        if (top_blob_num == 0)
        {
            for (int b = 0; b < layers[i]->bottom_size(); ++b)
            {
                std::string blob_name = layers[i]->bottom(b);
                // printf("blob name %s\n", blob_name.c_str());
                //TODO handle error: when blob_name has not been inserted into map.
                if (blob_map.find(blob_name) != blob_map.end())
                    layers[i]->SetupBottomBlob(blob_map[blob_name], blob_name);
                else
                {
                    fprintf(stderr, "Blob %s not setup yet, may be casued by wrong layer order. Aborted.\n");
                    exit(-1);
                }
            }
            layers[i]->GenerateTopBlobs();
        }
        for (int t = 0; t < top_num; ++t)
        {
            std::string blob_name = layers[i]->top(t);
            blob_map[blob_name] = layers[i]->top_blob(blob_name);
            //blob_map[blob_name]->PrintBlobInfo();
        }
    }

    //Try to fuse some layers together
    for (int i = 1; i < layers.size() - 1; ++i)
    {
        if (!layers[i]->fusible())
            continue;
        for (int j = i + 1; j < layers.size(); ++j)
        {
            Layer *next_layer = layers[j];
            while (layers[i]->TryFuse(next_layer) == 1)
            {
                //Update the respective bottoms in other layers.
                std::string new_bottom = layers[i]->top(0);
                std::string old_bottom = next_layer->top(0);
                //printf("old bottom %s to new bottom %s\n", old_bottom.c_str(), new_bottom.c_str());
                for (int k = i + 1; k < layers.size(); ++k)
                {
                    if (k == j)
                        continue;

                    for (int b = 0; b < layers[k]->bottom_size(); ++b)
                    {
                        if (layers[k]->bottom(b).compare(old_bottom) == 0)
                            layers[k]->ReplaceBottomBlob(old_bottom, new_bottom, layers[i]->top_blob(0));
                    }
                }
                //printf("Erasing layer %d %s\n", j, next_layer->name().c_str());
                layers.erase(layers.begin() + j);
                next_layer = layers[j];
                //printf("Layer %d after erasing: %s type %s\n", j, next_layer->name().c_str(), next_layer->type().c_str());
            }
        }
    }

    //Rebuild blob map
    blob_map.clear();
    for (int i = 1; i < layers.size(); ++i)
    {
        for (int t = 0; t < layers[i]->top_size(); ++t)
        {
            std::string blob_name = layers[i]->top(t);
            blob_map[blob_name] = layers[i]->top_blob(blob_name);
            //blob_map[blob_name]->PrintBlobInfo();
        }
        layers[i]->Init();
    }

    //Allocate for common mempool.
    rt_param->common_mempool()->Alloc();
    return true;
}
};
