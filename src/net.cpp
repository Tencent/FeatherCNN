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

#include "feather_generated.h"
#include "layer_factory.h"
//
#ifdef FEATHER_OPENCL
#include "layer_factory_cl.h"
#include "layers_cl/input_layer_cl.h"
#endif
// #include <CL/cl.h>
#include "net.h"
#include "layer.h"
#include "layers/input_layer.h"
//#include "log.h"

#include "mempool.h"

#include <stdio.h>
#include <cstring>

#define CHECK_TYPE(rt) if (!CheckDtype())           \
                    {                               \
                        LOGE("Dtype check failed. Please use uint16_t or float for GPU_CL; float for CPU"); \
                        return rt;                  \
                    }                               \

//#define LAYER_TIMING
//#define LAYER_INIT_TIMING
//#define PRINT_SETUP_LOG


namespace feather
{



bool judge_android7_opencl()
{
    //libOpenCL.so
    //android7.0 sdk api 24
    char sdk[93] = "";
    __system_property_get("ro.build.version.sdk", sdk);
    if (std::atoi(sdk) < 24)
    {
        LOGI("[device] sdk [%d] < 24\n", std::atoi(sdk));
        return true;
    }

    bool flage = false;
    std::string lib_name1 = "libOpenCL.so";
    std::string lib_name2 = "libGLES_mali.so";
    std::vector<std::string> libraries_list = {
        "/vendor/etc/public.libraries.txt",
        "/system/etc/public.libraries.txt",
    };
    for(int i = 0; i < libraries_list.size(); i++)
    {
        std::ifstream out;
        std::string line;
        out.open(libraries_list[i].c_str());
        while(!out.eof()){
            std::getline(out, line);
            if(line.find(lib_name1) != line.npos || line.find(lib_name2) != line.npos)
            {
                LOGI("[public] %s:%s",libraries_list[i].c_str(), line.c_str());
                flage = true;
                break;
            }

        }
        out.close();
    }

    const std::vector<std::string> libpaths = {
        "libOpenCL.so",
    #if defined(__aarch64__)
        // Qualcomm Adreno with Android
        "/system/vendor/lib64/libOpenCL.so",
        "/system/lib64/libOpenCL.so",
        // Mali with Android
        "/system/vendor/lib64/egl/libGLES_mali.so",
        "/system/lib64/egl/libGLES_mali.so",
        // Typical Linux board
        "/usr/lib/aarch64-linux-gnu/libOpenCL.so",
    #else
        // Qualcomm Adreno with Android
        "/system/vendor/lib/libOpenCL.so",
        "/system/lib/libOpenCL.so",
        // Mali with Android
        "/system/vendor/lib/egl/libGLES_mali.so",
        "/system/lib/egl/libGLES_mali.so",
        // Typical Linux board
        "/usr/lib/arm-linux-gnueabihf/libOpenCL.so",
    #endif
    };
    for(int i = 0; i < libpaths.size(); i++) {
        ifstream f(libpaths[i].c_str());
        if(f.good()) {
            flage = true;
            LOGI("[libpaths]:%s", libpaths[i].c_str());
            break;
        }
    }
    return flage;
}


template<class Dtype>
bool Net<Dtype>::CheckDtype()
{
    if (rt_param->device_type() == DeviceType::GPU_CL) {
#ifdef FEATHER_OPENCL
      return std::is_same<Dtype, uint16_t>::value | std::is_same<Dtype, float>::value;
#else
      LOGE("Please compile with FEATHER_OPENCL on to use GPU_CL type");
      return false;
#endif
    }
    return std::is_same<Dtype, float>::value;
}


template<class Dtype>
Net<Dtype>::Net(size_t num_threads, DeviceType device_type)
{


#ifdef  FEATHER_OPENCL
    register_layer_creators_cl();
#endif
    register_layer_creators();
    CommonMemPool<float> *mempool = new CommonMemPool<float>();
    rt_param = new RuntimeParameter<float>(mempool, device_type, num_threads);

    CHECK_TYPE();

}


template<class Dtype>
Net<Dtype>::~Net()
{

    for(int i = 0; i < layers.size(); ++i)
    {
        delete layers[i];
        layers[i] = NULL;
    }
    delete rt_param->common_mempool();
    delete rt_param;
    rt_param = NULL;

}

template<class Dtype>
int Net<Dtype>::ExtractBlob(float* output_ptr, std::string name)
{
    CHECK_TYPE(1);
    if (blob_map.find(std::string(name)) == blob_map.end())
    {
        LOGE("Cannot find blob %s\n", name.c_str());
        return -1;
    }
    const Blob<Dtype> *p_blob = blob_map[name];
    switch(rt_param->device_type())
    {
        case DeviceType::CPU:
        {
            const size_t data_size = p_blob->data_size();
            const Dtype *data = p_blob->data();
            memcpy(output_ptr, data, sizeof(Dtype) * data_size);
            break;
        }
        case DeviceType::GPU_CL:
#ifdef FEATHER_OPENCL
            p_blob->ReadFromDeviceCHW(rt_param->command_queue(), output_ptr);
            break;
#else
            LOGE("Please compile OpenCL to use device type GPU_CL.");
            return -1;
#endif
        case DeviceType::GPU_GL:
            LOGE("Not Implemented yet");
            return -1;
        default:
            LOGE("Unsupported device type");
            return -1;
    }

    return 0;
}

template<class Dtype>
int Net<Dtype>::PrintBlobData(std::string blob_name)
{
    CHECK_TYPE(1);
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

template<class Dtype>
int Net<Dtype>::GetBlobDataSize(size_t *data_size, std::string name)
{
    CHECK_TYPE(1);

    if (blob_map.find(std::string(name)) == blob_map.end())
    {
        LOGE("Cannot find blob %s\n", name.c_str());
        return -1;
    }
    const Blob<Dtype> *p_blob = blob_map[name];
    *data_size = p_blob->data_size();
    return 0;
}

template<class Dtype>
int Net<Dtype>::Forward(float *input)
{
    CHECK_TYPE(1);

    switch (rt_param->device_type()) {
        case DeviceType::CPU:
        {
            InputLayer *input_layer = (InputLayer *)layers[0];
            for (int i = 0; i < input_layer->input_size(); ++i)
            {
                input_layer->CopyInput(input_layer->input_name(i), input);
            }
            break;
        }
        case DeviceType::GPU_CL:
#ifdef FEATHER_OPENCL
        {
            InputLayerCL<Dtype> *input_layer = (InputLayerCL<Dtype> *)layers[0];
            for (int i = 0; i < input_layer->input_size(); ++i)
            {
                //LOGI("%s", input_layer->input_name(i).c_str());
                input_layer->CopyInput(input_layer->input_name(i), input);
            }
            break;
        }
#else
        LOGE("Please compile OpenCL to use device type GPU_CL.");
        return -1;
#endif
        case DeviceType::GPU_GL:
            LOGE("Not implemented yet");
            return -1;
        default:
            LOGE("Unsupported device type");
            return -1;
    }

    int layer_size = layers.size();

    for (int i = 1; i < layer_size; ++i)
    {
        // sleep(2);
#ifdef LAYER_TIMING
        timespec tpstart, tpend;
        LOGD("Entering layer %s type %s\n", layers[i]->name().c_str(), layers[i]->type().c_str());
        clock_gettime(CLOCK_MONOTONIC, &tpstart);
#endif
        //LOGD("Forward layer%d:%s %s\n", i, layers_cl[i]->name().c_str(), layers_cl[i]->type().c_str());
        // layers[i]->Forward();
        switch (rt_param->device_type()) {
            case DeviceType::CPU:
                layers[i]->Forward();
                break;
            case DeviceType::GPU_CL:
#ifdef FEATHER_OPENCL
                layers[i]->ForwardCL();
                break;
#else
                LOGE("Please compile OpenCL to use device type GPU_CL.");
                return -1;
#endif
            case DeviceType::GPU_GL:
                LOGE("Not implemented yet");
                return -1;
            default:
                LOGE("Unsupported device type");
                return -1;
        }


#if 0
        for (size_t j = 0; j < layers[i]->top_blob_size(); j++)
            layers[i]->top_blob(j)->PrintBlobInfo();

	PrintBlobData(layers[i]->name());
#endif

#ifdef LAYER_TIMING
        clock_gettime(CLOCK_MONOTONIC, &tpend);
        double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
        LOGD("Layer %s type %s spent %lfms\n", layers[i]->name().c_str(), layers[i]->type().c_str(), timedif / 1000.0);
#endif
    }
    return 0;
}

template<class Dtype>
int Net<Dtype>::Forward(float* input, int height, int width)
{
    CHECK_TYPE(1);
    switch (rt_param->device_type()) {
        case DeviceType::CPU:
        {
            InputLayer *input_layer = (InputLayer *)layers[0];
            input_layer->Reshape(input_layer->input_name(0), height, width);
            input_layer->CopyInput(input_layer->input_name(0), input);
            break;
        }
        case DeviceType::GPU_CL:
#ifdef FEATHER_OPENCL
        {
            InputLayerCL<Dtype> *input_layer = (InputLayerCL<Dtype> *)layers[0];
            input_layer->Reshape(input_layer->input_name(0), height, width);
            input_layer->CopyInput(input_layer->input_name(0), input);
            break;
        }
#else
        LOGE("Please compile OpenCL to use device type GPU_CL.");
        return -1;
#endif
        case DeviceType::GPU_GL:
            LOGE("Not implemented yet");
            return -1;
        default:
            LOGE("Unsupported device type");
            return -1;
    }

    int layer_size = layers.size();

    for (int i = 1; i < layer_size; ++i)
    {
        // sleep(2);
#ifdef LAYER_TIMING
        timespec tpstart, tpend;
        LOGD("Entering layer %s type %s\n", layers[i]->name().c_str(), layers[i]->type().c_str());
        clock_gettime(CLOCK_MONOTONIC, &tpstart);
#endif
        switch (rt_param->device_type()) {
            case DeviceType::CPU:
                layers[i]->ForwardReshape();
                break;
            case DeviceType::GPU_CL:
#ifdef FEATHER_OPENCL
                layers[i]->ForwardReshapeCL();
                break;
#else
                LOGE("Please compile OpenCL to use device type GPU_CL.");
                return -1;
#endif
            case DeviceType::GPU_GL:
                LOGE("Not implemented yet");
                return -1;
            default:
                LOGE("Unsupported device type");
                return -1;
        }
    }
    return 0;
}

template<class Dtype>
void Net<Dtype>::TraverseNet()
{
    CHECK_TYPE();
    for (int i = 0; i < layers.size(); ++i)
    {
        LOGD("Layer %s %s %s\n", layers[i]->name().c_str(),
               layers[i]->bottom(0).c_str(),
               layers[i]->top(0).c_str());
    }
}

template<class Dtype>
void Net<Dtype>::InitFromPath(const char *model_path)
{
    CHECK_TYPE();
    FILE *fp = NULL;
    fp = fopen(model_path, "rb");
    if (fp == NULL)
    {
        LOGE("Cannot open feather model!\n");
        exit(-1);
    }
    fseek ( fp , 0 , SEEK_SET );
    this->InitFromFile(fp);
    fclose(fp);
}

template<class Dtype>
void Net<Dtype>::InitFromStringPath(std::string model_path)
{
    CHECK_TYPE();
    LOGI("Init model path %s", model_path.c_str());
	InitFromPath(model_path.c_str());
}

template<class Dtype>
void Net<Dtype>::InitFromFile(FILE* fp)
{
    CHECK_TYPE();
    if (fp == NULL)
    {
        LOGE("Cannot open feather model!\n");
        exit(-1);
    }
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t *net_buffer = (uint8_t *) malloc(sizeof(uint8_t) * file_size);
    size_t read_size = fread(net_buffer, sizeof(uint8_t), file_size, fp);
    if (read_size != file_size)
    {
        LOGE("Reading model failed! file_size %ld read size %ld\n", file_size, read_size);
        exit(-1);
    }
    LOGD("Finished loading from file");
    this->InitFromBuffer(net_buffer);

    free(net_buffer);
}

template<class Dtype>
int Net<Dtype>::RemoveLayer(Layer<Dtype>* target_layer)
{
  CHECK_TYPE(1);
	if(target_layer->bottom_size() != 1 || target_layer->top_size() != 1)
	{
		LOGE("Cannot remove target layer %s type %s with mutliple input/outputs!", target_layer->name().c_str(), target_layer->type().c_str());
		return -1;
	}

	std::string new_bottom = target_layer->bottom(0);
	std::string old_bottom = target_layer->top(0);
#ifdef PRINT_SETUP_LOG
                LOGD("Old bottom %s to new bottom %s", old_bottom.c_str(), new_bottom.c_str());
#endif
	const Blob<Dtype> * new_bottom_blob = target_layer->bottom_blob(0);

	for(int i = 0; i < layers.size(); ++i)
	{
		if(layers[i] == target_layer)
		{
			layers.erase(layers.begin() + i);
			--i;
			continue;
		}
		Layer<Dtype> *next_layer = layers[i];
		for(int b = 0; b < next_layer->bottom_size(); ++b)
		{
			if (next_layer->bottom(b).compare(old_bottom) == 0)
			{
				next_layer->ReplaceBottomBlob(old_bottom, new_bottom, new_bottom_blob);
				break;
			}
		}
	}
	delete target_layer;
  target_layer = NULL;
	return 0;
}
template<class Dtype>
bool Net<Dtype>::InitFromBuffer(const void *net_buffer)
{
    CHECK_TYPE(false);
    //rt_param in the param list just to distinguish.
    const NetParameter *net_param = feather::GetNetParameter(net_buffer);
    size_t layer_num = VectorLength(net_param->layer());
    //Find input layer.
#ifdef PRINT_SETUP_LOG
    LOGD("Loading %d layers", layer_num);
#endif
    for (int i = 0; i < layer_num; ++i)
    {
        if (net_param->layer()->Get(i)->type()->str().compare("Input") == 0)
        {
            switch (rt_param->device_type())
            {
                case DeviceType::CPU:
                  layers.push_back(LayerRegistry<Dtype>::CreateLayer(net_param->layer()->Get(i), rt_param));
                  break;
                case DeviceType::GPU_CL:
#ifdef FEATHER_OPENCL
                  layers.push_back(LayerRegistryCL<Dtype>::CreateLayer(net_param->layer()->Get(i), rt_param));
#else
                  LOGE("Please compile OpenCL to use device type GPU_CL.");
#endif
                  break;
                case DeviceType::GPU_GL:
                  LOGE("Not implemented yet.");
                  break;
                default:
                  LOGE("Have not supported yet.");
                  break;
            }
            break;
        }
    }

    for (int i = 1; i < layer_num; ++i)
    {
        const LayerParameter *layer_param = net_param->layer()->Get(i);
        switch (rt_param->device_type())
        {
            case DeviceType::CPU:
                layers.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param, rt_param));
#ifdef PRINT_SETUP_LOG
                LOGD("Setup layer %d %s\n", i, layer_param->name()->c_str());
#endif
                break;
            case DeviceType::GPU_CL:
#ifdef FEATHER_OPENCL
                layers.push_back(LayerRegistryCL<Dtype>::CreateLayer(layer_param, rt_param));
#ifdef PRINT_SETUP_LOG
                LOGD("Setup layer cl %d %s\n", i, layer_param->name()->c_str());
#endif
#else
                LOGE("Please compile OpenCL to use device type GPU_CL.");
#endif
                break;
            case DeviceType::GPU_GL:
                LOGE("Not implemented yet.");
                break;
            default:
                LOGE("Have not supported yet.");
                break;
        }

    }

#ifdef PRINT_SETUP_LOG
    LOGD("Layer setup finish");
#endif

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
#ifdef PRINT_SETUP_LOG
                LOGI("Setting up blob %s", blob_name.c_str());
#endif
                //TODO handle error: when blob_name has not been inserted into map.
                if (blob_map.find(blob_name) != blob_map.end())
                    layers[i]->SetupBottomBlob(blob_map[blob_name], blob_name);
                else
                {
                    LOGE("Blob %s not setup yet, may be casued by wrong layer order. Aborted.\n");
                    return false;
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
            Layer<Dtype> *next_layer = layers[j];
            while (layers[i]->TryFuse(next_layer) == 1)
            {
#if 0
                //Update the respective bottoms in other layers.
                std::string new_bottom = layers[i]->top(0);
                std::string old_bottom = next_layer->top(0);
#ifdef PRINT_SETUP_LOG
                LOGD("Old bottom %s to new bottom %s\n", old_bottom.c_str(), new_bottom.c_str());
#endif
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
                delete layers[j];
                layers.erase(layers.begin() + j);
#else
		this->RemoveLayer(layers[j]);
#endif
#ifdef PRINT_SETUP_LOG
                LOGD("Erased layer %d %s\n", j, next_layer->name().c_str());
#endif
                next_layer = layers[j];
#ifdef PRINT_SETUP_LOG
                LOGD("Layer %d after erasing: %s type %s\n", j, next_layer->name().c_str(), next_layer->type().c_str());
#endif
            }
        }
    }
    //Remove Dropout Layers
    for (int i = 0; i < layers.size() - 1; ++i)
    {
	    if(layers[i]->type().compare("Dropout") == 0)
	    {

#ifdef PRINT_SETUP_LOG
                LOGD("Erase layer %d %s\n", i, layers[i]->name().c_str(), layers[i]->type().c_str());
#endif
		this->RemoveLayer(layers[i]);
#ifdef PRINT_SETUP_LOG
                LOGD("Layer %d after erasing: %s type %s\n", i, layers[i]->name().c_str(), layers[i]->type().c_str());
#endif
		--i;
	    }
    }

    //Rebuild blob map
#ifdef LAYER_INIT_TIMING
    timespec total_tpstart, total_tpend;
    double total_timedif, total_timedif_s1 = 0.0, total_timedif_s2 = 0.0, total_timedif_s3 = 0.0;
    clock_gettime(CLOCK_MONOTONIC, &total_tpstart);
#endif

    blob_map.clear();
    //std::map<std::string, cl::Program> cl_program_map;
    for (int i = 0; i < layers.size(); ++i)
    {
#ifdef LAYER_INIT_TIMING
        timespec tpstart, tpend;
        double timedif;
        clock_gettime(CLOCK_MONOTONIC, &tpstart);
#endif
        for (int t = 0; t < layers[i]->top_size(); ++t)
        {
            std::string blob_name = layers[i]->top(t);
            blob_map[blob_name] = layers[i]->top_blob(blob_name);
#ifdef PRINT_SETUP_LOG
	    LOGI("Blob %s", blob_name.c_str());
            blob_map[blob_name]->PrintBlobInfo();
#endif
        }
        switch (rt_param->device_type())
        {
            case DeviceType::CPU:
              layers[i]->Init();
              break;
            case DeviceType::GPU_CL:
#ifdef FEATHER_OPENCL
              layers[i]->SetBuildOptions();
#ifdef LAYER_INIT_TIMING
              clock_gettime(CLOCK_MONOTONIC, &tpend);
              timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
              LOGD("%s SetBuildOptions spent %lfms\n",layers[i]->name().c_str() , timedif / 1000.0);
              total_timedif_s1 += timedif;
              clock_gettime(CLOCK_MONOTONIC, &tpstart);
#endif
              if (layers[i]->BuildOpenCLProgram(rt_param->cl_runtime()->cl_program_map()))
              {
                  LOGE("Build layer programs failed");
                  return false;
              }
#ifdef LAYER_INIT_TIMING
              clock_gettime(CLOCK_MONOTONIC, &tpend);
              timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
              LOGD("%s BuildOpenCLProgram spent %lfms\n", layers[i]->name().c_str(), timedif / 1000.0);
              total_timedif_s2 += timedif;
              clock_gettime(CLOCK_MONOTONIC, &tpstart);
#endif
              if (layers[i]->SetKernelParameters())
              {
                  LOGE("Set up kernel parameters failed");
                  return false;
              }
#ifdef LAYER_INIT_TIMING
              clock_gettime(CLOCK_MONOTONIC, &tpend);
              timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
              LOGD("%s SetKernelParameters spent %lfms\n",layers[i]->name().c_str(), timedif / 1000.0);
              total_timedif_s3 += timedif;
#endif
#else
              LOGE("Please compile OpenCL to use device type GPU_CL.");
#endif
              break;
            case DeviceType::GPU_GL:
              LOGE("Not implemented yet.");
              break;
            default:
              LOGE("Have not supported yet.");
              break;
        }

    }

#ifdef LAYER_INIT_TIMING
    clock_gettime(CLOCK_MONOTONIC, &total_tpend);
    total_timedif = 1000000.0 * (total_tpend.tv_sec - total_tpstart.tv_sec) + (total_tpend.tv_nsec - total_tpstart.tv_nsec) / 1000.0;
    LOGD("Net Layer Init spent %lfms\n", total_timedif / 1000.0);
    LOGD("Total SetBuildOptions spent %lfms\n", total_timedif_s1 / 1000.0);
    LOGD("Total BuildOpenCLProgram spent %lfms\n", total_timedif_s2 / 1000.0);
    LOGD("Total SetKernelParameters spent %lfms\n", total_timedif_s3 / 1000.0);
#endif

    //Allocate for common mempool.
    rt_param->common_mempool()->Alloc();
    return true;
}

template<class Dtype>
int Net<Dtype>::SetProgMapFromNet(const Net<Dtype>* infer_net) {
#ifdef FEATHER_OPENCL
    if (infer_net->rt_param->device_type() == DeviceType::GPU_CL &&
        this->rt_param->device_type() == DeviceType::GPU_CL) {
          this->rt_param->cl_runtime()->cl_program_map().insert(infer_net->rt_param->cl_runtime()->cl_program_map().begin(),
                                                  infer_net->rt_param->cl_runtime()->cl_program_map().end());
    } else {
      LOGE("SetProgMapFromNet device type mismatch.");
      return -1;
    }
#endif
    return 0;
}



template class Net<float>;
#ifdef FEATHER_OPENCL
template class Net<uint16_t>;
#endif

};
