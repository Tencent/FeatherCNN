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

#include "reshape_layer.h"
#include "arm/generic_kernels.h"
#include "math.h"

namespace feather
{
int ReshapeLayer::Forward()
{
    const  Blob<float>* bottom_blob = _bottom_blobs[_bottom[0]];
    const float* input = bottom_blob->data();     

    size_t num = bottom_blob->num();
    size_t channels = bottom_blob->channels();
    size_t height = bottom_blob->height();
    size_t width = bottom_blob->width();

    float *output = _top_blobs[_top[0]]->data();
    float *p = output;

    memcpy(p, input, num*channels*height*width*(sizeof(float)));

    return 0;
}

int ReshapeLayer::GenerateTopBlobs()
{
    assert(_bottom.size() == 1);
    assert(_top.size() == 1);

    const Blob<float>* bottom_blob = _bottom_blobs[_bottom[0]];
    size_t num = bottom_blob->num();
    size_t channels = bottom_blob->channels();
    size_t height = bottom_blob->height();
    size_t width = bottom_blob->width();
    size_t total = num * channels * height * width;
   

    if(dim[0]==0)	dim[0] = num;
    if(dim[1]==0)	dim[1] = channels;
    if(dim[2]==0)	dim[2] = height;
    if(dim[3]==0)	dim[3] = width;

    size_t sum = 1;
    for(int i=0;i<4;i++)	
        if(dim[i]!=-1)  sum *= dim[i];	
    
    for(int i=0;i<4;i++)	
	if(dim[i]==-1)	dim[i] = total/sum;  
  
    _top_blobs[_top[0]] = new Blob<float>(dim[0], dim[1], dim[2], dim[3]);
    _top_blobs[_top[0]]->Alloc();

    return 0;
}

int ReshapeLayer::ForwardReshape()
{
    assert(_bottom.size() == 1);
    assert(_top.size() == 1);

    const Blob<float>* bottom_blob = _bottom_blobs[_bottom[0]];
    size_t num = bottom_blob->num();
    size_t channels = bottom_blob->channels();
    size_t height = bottom_blob->height();
    size_t width = bottom_blob->width();

    size_t total = num * channels * height * width;
   
    size_t sum = 1;
    for(int i=0;i<4;i++)	
        if(dim[i]!=0)  sum *= dim[i];	
    
    for(int i=0;i<4;i++)	
	if(dim[i]==0)	dim[i] = total/sum;  
    
    _top_blobs[_top[0]]->ReshapeWithRealloc(dim[0], dim[1], dim[2], dim[3]);
    return this->Forward();
}

int ReshapeLayer::Init()
{
//    select_weights = _weight_blobs[0]->data();
    const Blob<float>* bottom_blob = _bottom_blobs[_bottom[0]];
    size_t channels = bottom_blob->channels();
    return 0;
}
};
