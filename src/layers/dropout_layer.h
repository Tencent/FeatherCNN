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

#include "../feather_simple_generated.h"
#include "../layer.h"

namespace feather
{
class DropoutLayer : public Layer
{
  public:
	DropoutLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
		: Layer(layer_param, rt_param)
	{
		const DropoutParameter *dropout_param = layer_param->dropout_param();
		scale = 1.0 - dropout_param->dropout_ratio();
	}

	int Forward()
	{
		const Blob<float> *p_bottom = _bottom_blobs[_bottom[0]];
		const float* input = p_bottom->data();
		float* output = _top_blobs[_top[0]]->data();

		int n = p_bottom->num();
		int c = p_bottom->channels();
	    int w = p_bottom->width();
	    int h = p_bottom->height();
		int size = w * h;
	    //printf("[DROPOUT] bottom:%s top:%s c:%d h:%d w:%d [%f %f %f %f]\n", _bottom[0].c_str(), _top[0].c_str(), c,h,w, input[0], input[1], input[2], input[3]);

		if (scale == 1.f)
		    return 0;

		#pragma omp parallel for
		for (int q=0; q<c; q++)
		{
			const float* inPtr = input + q*size;
			float* outPtr = output + q*size;

		    for (int i=0; i<size; i++)
		    {
		        outPtr[i] = inPtr[i] * scale;
		    }
		}
#if 0
		printf("[DROPOUT] %f %f %f %f\n", 
			   _top_blobs[_top[0]]->data()[0], 
			   _top_blobs[_top[0]]->data()[1], 
			   _top_blobs[_top[0]]->data()[2], 
			   _top_blobs[_top[0]]->data()[3]);
#endif
		return 0;
	}
	protected:
		float scale;
};
};
