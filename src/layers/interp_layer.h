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

#include "../feather_generated.h"
#include "../layer.h"
#include "arm/helper.h"

namespace feather
{
class InterpLayer : public Layer
{
    public:
        InterpLayer(const LayerParameter* layer_param, const RuntimeParameter<float>* rt_param)
            : Layer(layer_param, rt_param)
        {
	    	height_out_ = layer_param->interp_param()->height();
	    	width_out_ = layer_param->interp_param()->width();
			pad_beg_ = layer_param->interp_param()->pad_beg();
			pad_end_ = layer_param->interp_param()->pad_end();

			LOGI("Interp layer only supports height & width formula.");
        }
        int Forward();
		int GenerateTopBlobs();
    private:
	int num_;
	int channels_;
	int height_in_;
	int width_in_;
	int height_in_eff_;
	int width_in_eff_;
	int height_out_;
	int width_out_;
	int pad_beg_;
	int pad_end_;

	LayerParameter *layer_param_;
	
};
};
