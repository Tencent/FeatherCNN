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

#include "depthwise.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

void dwConv(float* output, float* input, int inw, int inh, int stridew, int strideh, float* kernel, int kw, int kh, int group, int nThreads){
	int outw = (inw - kw) / stridew + 1;//for strided case in odd dimensions, should take the floor value as output dim.
	int outh = (inh - kh) / strideh + 1;

	for(int g = 0; g < group; ++g){
		float* kp = kernel + kw * kh* g;
		float* outg = output + g * outw * outh;
		float* ing = input + g * inw * inh;
		for(int i = 0; i < outh; ++i){
			for(int j = 0; j < outw; ++j){
				float* inp = ing + inw * (i*stridew) + (j*strideh);
				float convSum = 0.f;
				for(int m = 0; m < kh; m++){
					for(int n = 0; n < kw; n++){
						convSum += inp[m * inw + n]* kp[m * kw + n];
					}
				}	
				outg[j] = convSum;
			}
			outg += outw;
		}	
	}
}