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


#include <booster/depthwise.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

//#include <arm_neon.h>

#ifdef __APPLE__
#else
#include <omp.h>
#endif


template <bool fuseBias, bool fuseRelu>
void globalDwConv(float *output, const float *input, int input_channels, int inw, int inh, float *kernel, int group, int nThreads, float *bias_arr)
{
    assert(group > 0 || input_channels % group == 0);
    int step = inw * inh;
    int block = input_channels / group;
    int groupKernelSize = inw * inh * group;

    for (int i = 0; i < input_channels; i++)
    {
        int k = i / group, u = i % group;
        output[i] = 0;
        for (int j = 0; j < step; j++)
        {
            output[i] += input[i * step + j] * kernel[k * groupKernelSize + u * step + j];
        }
        if (fuseBias)
        {
            output[i] += bias_arr[i];
        }
        if (fuseRelu)
        {
            output[i] = (output[i] > 0.f) ? output[i] : 0.f;
        }
    }

    /*
	int kw = inw, kh = inh;
	int width = kw * kh;
	int widthAligned = width & 0xFFFFFFFC;
	int widthRem = width & 0x03; // int widthRem = width & 0x11;
	int height = group;
	int heightAligned = group & 0xFFFFFFFC;
	int heightRem = height & 0x03; // int heightRem = height & 0x11;
	float ext[8];
	for(int i = 0; i < heightAligned; i += 4)
	{
		float32x4_t sum = vdupq_n_f32(0.f);
		float* p0 = const_cast<float *>(input) + width * i;
		float* p1 = p0 + width;
		float* p2 = p1 + width;
		float* p3 = p2 + width;
		float* k0 = kernel + width * i;
		float* k1 = k0 + width;
		float* k2 = k1 + width;
		float* k3 = k2 + width;

		for(int j = 0; j < widthAligned; j += 4)
		{
			float32x4_t v0 = vld1q_f32(p0);
			p0 += 4;
			float32x4_t v1 = vld1q_f32(p1);
			p1 += 4;
			float32x4_t v2 = vld1q_f32(p2);
			p2 += 4;
			float32x4_t v3 = vld1q_f32(p3);
			p3 += 4;
			
			float32x4_t r0 = vld1q_f32(k0);
			k0 += 4;
			float32x4_t r1 = vld1q_f32(k1);
			k1 += 4;
			float32x4_t r2 = vld1q_f32(k2);
			k2 += 4;
			float32x4_t r3 = vld1q_f32(k3);
			k3 += 4;
			
			float32x4x2_t row01 = vtrnq_f32(v0, v1);
			float32x4x2_t row23 = vtrnq_f32(v2, v3);

//			 * row0 = ( x00 x10 x20 x30 )
//			 * row1 = ( x01 x11 x21 x31 )
//			 * row2 = ( x02 x12 x22 x32 )
//			 * row3 = ( x03 x13 x23 x33 )

			v0 = vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0]));
			v1 = vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1]));
			v2 = vcombine_f32(vget_high_f32(row01.val[0]), vget_high_f32(row23.val[0]));
			v3 = vcombine_f32(vget_high_f32(row01.val[1]), vget_high_f32(row23.val[1]));
			row01 = vtrnq_f32(r0, r1);
			row23 = vtrnq_f32(r2, r3);
			r0 = vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0]));
			r1 = vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1]));
			r2 = vcombine_f32(vget_high_f32(row01.val[0]), vget_high_f32(row23.val[0]));
			r3 = vcombine_f32(vget_high_f32(row01.val[1]), vget_high_f32(row23.val[1]));
#ifdef __aarch64__
			sum = vfmaq_f32(sum, v0, r0);	
			sum = vfmaq_f32(sum, v1, r1);	
			sum = vfmaq_f32(sum, v2, r2);	
			sum = vfmaq_f32(sum, v3, r3);	
#else
			sum = vmlaq_f32(sum, v0, r0);	
			sum = vmlaq_f32(sum, v1, r1);	
			sum = vmlaq_f32(sum, v2, r2);	
			sum = vmlaq_f32(sum, v3, r3);	
#endif
		}
		if(widthRem){
			for(int j = 0; j < widthRem; ++j)
			{
				ext[0] = p0[j];
				ext[1] = p1[j];
				ext[2] = p2[j];
				ext[3] = p3[j];
				ext[4] = k0[j];
				ext[5] = k1[j];
				ext[6] = k2[j];
				ext[7] = k3[j];
#ifdef __aarch64__
				sum = vfmaq_f32(sum, vld1q_f32(ext + 4), vld1q_f32(ext));
#else
				sum = vmlaq_f32(sum, vld1q_f32(ext + 4), vld1q_f32(ext));
#endif
			}
		} 
		vst1q_f32(output + i, sum);
	}
	for(int i = heightAligned; i < height; ++i)
	{
		float* p = const_cast<float *>(input) + i * width;
		float* k = kernel + i * width;
		float sum = 0.f;
		for(int j = 0; j < width; ++j)
		{
			sum += p[j] * k[j];
		}
		output[i] = sum; // output[heightAligned + i] = sum;
	}
*/
}

template <bool fuseBias, bool fuseRelu>
void dwConv_template(float *output, float *input, int input_channels, int inw, int inh, int stridew, int strideh, float *kernel, int kw, int kh, int group, int nThreads, float *bias_arr)
{
    if ((kw == inw) && (kh == inh)) 
    {
        globalDwConv<fuseBias, fuseRelu>(output, input, input_channels, inw, inh, kernel, group, nThreads, bias_arr);
    }
    else
    {
        int outw = (inw - kw) / stridew + 1; //for strided case in odd dimensions, should take the floor value as output dim.
        int outh = (inh - kh) / strideh + 1;

// #pragma omp parallel for num_threads(nThreads) schedule(static)
        printf("dw param %d kernel %d %d stride %d %d input %d %d %d output %d %d\n", group, kh, kw, strideh, stridew, input_channels, inh, inw, outh, outw);
        for (int g = 0; g < group; ++g)
        {
            float *kp = kernel + kw * kh * g;
            float *outg = output + g * outw * outh;
            float *ing = input + g * inw * inh;
            for (int i = 0; i < outh; ++i)
            {
                for (int j = 0; j < outw; ++j)
                {
                    float *inp = ing + inw * (i * stridew) + (j * strideh);
                    float convSum = 0.f;
                    for (int m = 0; m < kh; m++)
                    {
                        for (int n = 0; n < kw; n++)
                        {
                            convSum += inp[m * inw + n] * kp[m * kw + n];
                        }
                    }
                    if (fuseBias)
                    {
                        convSum += bias_arr[g];
                    }
                    if (fuseRelu)
                    {
                        convSum = (convSum > 0.f) ? convSum : 0.f;
                    }
                    outg[j] = convSum;
                }
                outg += outw;
            }
        }
    }
}

template void dwConv_template<false, false>(float *, float *, int, int, int, int, int, float *, int, int, int, int, float *);
template void dwConv_template<false,  true>(float *, float *, int, int, int, int, int, float *, int, int, int, int, float *);
template void dwConv_template<true,  false>(float *, float *, int, int, int, int, int, float *, int, int, int, int, float *);
template void dwConv_template<true,   true>(float *, float *, int, int, int, int, int, float *, int, int, int, int, float *);