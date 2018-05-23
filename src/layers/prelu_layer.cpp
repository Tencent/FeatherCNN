#include "prelu_layer.h"
#include "arm/generic_kernels.h"

namespace feather{
int PReluLayer::Forward()
{
    const Blob<float> *p_bottom = _bottom_blobs[_bottom[0]];
    const float* input = p_bottom->data();
    float* output = _top_blobs[_top[0]]->data();

    int n = p_bottom->num();
    int c = p_bottom->channels();
    int h = p_bottom->height();
    int w = p_bottom->width();

    if ((0 == c) && (0 == h) && (0 != w))
    {
        if (shared)
        {
            float slope = slope_data[0];
#ifdef __ARM_NEON
			float32x4_t vzerof32x4 = vdupq_n_f32(0.f);
			float32x4_t vslopef32x4 = vdupq_n_f32(slope);
			int i = 0;
	        for (; i < w; i += 4)
	        {
	            float32x4_t vsrcf32x4 = vld1q_f32(&input[i]);
	            uint32x4_t vmasku32x4 = vcleq_f32(vsrcf32x4, vzerof32x4);
	            float32x4_t vmulf32x4 = vmulq_f32(vsrcf32x4, vslopef32x4);
	            vmulf32x4 = vbslq_f32(vmasku32x4, vmulf32x4, vsrcf32x4);
	            vst1q_f32(&output[i], vmulf32x4);
	        }
			for (; i < w; i++)
            {
                if (input[i] < 0)
                    output[i] = input[i]*slope;
                else
                    output[i] = input[i];
            }
#else
            for (int i=0; i<w; i++)
            {
                if (input[i] < 0)
                    output[i] = input[i]*slope;
                else
                    output[i] = input[i];
            }
#endif
        }
        else
        {
#ifdef __ARM_NEON
			float32x4_t vzerof32x4 = vdupq_n_f32(0.f);
			int i = 0;
	        for (; i < w; i += 4)
	        {
	        	float32x4_t vslopef32x4 = vld1q_f32(&slope_data[i]);
	            float32x4_t vsrcf32x4 = vld1q_f32(&input[i]);
	            uint32x4_t vmasku32x4 = vcleq_f32(vsrcf32x4, vzerof32x4);
	            float32x4_t vmulf32x4 = vmulq_f32(vsrcf32x4, vslopef32x4);
	            vmulf32x4 = vbslq_f32(vmasku32x4, vmulf32x4, vsrcf32x4);
	            vst1q_f32(&output[i], vmulf32x4);
	        }
			for (; i < w; i++)
            {
                if (input[i] < 0)
                    output[i] = input[i]*slope_data[i];
                else
                    output[i] = input[i];
            }
#else
            for (int i=0; i<w; i++)
            {
                if (input[i] < 0)
                    output[i] = input[i]*slope_data[i];
                 else
                    output[i] = input[i];
            }
#endif
        }
    }
    else if ((0 == c) && (0 != h) && (0 != w))
    {
        for (int i=0; i<h; i++)
        {
            const float* inPtr = input + i*w;
            float* outPtr = output + i*w;
            float slope = shared ? slope_data[0]:slope_data[i];
#ifdef __ARM_NEON
			float32x4_t vzerof32x4 = vdupq_n_f32(0.f);
	        float32x4_t vslopef32x4 = vdupq_n_f32(slope);
			int j = 0;
	        for (; j < w; j += 4)
	        {
	            float32x4_t vsrcf32x4 = vld1q_f32(&inPtr[j]);
	            uint32x4_t vmasku32x4 = vcleq_f32(vsrcf32x4, vzerof32x4);
	            float32x4_t vmulf32x4 = vmulq_f32(vsrcf32x4, vslopef32x4);
	            vmulf32x4 = vbslq_f32(vmasku32x4, vmulf32x4, vsrcf32x4);
	            vst1q_f32(&outPtr[j], vmulf32x4);
	        }
			for (; j < w; j++)
            {
                if (inPtr[j] < 0)
                    outPtr[j] = inPtr[j]*slope;
                else
                    outPtr[j] = inPtr[j];
            }
#else
            for (int j=0; j<w; j++)
            {
                if (inPtr[j] < 0)
                    outPtr[j] = inPtr[j]*slope;
                else
                    outPtr[j] = inPtr[j];
            }
#endif
        }
    }
    else if ((0 != c) && (0 != h) && (0 != w))
    {
        int size = w * h;
		#pragma omp parallel for num_threads(num_threads) schedule(guided)
        for (int q=0; q<c; q++)
        {
            const float* inPtr = input + q*size;
            float* outPtr = output + q*size;
            float slope = shared ? slope_data[0]:slope_data[q];

#ifdef __ARM_NEON
			float32x4_t vzerof32x4 = vdupq_n_f32(0.f);
	        float32x4_t vslopef32x4 = vdupq_n_f32(slope);
			int i = 0;
	        for (; i < size; i += 4)
	        {
	            float32x4_t vsrcf32x4 = vld1q_f32(&inPtr[i]);
	            uint32x4_t vmasku32x4 = vcleq_f32(vsrcf32x4, vzerof32x4);
	            float32x4_t vmulf32x4 = vmulq_f32(vsrcf32x4, vslopef32x4);
	            vmulf32x4 = vbslq_f32(vmasku32x4, vmulf32x4, vsrcf32x4);
	            vst1q_f32(&outPtr[i], vmulf32x4);
	        }
			for (; i < size; i++)
            {
                if (inPtr[i] < 0)
                    outPtr[i] = inPtr[i]*slope;
                else
                    outPtr[i] = inPtr[i];
            }
#else
            for (int i=0; i<size; i++)
            {
                if (inPtr[i] < 0)
                    outPtr[i] = inPtr[i]*slope;
                else
                    outPtr[i] = inPtr[i];
            }
#endif
        }
    }
    return 0;
}
};
