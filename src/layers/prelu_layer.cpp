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
    int w = p_bottom->width();
    int h = p_bottom->height();
    //printf("[PReLU] c:%d h:%d w:%d [%f %f %f %f]\n",c,h,w, input[0], input[1], input[2], input[3]);
    if ((0 == c) && (0 == h) && (0 != w))
    {
        if (shared)
        {
            float slope = slope_data[0];
            for (int i=0; i<w; i++)
            {
                if (input[i] < 0)
                    output[i] = input[i]*slope;
                else
                    output[i] = input[i];
            }
        }
        else
        {
            for (int i=0; i<w; i++)
            {
                if (input[i] < 0)
                    output[i] = input[i]*slope_data[i];
                 else
                    output[i] = input[i];
            }
        }
    }
    else if ((0 == c) && (0 != h) && (0 != w))
    {
        for (int i=0; i<h; i++)
        {
            const float* inPtr = input + i*w;
            float* outPtr = output + i*w;
            float slope = shared ? slope_data[0]:slope_data[i];

            for (int j=0; j<w; j++)
            {
                if (inPtr[j] < 0)
                    outPtr[j] = inPtr[j]*slope;
                else
                    outPtr[j] = inPtr[j];
            }
        }
    }
    else if ((0 != c) && (0 != h) && (0 != w))
    {
        int size = w * h;

        for (int q=0; q<c; q++)
        {
            const float* inPtr = input + q*size;
            float* outPtr = output + q*size;
            float slope = shared ? slope_data[0]:slope_data[q];
            #if 0
            //if ( 0 == _top[0].compare("prelu4"))
            {
                if ((0 != q) && (0 == (q%16))) printf("\n");
                printf("%9.6f ", slope);
                if (q == (c -1))
                    printf("\n");
            }
            #endif
            for (int i=0; i<size; i++)
            {
                if (inPtr[i] < 0)
                    outPtr[i] = inPtr[i]*slope;
                else
                    outPtr[i] = inPtr[i];
            }
        }
    }

#if 0
    printf("\n\n\n");
    if ( 0 == _top[0].compare("prelu4"))
    {
        for (int i = 0; i < w * h *c; i++)
        {
            printf(" %9.6f", input[i]);
            if((0 != i)&& (0 == i%16))
                printf("\n");
        }
        printf("\n\n");
        for (int i = 0; i < w * h *c; i++)
        {
            printf(" %9.6f", output[i]);
            if((0 != i)&& (0 == i%16))
                printf("\n");
        }   
    }
    printf("\n");
#endif
    return 0;
}
};
