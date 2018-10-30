#include <booster/booster.h>

#include "util.h"

int main()
{
    booster::ConvParam conv_param;
    
    float* kernel_data = (float*) malloc(sizeof(float) * )
    

    booster::ConvBooster naive_booster;
    naive_booster.ForceSelectAlgo(booster::NAIVE);
    naive_booster.Init(&conv_param, naive_processed_kernel, kernel_data);

    booster::ConvBooster im2col_booster;
    
    
    return 0;
}
