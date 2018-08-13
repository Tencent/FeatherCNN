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
#include "conv_layer.h"
#include "blob.h"

#include "arm/generic_kernels.h"
#include "arm/helper.h"
#include "arm/sgeconv.h"

#include "arm/sgemm.h"

#include <assert.h>
#include <stdio.h>

//#define CHECK_RESULTS


#ifdef CHECK_RESULTS
void im2col(ConvParam *conv_param, float *img_buffer, float *input)
{
    const int stride = conv_param->kernel_h * conv_param->kernel_w * conv_param->output_h * conv_param->output_w;
    float *ret = img_buffer;
    for (int k = 0; k < conv_param->input_channels; k++)
    {
        int retID = stride * k;
        for (int u = 0; u < conv_param->kernel_h; u++)
        {
            for (int v = 0; v < conv_param->kernel_w; v++)
            {
                for (int i = 0; i < conv_param->output_h; i++)
                {
                    for (int j = 0; j < conv_param->output_w; j++)
                    {
                        //calculate each row
                        int row = u - conv_param->pad_top + i * conv_param->stride_h;
                        int col = v - conv_param->pad_left + j * conv_param->stride_w;
                        //printf("row %d, col %d\n", row, col);
                        if (row < 0 || row >= conv_param->input_h || col < 0 || col >= conv_param->input_w)
                        {
                            ret[retID] = 0;
                        }
                        else
                        {
                            size_t index = k * conv_param->input_w * conv_param->input_h + row * conv_param->input_w + col; //(i+u)*input_width+j+v;
                            ret[retID] = input[index];
                        }
                        retID++;
                    }
                }
            }
        }
    }
}
void naive_sgemm_chk(int M, int N, int L, float* A, float* B, float* C)
{
    for (int i = 0; i < M; ++i) //loop over rows in C
    {
        for (int j = 0; j < N; ++j) //loop over columns in C
        {
            float sigma = 0;
            for (int k = 0; k < L; ++k)
            {
                sigma += A[i * L + k] * B[k * N + j];
            }
            C[i * N + j] = sigma;
        }
    }
}
#endif
namespace feather
{
class ConvSgeconvLayer : public ConvLayer
{
  public:
    ConvSgeconvLayer(const LayerParameter *layer_param, const RuntimeParameter<float> *rt_param)
        : fuse_relu(false), kc(0), nc(0), ConvLayer(layer_param, rt_param)
    {
        _fusible = true;
        //kc = 304;
        //nc = 304;
        kc = 32;
        nc = 360;
        //kc = 400;
        //nc = 400;
    }

    int Forward()
    {
        const int M = output_channels;
        const int N = conv_param.output_h * conv_param.output_w;
        const int K = input_channels * kernel_width * kernel_height;
        float *padded_input = NULL;
        int offset = conv_param_no_padding.input_h * conv_param_no_padding.input_w * conv_param_no_padding.input_channels;
        MEMPOOL_CHECK_RETURN(common_mempool->GetPtr(&padded_input));
//        memset(padded_input, 0, sizeof(float) * conv_param_no_padding.input_h * conv_param_no_padding.input_w * input_channels);
//        sgeconv_dev::pad_input_neon(&conv_param, padded_input, input);
         pad_input(padded_input, input, input_channels, input_width, input_height, padding_left, padding_top, padding_right, padding_bottom);
//        sgeconv_dev::packed_sgeconv_im2col_activation<false, false>(&conv_param_no_padding, packed_kernel, padded_input, N, output, N, nc, kc, bias_data, num_threads, padded_input+pad_size);
        packed_sgeconv(&conv_param_no_padding, packed_kernel, padded_input, N, output, N, nc, kc, bias_data, num_threads, padded_input + offset);
#ifdef CHECK_RESULTS
//        float pack_array[(kc * conv_param.kernel_w * conv_param.kernel_h + 8) * nc];
        // Im2col();
        im2col(&conv_param, img_buffer, input);
        printf("SGEMM M %d N %d K %d\n", M, N, K);
        // packed_sgemm_activation<false, false>(M, N, K, packed_kernel, img_buffer, N, output_dup, N, nc, kc * conv_param.kernel_w * conv_param.kernel_h, bias_data, 1, pack_array);
        naive_sgemm_chk(output_channels, output_height * output_width, input_channels * kernel_width * kernel_height, kernel_data, img_buffer, output);

        double err = 0.f;
        // for (int c = 0; c < output_channels; ++c)
        // if(this->name() == "conv1")
        {
            for (int i = 0; i < output_width * output_height; ++i)
            {
                err += output[i] - output_dup[i];
                if ((output[i] - output_dup[i]) > 0.1f)
                {
                    // printf("ERROR in layer %s\n", this->name().c_str());
                    // break;
                    printf("idx (%d %d) %f vs correct %f\n", i / output_width, i % output_width, output[i], output_dup[i]);
                }
            }
            printf("err %lf\n", err);
        }
        if (bias_term)
        {
            size_t out_stride = output_width * output_height;
            for (int i = 0; i < output_channels; ++i)
            {
                float bias = bias_data[i];
                for (int j = 0; j < out_stride; ++j)
                {
                    output[out_stride * i + j] = output[out_stride * i + j] + bias;
                }
            }
        }
#endif

        return 0;
    }

    virtual int ForwardReshape()
    {
        const Blob<float> *bottom_blob = _bottom_blobs[_bottom[0]];
        input_height = bottom_blob->height();
        input_width = bottom_blob->width();

        output_width = (input_width + padding_left + padding_right - kernel_width) / stride_width + 1;
        output_height = (input_height + padding_top + padding_bottom - kernel_height) / stride_height + 1;

        conv_param.input_h = bottom_blob->height();
        conv_param.input_w = bottom_blob->width();
        conv_param.AssignOutputDim();

        conv_param_no_padding = conv_param;
        conv_param_no_padding.AssignPaddedDim();

        //Global memory allocations
        _top_blobs[_top[0]]->ReshapeWithRealloc(1, output_channels, output_height, output_width);
        input = _bottom_blobs[_bottom[0]]->data();
        output = _top_blobs[_top[0]]->data();
        MEMPOOL_CHECK_RETURN(common_mempool->Alloc(sizeof(float) * (input_channels * kernel_height * kernel_width) * (output_width * output_height)))

        return this->Forward();
    }

    int GenerateTopBlobs()
    {
        //Conv layer has and only has one bottom blob.
        const Blob<float> *bottom_blob = _bottom_blobs[_bottom[0]];

        input_width = bottom_blob->width();
        input_height = bottom_blob->height();
        input_channels = bottom_blob->channels();
        if (stride_width == 0 || stride_height == 0)
        {
            stride_width = 1;
            stride_height = 1;
        }
        output_width = (input_width + padding_left + padding_right - kernel_width) / stride_width + 1;
        output_height = (input_height + padding_top + padding_bottom - kernel_height) / stride_height + 1;
        _top_blobs[_top[0]] = new Blob<float>(1, output_channels, output_height, output_width);
        _top_blobs[_top[0]]->Alloc();
        int M = output_channels;
        return 0;
    }

    int Fuse(Layer *next_layer)
    {
        if (next_layer->type().compare("ReLU") == 0)
        {
            fuse_relu = true;
            return 1;
        }
        else
        {
            return 0;
        }
    }

    int Init()
    {
        conv_param.output_channels = output_channels;
        conv_param.input_channels = input_channels;
        conv_param.input_w = input_width;
        conv_param.input_h = input_height;
        conv_param.kernel_h = kernel_height;
        conv_param.kernel_w = kernel_width;
        conv_param.stride_h = stride_height;
        conv_param.stride_w = stride_width;
        conv_param.pad_left = padding_left;
        conv_param.pad_bottom = padding_bottom;
        conv_param.pad_right = padding_right;
        conv_param.pad_top = padding_top;
        conv_param.AssignOutputDim();
        conv_param_no_padding = conv_param;
        conv_param_no_padding.AssignPaddedDim();
        printf("input w %d input h %d\n", conv_param.input_w, conv_param.input_h);
        int M = (int)output_channels;
        int N = output_height * output_width;
        int K = (int)input_channels * (int)kernel_height * (int)kernel_width;

        MEMPOOL_CHECK_RETURN(private_mempool.Alloc(&packed_kernel, sizeof(float) * (M * K)))

        sgeconv_dev::packed_sgeconv_init<4>(&conv_param, kc * kernel_height * kernel_width, packed_kernel, kernel_data);
                //  packed_sgeconv_init<4>(&conv_param, kc * conv_param.kernel_h * conv_param.kernel_w, packed_kernel, testKernel);
        if (bias_term && fuse_relu)
            packed_sgeconv = sgeconv_dev::packed_sgeconv_im2col_activation<true, true>;
        else if (bias_term)
            packed_sgeconv = sgeconv_dev::packed_sgeconv_im2col_activation<true, false>;
        else if (fuse_relu)
            packed_sgeconv = sgeconv_dev::packed_sgeconv_im2col_activation<false, true>;
        else
            packed_sgeconv = sgeconv_dev::packed_sgeconv_im2col_activation<false, false>;
        MEMPOOL_CHECK_RETURN(common_mempool->Request(sizeof(float) * (conv_param_no_padding.input_channels * conv_param_no_padding.input_w * conv_param_no_padding.input_h + kc * nc * kernel_width * kernel_height)))
        //Setup input and output pointers.
        input = _bottom_blobs[_bottom[0]]->data();
        output = _top_blobs[_top[0]]->data();
#ifdef CHECK_RESULTS
        output_dup = (float *)malloc(sizeof(float) * output_channels * output_width * output_height);
        img_buffer = (float *)malloc(sizeof(float) * input_channels * output_width * output_height * kernel_width * kernel_height);
#endif
        return 0;
    }

  private:
    float *packed_kernel;
    float *packed_kernel_dup;

    float *input;
    float *output;

    bool fuse_relu;
    int kc, nc;

    void (*packed_sgeconv)(ConvParam *, float *packA, float *B, int ldb, float *C, int ldc, int nc, int kc, float *bias, int num_threads, float* pack_array);

    ConvParam conv_param;
    ConvParam conv_param_no_padding;
#ifdef CHECK_RESULTS
    float *output_dup;
    float *img_buffer;
#endif
};
}; // namespace feather
