#include "caffe.pb.h"
#include "feather_simple_generated.h"

#include <iostream>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdint.h>
#include <unistd.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

using namespace caffe;
using google::protobuf::io::FileInputStream;
using google::protobuf::Message;

class CaffeModelWeightsConvert
{
    public:
        CaffeModelWeightsConvert(std::string caffe_prototxt_name, std::string caffe_model_name, std::string output_name);
        bool Convert();
        void SaveModelWeights();

    private :
        bool ReadNetParam();

    private :
        std::string caffe_prototxt_name;
        std::string caffe_model_name;
        std::string output_name;
        NetParameter net_param;
        NetParameter net_param_prototxt;
};

CaffeModelWeightsConvert::CaffeModelWeightsConvert(std::string caffe_prototxt_name, std::string caffe_model_name, std::string output_name)
{
    this->caffe_prototxt_name = caffe_prototxt_name;
    this->caffe_model_name = caffe_model_name;
    this->output_name = output_name + ".feathermodel";
    printf("output_name %s\n", output_name.c_str());
}

bool CaffeModelWeightsConvert::Convert()
{

    if (!ReadNetParam())
    {
        std::cerr << "Read net params fail!" << std::endl;
        return false;
    }

    return true;

}

bool CaffeModelWeightsConvert::ReadNetParam()
{
    {
        std::ifstream in(caffe_model_name.c_str());
        std::stringstream buffer;
        buffer << in.rdbuf();
        std::string net_param_str(buffer.str());
        if (!net_param.ParseFromString(net_param_str))
        {
            std::cerr << "read caffe model weights file " << caffe_model_name  << " fail!" << std::endl;
            return false;
        }

        in.close();
    }

    {
        int fd = open(caffe_prototxt_name.c_str(), O_RDONLY);
        if (fd < 0)
        {
            std::cerr << "read caffe model prototxt " << caffe_prototxt_name  << " fail!" << std::endl;
            return false;
        }

        FileInputStream* input = new FileInputStream(fd);
        bool success = google::protobuf::TextFormat::Parse(input, &net_param_prototxt);
        delete input;
        close(fd);
    }
    return true;
}

void CaffeModelWeightsConvert::SaveModelWeights()
{
    //Writer
    {
        size_t input_layer_idx = -1;
        flatbuffers::FlatBufferBuilder fbb(204800);
        std::vector<flatbuffers::Offset<feather::LayerParameter>> layer_vec;
        /*
         * Loading net parameter from prototxt
         */
        size_t input_num = net_param_prototxt.input_size();
        printf("Input Num %ld\n", input_num);
        if (input_num > 0)
        {
            printf("Prototxt using legacy input params\n");
            std::vector<flatbuffers::Offset<flatbuffers::String>>   input_name_vec;
            std::vector<int64_t>                                    input_dim_vec;
            for (int i = 0; i < input_num; ++i)
            {
                std::string input_name = net_param_prototxt.input(i);
                printf("input name %s\n", input_name.c_str());
                input_name_vec.push_back(fbb.CreateString(input_name));
            }
            if (net_param_prototxt.input_shape_size())
            {
                for (int i = 0; i < net_param_prototxt.input_shape_size(); ++i)
                {
                    for (int j = 0; j < net_param_prototxt.input_shape(i).dim_size(); ++j)
                    {
                        size_t dim = net_param_prototxt.input_shape(i).dim(j);
                        printf("dim %ld\n", dim);
                        input_dim_vec.push_back((int64_t) dim);
                    }
                }
            }
            if (net_param_prototxt.input_dim_size())
            {
                printf("has input dim\n");
                for (int i = 0; i < net_param_prototxt.input_dim_size(); ++i)
                {
                    size_t dim = net_param_prototxt.input_dim(i);
                    printf("dim %ld\n", dim);
                    input_dim_vec.push_back(net_param_prototxt.input_dim(i));
                }
            }
            //Dim holds numbers of 4 * input_blob_num
            auto input_param = feather::CreateInputParameterDirect(fbb,
                               &input_name_vec,
                               &input_dim_vec);

            //build this layer
            auto input_layer_name = fbb.CreateString("input_layer");
            auto input_layer_type = fbb.CreateString("Input");
            //auto input_bottom = fbb.CreateString("");
            feather::LayerParameterBuilder layer_builder(fbb);
            layer_builder.add_name(input_layer_name);
            layer_builder.add_type(input_layer_type);
            layer_builder.add_input_param(input_param);
            //layer_builder.add_bottom(input_bottom);
            layer_vec.push_back(layer_builder.Finish());
        }
        else
        {
            //There should be an input layer
            printf("Input Layer\n");
            std::vector<flatbuffers::Offset<flatbuffers::String>>   input_name_vec;
            std::vector<int64_t>                                    input_dim_vec;
            for (int i = 0; i != net_param_prototxt.layer_size(); ++i)
            {
                auto caffe_layer = net_param_prototxt.layer(i);
                std::string layer_name = caffe_layer.name();
                std::string layer_type = caffe_layer.type();

                if (layer_type.compare("Input") == 0)
                {
                    input_name_vec.push_back(fbb.CreateString(layer_name));

                    assert(caffe_layer.input_param().shape_size() == 1);
                    for (int j = 0; j < caffe_layer.input_param().shape(0).dim_size(); ++j)
                    {
                        int64_t dim = caffe_layer.input_param().shape(0).dim(j);
                        printf("Input dim %ld\n", dim);
                        input_dim_vec.push_back(dim);
                    }
                }
            }
            auto input_param = feather::CreateInputParameterDirect(fbb,
                               &input_name_vec,
                               &input_dim_vec);

            //build this layer
            auto input_layer_name = fbb.CreateString("input_layer");
            auto input_layer_type = fbb.CreateString("Input");
            //auto input_bottom = fbb.CreateString("");
            feather::LayerParameterBuilder layer_builder(fbb);
            layer_builder.add_name(input_layer_name);
            layer_builder.add_type(input_layer_type);
            layer_builder.add_input_param(input_param);
            //layer_builder.add_bottom(input_bottom);
            layer_vec.push_back(layer_builder.Finish());
        }

        std::vector<std::string> prototxt_name_map;

        for (int i = 0; i != net_param_prototxt.layer_size(); ++i)
        {
            prototxt_name_map.push_back(net_param_prototxt.layer(i).name());
        }

        std::vector<float> blob_data_vec;
        printf("Layer num %d\n", net_param.layer_size());
        printf("Legacy layer num %d\n", net_param.layers_size());


        std::map<std::string, int> caffe_model_layer_map;
        for (int i = 0; i != net_param.layer_size(); ++i)
        {
            std::string layer_name = net_param.layer(i).name();
            caffe_model_layer_map[layer_name] = i;
        }

        std::map<std::string, std::string> inplace_blob_map;
        for (int i = 0; i != net_param_prototxt.layer_size(); ++i)
        {
            auto caffe_layer = net_param_prototxt.layer(i);
            std::string layer_name = caffe_layer.name();
            std::string layer_type = caffe_layer.type();

            if (layer_type.compare("Input") == 0)
                continue;

            std::vector<std::string> bottom_vec;
            std::vector<std::string> top_vec;

            auto caffe_model_layer = net_param.layer(caffe_model_layer_map[layer_name]);


            /*Bottom and top*/
            for (int j = 0; j < caffe_layer.bottom_size(); ++j)
                bottom_vec.push_back(caffe_layer.bottom(j));
            for (int j = 0; j < caffe_layer.top_size(); ++j)
                top_vec.push_back(caffe_layer.top(j));

            auto phase = caffe_layer.phase();
            printf("---------------------------------------\n");
            printf("Layer %d name %s type %s\n", i, layer_name.c_str(), layer_type.c_str());
            /*Print bottom and tops*/
            printf("Bottom ");
            for (int t = 0; t < bottom_vec.size(); ++t)
                printf("%s ", bottom_vec[t].c_str());
            printf("\n");
            printf("Top ");
            for (int t = 0; t < top_vec.size(); ++t)
                printf("%s ", top_vec[t].c_str());
            printf("\n");
            /*Reform inplace layers*/
            //If the bottom is in the inplace blob map.
            //should update this bottom
            if (bottom_vec.size() > 0 && top_vec.size() > 0)
            {
                if (bottom_vec[0].compare(top_vec[0]) == 0)
                {
                    assert(bottom_vec.size() == 1 && top_vec.size() == 1);
                    top_vec[0] = layer_name;
                    std::string bottom_name = bottom_vec[0];
                    if (inplace_blob_map.find(bottom_name) == inplace_blob_map.end())
                    {
                        inplace_blob_map[bottom_name] = bottom_name;
                    }
                    bottom_vec[0] = inplace_blob_map[bottom_name];
                    printf("bottom %s top %s\n", bottom_vec[0].c_str(), top_vec[0].c_str());
                    inplace_blob_map[bottom_name] = layer_name;
                }
                else
                {
                    for (int t = 0; t < bottom_vec.size(); ++t)
                    {
                        std::string bottom_name = bottom_vec[t];
                        if (inplace_blob_map.find(bottom_name) != inplace_blob_map.end())
                        {
                            bottom_vec[t] = inplace_blob_map[bottom_vec[t]];
                            printf("new bottom %s\n", bottom_vec[t].c_str());
                        }
                    }
                }
            }

            std::vector<flatbuffers::Offset<flatbuffers::String>> bottom_fbstr_vec;
            for (int i = 0; i < bottom_vec.size(); ++i)
                bottom_fbstr_vec.push_back(fbb.CreateString(bottom_vec[i]));
            auto bottom_fbvec = fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(bottom_fbstr_vec);

            std::vector<flatbuffers::Offset<flatbuffers::String>> top_fbstr_vec;
            for (int i = 0; i < top_vec.size(); ++i)
                top_fbstr_vec.push_back(fbb.CreateString(top_vec[i]));
            auto top_fbvec = fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(top_fbstr_vec);
            /*Blobs*/
            int blob_size = caffe_model_layer.blobs_size();
            printf("Blob num %d\n", blob_size);
            std::vector<flatbuffers::Offset<feather::BlobProto> > blob_vec;
            for (int j = 0; j != caffe_model_layer.blobs_size(); ++j)
            {
                auto caffe_blob = caffe_model_layer.blobs(j);
                for (int k = 0; k != caffe_blob.data_size(); ++k)
                {
                    float data = caffe_blob.data(k);
                    blob_data_vec.push_back(data);
                }
                auto blob_data_fbvec = fbb.CreateVector<float>(blob_data_vec);
                int dim_len = caffe_blob.shape().dim_size();
                long data_size = 1;
                feather::BlobProtoBuilder blob_builder(fbb);
                for (int m = 0; m < dim_len; ++m)
                {
                    long dim = caffe_blob.shape().dim(m);
                    data_size *= dim;
                    printf("%ld ", dim);
                }
                printf("data size %ld\n", data_size);
                blob_builder.add_data(blob_data_fbvec);

                if (dim_len == 0)
                {
                    size_t num = caffe_blob.num();
                    size_t channels = caffe_blob.channels();
                    size_t height = caffe_blob.height();
                    size_t width = caffe_blob.width();
                    printf("legacy shape (%lu %lu %lu %lu)\n", num, channels, height, width);
                    if (num == 1 && channels == 1 && height == 1 && width > 1)
                    {
                        num = width;
                        width = 1;
                    }
                    if (num == 1 && channels == 1 && height > 1 && width > 1)
                    {
                        num = height;
                        channels = width;
                        height = 1;
                        width = 1;
                    }
                    printf("Updated shape (%lu %lu %lu %lu)\n", num, channels, height, width);
                    blob_builder.add_num(num);
                    blob_builder.add_channels(channels);
                    blob_builder.add_height(height);
                    blob_builder.add_width(width);
                }
                else
                {
                    if (caffe_blob.shape().dim_size() == 4)
                    {
                        blob_builder.add_num(caffe_blob.shape().dim(0));
                        blob_builder.add_channels(caffe_blob.shape().dim(1));
                        blob_builder.add_height(caffe_blob.shape().dim(2));
                        blob_builder.add_width(caffe_blob.shape().dim(3));
                    }
                    else if (caffe_blob.shape().dim_size() == 1)
                    {
                        blob_builder.add_num(caffe_blob.shape().dim(0));
                        blob_builder.add_channels(1);
                        blob_builder.add_height(1);
                        blob_builder.add_width(1);
                    }
                    else if (caffe_blob.shape().dim_size() == 2)
                    {
                        blob_builder.add_num(caffe_blob.shape().dim(0));
                        blob_builder.add_channels(caffe_blob.shape().dim(1));
                        blob_builder.add_height(1);
                        blob_builder.add_width(1);
                    }
                    else if (caffe_blob.shape().dim_size() == 3)
                    {
                        blob_builder.add_num(1);
                        blob_builder.add_channels(caffe_blob.shape().dim(0));
                        blob_builder.add_height(caffe_blob.shape().dim(1));
                        blob_builder.add_width(caffe_blob.shape().dim(2));
                    }
                    else
                    {
                        fprintf(stderr, "Unsupported dimension with dim size %d\n", caffe_blob.shape().dim_size());
                    }
                }
                blob_vec.push_back(blob_builder.Finish());
                blob_data_vec.clear();
            }
            auto blobs_fbvec = fbb.CreateVector<flatbuffers::Offset<feather::BlobProto> >(blob_vec);
            blob_vec.clear();

            /*Params*/
            flatbuffers::Offset<feather::ConvolutionParameter> conv_param;
            flatbuffers::Offset<feather::LRNParameter> lrn_param;
            flatbuffers::Offset<feather::PoolingParameter> pooling_param;
            flatbuffers::Offset<feather::BatchNormParameter> bn_param;
            flatbuffers::Offset<feather::ScaleParameter> scale_param;
            flatbuffers::Offset<feather::SoftmaxParameter> softmax_param;
            flatbuffers::Offset<feather::EltwiseParameter> eltwise_param;
            flatbuffers::Offset<feather::InnerProductParameter> inner_product_param;
            flatbuffers::Offset<feather::PReLUParameter> prelu_param;
            flatbuffers::Offset<feather::DropoutParameter> dropout_param;

            if (layer_type.compare("Convolution") == 0 || (layer_type.compare("ConvolutionDepthwise") == 0))
            {
                printf("+ %s\n", layer_type.c_str());
                auto caffe_conv_param = caffe_layer.convolution_param();
                feather::ConvolutionParameterBuilder conv_param_builder(fbb);
                printf("bias term %d\n", caffe_conv_param.bias_term());
                conv_param_builder.add_bias_term(caffe_conv_param.bias_term());

                conv_param_builder.add_kernel_h(caffe_conv_param.kernel_size(0));
                if (caffe_conv_param.kernel_size_size() == 1)
                    conv_param_builder.add_kernel_w(caffe_conv_param.kernel_size(0));
                else if (caffe_conv_param.kernel_size_size() == 2)
                    conv_param_builder.add_kernel_w(caffe_conv_param.kernel_size(1));
                else
                    ;

                if (caffe_conv_param.stride_size() == 1)
                {
                    printf("+ stride %d\n", caffe_conv_param.stride(0));
                    conv_param_builder.add_stride_h(caffe_conv_param.stride(0));
                    conv_param_builder.add_stride_w(caffe_conv_param.stride(0));
                }
                else if (caffe_conv_param.stride_size() == 2)
                {
                    conv_param_builder.add_stride_h(caffe_conv_param.stride(0));
                    conv_param_builder.add_stride_w(caffe_conv_param.stride(1));
                }
                else if (caffe_conv_param.stride_size() == 0)
                {
                    //defaults to 1
                    conv_param_builder.add_stride_h(1);
                    conv_param_builder.add_stride_w(1);
                }
                else
                {
                    fprintf(stderr, "More stride dim than expected!\n");
                    exit(-1);
                }
                printf("pad %d pad_h %d pad_w %d", caffe_conv_param.pad_size(), caffe_conv_param.has_pad_h(), caffe_conv_param.has_pad_w());
                if (caffe_conv_param.pad_size() == 1)
                {
                    conv_param_builder.add_pad_h(caffe_conv_param.pad(0));
                    conv_param_builder.add_pad_w(caffe_conv_param.pad(0));
                }
                else if (caffe_conv_param.pad_size() == 2)
                {
                    conv_param_builder.add_pad_h(caffe_conv_param.pad(0));
                    conv_param_builder.add_pad_w(caffe_conv_param.pad(1));
                }
                else if (caffe_conv_param.pad_size() == 0 && caffe_conv_param.has_pad_h() && caffe_conv_param.has_pad_w())
                {
                    conv_param_builder.add_pad_h(caffe_conv_param.pad_h());
                    conv_param_builder.add_pad_w(caffe_conv_param.pad_w());
                }
                else
                {
                    fprintf(stderr, "Going for default padding config pad_size %d\n", caffe_conv_param.pad_size());
                    //Go for default padding
                    conv_param_builder.add_pad_h(0);
                    conv_param_builder.add_pad_w(0);
                }

                if (layer_type.compare("ConvolutionDepthwise") == 0)
                    conv_param_builder.add_group(caffe_conv_param.num_output());
                else
                    conv_param_builder.add_group(caffe_conv_param.group());

                printf("+ num_output %u\n", caffe_conv_param.num_output());
                printf("+ kernel_h %d\n", caffe_conv_param.kernel_size(0));
                printf("+ stride_size %d\n", caffe_conv_param.stride_size());
                printf("+ group %d\n", caffe_conv_param.group());
                conv_param = conv_param_builder.Finish();
            }
            else if (layer_type.compare("LRN") == 0)
            {
                auto caffe_lrn_param = caffe_layer.lrn_param();
                size_t local_size = caffe_lrn_param.local_size();
                float alpha = caffe_lrn_param.alpha();
                float beta = caffe_lrn_param.beta();
                float k = caffe_lrn_param.k();
                printf("LRN param: local_size %ld alpha %f beta %f k %f\n", local_size, alpha, beta, k);
                feather::LRNParameterBuilder lrn_param_builder(fbb);
                lrn_param_builder.add_local_size(local_size);
                lrn_param_builder.add_alpha(alpha);
                lrn_param_builder.add_beta(beta);
                lrn_param_builder.add_k(k);
                switch (caffe_lrn_param.norm_region())
                {
                    case caffe::LRNParameter_NormRegion_ACROSS_CHANNELS:
                        printf("Across channels\n");
                        lrn_param_builder.add_norm_region(feather::LRNParameter_::NormRegion_ACROSS_CHANNELS);
                        break;
                    case caffe::LRNParameter_NormRegion_WITHIN_CHANNEL:
                        printf("Within channels\n");
                        lrn_param_builder.add_norm_region(feather::LRNParameter_::NormRegion_WITHIN_CHANNEL);
                        break;
                    default:
                        fprintf(stderr, "Unknown LRN method\n");
                        exit(-1);
                }
                lrn_param = lrn_param_builder.Finish();
            }
            else if (layer_type.compare("Pooling") == 0)
            {
                auto caffe_pooling_param = caffe_layer.pooling_param();
                feather::PoolingParameterBuilder pooling_param_builder(fbb);
                switch (caffe_pooling_param.pool())
                {
                    case caffe::PoolingParameter_PoolMethod_MAX:
                        pooling_param_builder.add_pool(feather::PoolingParameter_::PoolMethod_MAX_);
                        break;
                    case caffe::PoolingParameter_PoolMethod_AVE:
                        pooling_param_builder.add_pool(feather::PoolingParameter_::PoolMethod_AVE);
                        break;
                    case caffe::PoolingParameter_PoolMethod_STOCHASTIC:
                        pooling_param_builder.add_pool(feather::PoolingParameter_::PoolMethod_STOCHASTIC);
                        break;
                    default:
                        //error handling
                        ;
                }
                if (caffe_pooling_param.has_pad())
                {
                    pooling_param_builder.add_pad_h(caffe_pooling_param.pad());
                    pooling_param_builder.add_pad_w(caffe_pooling_param.pad());
                }
                else
                {
                    pooling_param_builder.add_pad_h(caffe_pooling_param.pad_h());
                    pooling_param_builder.add_pad_w(caffe_pooling_param.pad_w());
                }
                if (caffe_pooling_param.has_kernel_size())
                {
                    pooling_param_builder.add_kernel_h(caffe_pooling_param.kernel_size());
                    pooling_param_builder.add_kernel_w(caffe_pooling_param.kernel_size());
                }
                else
                {
                    pooling_param_builder.add_kernel_h(caffe_pooling_param.kernel_h());
                    pooling_param_builder.add_kernel_w(caffe_pooling_param.kernel_w());
                }
                //pooling_param_builder.add_kernel_size(caffe_pooling_param.kernel_size());
                if (caffe_pooling_param.has_stride())
                {
                    pooling_param_builder.add_stride_h(caffe_pooling_param.stride());
                    pooling_param_builder.add_stride_w(caffe_pooling_param.stride());
                }
                else
                {
                    pooling_param_builder.add_stride_h(caffe_pooling_param.stride_h());
                    pooling_param_builder.add_stride_w(caffe_pooling_param.stride_w());
                }
                pooling_param_builder.add_global_pooling(caffe_pooling_param.global_pooling());
                pooling_param = pooling_param_builder.Finish();
            }
            else if (layer_type.compare("InnerProduct") == 0)
            {
                auto caffe_inner_product_param = caffe_layer.inner_product_param();
                feather::InnerProductParameterBuilder inner_product_param_builder(fbb);
                inner_product_param_builder.add_bias_term(caffe_inner_product_param.bias_term());
                inner_product_param = inner_product_param_builder.Finish();
            }
            else if (layer_type.compare("BatchNorm") == 0)
            {
                //Do nothing
            }
            else if (layer_type.compare("Softmax") == 0)
            {
                auto caffe_softmax_param = caffe_layer.softmax_param();
                feather::SoftmaxParameterBuilder softmax_param_builder(fbb);
                softmax_param_builder.add_axis(caffe_softmax_param.axis());
            }
            else if (layer_type.compare("Scale") == 0)
            {
                auto caffe_scale_param = caffe_layer.scale_param();
                printf("Scale param %d\n", caffe_scale_param.bias_term());
                feather::ScaleParameterBuilder scale_param_builder(fbb);
                scale_param_builder.add_bias_term(caffe_scale_param.bias_term());
                scale_param = scale_param_builder.Finish();
            }
            else if (layer_type.compare("Eltwise") == 0)
            {
                auto caffe_eltwise_param = caffe_layer.eltwise_param();
                auto op = caffe_eltwise_param.operation();
                feather::EltwiseParameter_::EltwiseOp feather_op;
                switch (op)
                {
                    case EltwiseParameter_EltwiseOp_PROD:
                        printf("PROD op\n");
                        feather_op = feather::EltwiseParameter_::EltwiseOp_PROD;
                        break;
                    case EltwiseParameter_EltwiseOp_SUM:
                        printf("SUM op\n");
                        feather_op = feather::EltwiseParameter_::EltwiseOp_SUM;
                        break;
                    case EltwiseParameter_EltwiseOp_MAX:
                        printf("MAX op\n");
                        feather_op = feather::EltwiseParameter_::EltwiseOp_MAX;
                        break;
defalut:
                        fprintf(stderr, "Unknown eltwise parameter.\n");
                }
                std::vector<float> coeff_vec;
                for (int i = 0; i < caffe_eltwise_param.coeff_size(); ++i)
                {
                    coeff_vec.push_back(caffe_eltwise_param.coeff(i));
                }
                printf("Loaded coeff size %ld\n", coeff_vec.size());
                eltwise_param = feather::CreateEltwiseParameterDirect(fbb, feather_op, &coeff_vec);
            }
            else if (layer_type.compare("ReLU") == 0)
            {
                //Do nothing
            }
            else if (layer_type.compare("PReLU") == 0)
            {

            }
            else if (layer_type.compare("Dropout") == 0)
            {

            }

            auto layer_name_fbb = fbb.CreateString(layer_name);
            auto layer_type_fbb = fbb.CreateString(layer_type);
            feather::LayerParameterBuilder layer_builder(fbb);
            layer_builder.add_bottom(bottom_fbvec);
            layer_builder.add_top(top_fbvec);
            layer_builder.add_blobs(blobs_fbvec);
            layer_builder.add_name(layer_name_fbb);
            layer_builder.add_type(layer_type_fbb);
            if (layer_type.compare("Convolution") == 0 || (layer_type.compare("ConvolutionDepthwise") == 0))
                layer_builder.add_convolution_param(conv_param);
            else if (layer_type.compare("LRN") == 0)
                layer_builder.add_lrn_param(lrn_param);
            else if (layer_type.compare("Pooling") == 0)
                layer_builder.add_pooling_param(pooling_param);
            else if (layer_type.compare("InnerProduct") == 0)
                layer_builder.add_inner_product_param(inner_product_param);
            else if (layer_type.compare("Scale") == 0)
                layer_builder.add_scale_param(scale_param);
            else if (layer_type.compare("Eltwise") == 0)
                layer_builder.add_eltwise_param(eltwise_param);
            layer_vec.push_back(layer_builder.Finish());
        }
        auto layer_fbvec = fbb.CreateVector<flatbuffers::Offset<feather::LayerParameter>>(layer_vec);
        auto name_fbb = fbb.CreateString(net_param_prototxt.name());
        feather::NetParameterBuilder net_builder(fbb);
        net_builder.add_layer(layer_fbvec);
        net_builder.add_name(name_fbb);
        auto net = net_builder.Finish();
        fbb.Finish(net);
        uint8_t* net_buffer_pointer = fbb.GetBufferPointer();
        size_t size = fbb.GetSize();
        printf("Net buffer %lx size %ld\n", (long)net_buffer_pointer, size);

        //Writer
        FILE *netfp = NULL;
        netfp = fopen(output_name.c_str(), "wb");
        fwrite(net_buffer_pointer, sizeof(uint8_t), size, netfp);
        fclose(netfp);
    }

#if 1
    //Loader
    {
        printf("++++++Start Loader++++++\n");
        FILE *netfp = NULL;
        netfp = fopen(output_name.c_str(), "rb");
        fseek(netfp, 0, SEEK_END);
        long file_size = ftell(netfp);
        ++file_size;
        fseek(netfp, 0, SEEK_SET);
        uint8_t *net_buffer_pointer = (uint8_t *) malloc(sizeof(uint8_t) * file_size);
        size_t read_size = fread(net_buffer_pointer, sizeof(uint8_t), file_size, netfp);
        fclose(netfp);

        auto net_loader = feather::GetNetParameter(net_buffer_pointer);
        auto layer_num = net_loader->layer()->Length();
        printf("++++++%d layers loaded++++++\n", layer_num);
        for (int i = 0; i < layer_num; ++i)
        {
            auto layer = net_loader->layer()->Get(i);
            std::string layer_name(layer->name()->str());
            std::string layer_type(layer->type()->str());
            printf("Layer %s id %d type %s\n", layer_name.c_str(), i, layer_type.c_str());
            for (int b = 0; b < flatbuffers::VectorLength(layer->bottom()); ++b)
            {
                printf("Bottom %s\n", layer->bottom()->Get(b)->c_str());
            }
            for (int b = 0; b < flatbuffers::VectorLength(layer->top()); ++b)
            {
                printf("Top %s\n", layer->top()->Get(b)->c_str());
            }
        }
        printf("+++++++++++++++++++++++++++\n");
        free(net_buffer_pointer);
    }
#endif
}



int main(int argc, char *argv[])
{
    if (argc < 3 || argc > 4)
    {
        printf("Usage: ./caffe_model_convert $1(caffe_prototxt) $2(caffe_model_name) [$3(output_model_name_prefix)]\n");
        return -1;
    }
    std::string caffe_prototxt_name = argv[1];
    std::string caffe_model_name = argv[2];
    std::string output_model_name;
    if (argc == 3)
        output_model_name = "out";//Default output name
    else if (argc == 4)
        output_model_name = (argv[3]);
    else
    {
        fprintf(stderr, "Unexpected argc value.\n");
        return -1;
    }
    CaffeModelWeightsConvert convert(caffe_prototxt_name, caffe_model_name, output_model_name);
    convert.Convert();
    convert.SaveModelWeights();
    return 0;
}
