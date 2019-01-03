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

#include "interp_layer.h"
#include "booster/generic_kernels.h"
#include "booster/caffe_interp.h"


namespace feather
{
int InterpLayer::Forward()
{
    caffe_cpu_interp2<float, false>(num_ * channels_,
                                    _bottom_blobs[_bottom[0]]->data(), -pad_beg_, -pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
                                    _top_blobs[_top[0]]->data(), 0, 0, height_out_, width_out_, height_out_, width_out_);
    return 0;
}
int InterpLayer::GenerateTopBlobs()
{
    if (_top.size() != 1 || _bottom.size() != 1)
        return -1;
    const Blob<float> *p_bottom = _bottom_blobs[_bottom[0]];
    num_ = p_bottom->num();
    channels_ = p_bottom->channels();
    height_in_ = p_bottom->height();
    width_in_ = p_bottom->width();
    height_in_eff_ = height_in_ + pad_beg_ + pad_end_;
    width_in_eff_ = width_in_ + pad_beg_ + pad_end_;
    // InterpParameter *interp_param = this->layer_param_->interp_param();


    // else if (interp_param.has_height() && interp_param.has_width())
    // {
    //     height_out_ = interp_param->height();
    //     width_out_ = interp_param->width();
    // }
    // CHECK_GT(height_in_eff_, 0) << "height should be positive";
    // CHECK_GT(width_in_eff_, 0) << "width should be positive";
    // CHECK_GT(height_out_, 0) << "height should be positive";
    // CHECK_GT(width_out_, 0) << "width should be positive";

    // _top_blobs[_top[0]]->Reshape(num_, channels_, height_out_, width_out_);

    // p_blob->CopyShape(_bottom_blobs[_bottom[0]]);
    Blob<float> *p_blob = new Blob<float>(num_, channels_, height_out_, width_out_);
    p_blob->Alloc();
    _top_blobs[_top[0]] = p_blob;
    return 0;
}
}; // namespace feather
