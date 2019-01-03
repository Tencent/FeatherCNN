//Tencent is pleased to support the open source community by making FeatherCNN available.

//Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.

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

template <typename Dtype, bool packed>
void caffe_cpu_interp2(const int channels,
                       const Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
                       Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2);