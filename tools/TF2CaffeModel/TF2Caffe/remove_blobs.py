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

import sys
import numpy as np
import google.protobuf as pb
import google.protobuf.text_format as pbtext
import cv2
import caffe_pb2 as caffe

cf_prototxt = "tran7.prototxt"
cf_prototxt_new = "tran7.prototxt"

def tensor4d_transform(tensor):
    #hwcn -> nchw
    return tensor.transpose((3, 2, 0, 1))

def tensor4d_transform_depthwise(tensor):
    #hwn1 -> n1hw
    return tensor.transpose((2, 3, 0, 1))

def tensor2d_transform(tensor):
    #co,ci -> ci,co
    return tensor.transpose((1, 0))

def tf2caffe_proto():
    net = caffe.NetParameter()
    with open(cf_prototxt, 'r') as f:
        pbtext.Merge(f.read(), net)

    for layer in net.layer:
        print(layer.name + ': ' + str(len(layer.blobs)) + ' blobs', layer.type)
        layer.ClearField('blobs')
    
    with open(cf_prototxt_new, 'w') as f:
        f.write(str(net))

    print("\n- Finished.\n")

if __name__ == '__main__':
    tf2caffe_proto()
