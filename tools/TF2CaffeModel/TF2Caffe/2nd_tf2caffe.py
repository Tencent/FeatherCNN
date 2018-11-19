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

import tensorflow as tf
import numpy as np
import google.protobuf as pb
import google.protobuf.text_format as pbtext
import caffe_pb2 as caffe
import pdb

tf.app.flags.DEFINE_boolean('use_tfplus', False, 'tfplus mode')
tf.app.flags.DEFINE_string('model_checkpoint_path', '../TFtrain/model/', 'the path of the saved model')
tf.app.flags.DEFINE_string('prototxt', './tran.prototxt', 'the net definition file')
tf.app.flags.DEFINE_string('model_name', './tran', 'the model name')

FLAGS = tf.app.flags.FLAGS
if FLAGS.use_tfplus:
    import tfplus.tensorflow as tfp

def tensor4d_transform_hwnc2cnhw(tensor):
    #hwcn -> nchw
    return tensor.transpose((3, 2, 0, 1))

def tensor4d_transformnhwc(tensor):
    #hwcn -> nchw
    return tensor.transpose((3, 0, 1, 2))

def tensor4d_transform(tensor):
    #hwcn -> nchw
    return tensor.transpose((3, 2, 0, 1))

def tensor4d_transform_depthwise(tensor):
    #hwn1 -> n1hw
    return tensor.transpose((2, 3, 0, 1))

def tensor2d_transform(tensor):
    #co,ci -> ci,co
    return tensor.transpose((1, 0))

def main(_):
    print(FLAGS.model_checkpoint_path)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
#        saver = tf.train.import_meta_graph('model.ckpt-7182.meta')
#        pdb.set_trace()  
        config = tf.ConfigProto()
        if FLAGS.use_tfplus:
            tfp.init()
            config.gpu_options.visible_device_list = str(tfp.local_rank())

        with tf.Session(config=config) as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)

            for v in tf.global_variables():
                print(v)

            net = caffe.NetParameter()
            with open(FLAGS.prototxt, 'r') as f:
                pbtext.Merge(f.read(), net)

            for layer in net.layer:
                print(layer.name + ': ' + str(len(layer.blobs)) + ' blobs', layer.type)
                for blob in layer.blobs:
                    if len(blob.shape.dim) > 0:
                        continue

                    name = ''.join([chr(int(c)) for c in blob.data])
                    name = name.replace("weights", "kernel", 1)
                    print(name)

                    variable = sess.graph.get_tensor_by_name(name)
                    if variable is not None:
                        data = np.array(sess.run(variable))
                        print(data.ndim, data.shape)

                        if data.ndim == 4:
                            if layer.type == "Convolution":
                                if layer.convolution_param.HasField('group'):
#                                    data = tensor4d_transform_depthwise(data)
                                    data = tensor4d_transform_hwnc2cnhw(data)
                                else:
                                    data = tensor4d_transform_hwnc2cnhw(data)
#                                    data = tensor4d_transformnhwc(data)
                                    print(data.shape)
                            else:
                                print('invalida type, check')
                                exit(-1)

                            shape = data.shape[:]
                            data = data.flatten()

                            blob.data[:] = data
                            blob.shape.dim[:] = shape
                            print("caffe shape ", shape)

                        elif data.ndim == 1:
                            data = data.flatten()
                            blob.data[:] = data
                            blob.shape.dim.extend([len(data)])
                        else:
                            print("invalid ndim")
                            exit(-1)
                    else:
                        print("name is None" % name)
                
            """
            caffemodel = FLAGS.model_name + '.caffemodel.txt'
            with open(caffemodel, 'w') as f:
                f.write(str(net))
            """

            caffemodel = FLAGS.model_name + '.caffemodel'
            with open(caffemodel, 'wb') as f:
                f.write(net.SerializeToString())

if __name__ == '__main__':
    tf.app.run()
