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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# add itl to python path 
import sys
sys.path.insert(0, "./")
import tensorflow as tf
sys.path.insert(0, "../")
import itl as tl
from itl.layers.inputs import InputLayer
from itl.layers.convolution import Conv2d
from itl.layers.pooling import MaxPool2d
from itl.layers.pooling import MeanPool2d
from itl.layers.shape import FlattenLayer
from itl.layers.dense import DenseLayer
from itl.layers.dropout import DropoutLayer

import tensorflow.contrib.slim as slim

# training setting
# weights_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
weights_init = tf.random_normal_initializer(stddev=0.01)
bias_init = tf.constant_initializer(0.0, tf.float32)
activation = tf.nn.relu

lr_mult = {"weights":1, "bias":2, "W":1, "b":2}
#lr_mult = {"kernel":1, "bias":2, "W":1, "b":2}
weight_decay = 0.00001
bias_decay = 0
W_lr_and_decay=(lr_mult["weights"], weight_decay)
#W_lr_and_decay=(lr_mult["kernel"], weight_decay)
b_lr_and_decay=(lr_mult["bias"], bias_decay)


class Model(object):

    def __init__(self):
        self.gradient_ops = {}
        self.img_size = [32, 32]
        self.num_classes = 3
        self.is_training = True
        self.growth_rate = 48
        self.dropout_rate = 0.6
        self.num_classes = 2

        self.data_format = "NHWC"
        if self.data_format == "NHWC":
            self.channel_pos = "channels_last"
            self.channel_index = 3
        else:
            self.channel_pos = "channels_first"
            self.channel_index = 1
        pass

    def conv_layer(self, inputs, filters, kernel, stride=1, padding=(0,0), layer_name="conv"):
        kernel_size = kernel[0]
#        network = tl.layers.Conv2dLayer(inputs, act=None, shape=(kernel_size, kernel_size, inputs.outputs.shape[self.channel_index], filters), strides=(1,stride,stride,1), name=layer_name)
        network = tl.layers.Conv2d(inputs, filters, kernel, strides=(stride,stride), act=None, padding=padding, name=layer_name)
        return network

    def global_average_pooling(self, inputs):
        network = tl.layers.GlobalMeanPool2d(inputs)
        return network
#        if self.data_format == "NHWC":
#            return tf.reduce_mean(inputs, [1, 2])
#        else:
#            return tf.reduce_mean(inputs, [2, 3])

    def batch_normlization(self, inputs, layer_name="batchnorm"):
        network = tl.layers.BatchNormLayer(inputs, decay=0.9, act=tf.nn.relu, is_train=self.is_training, name=layer_name)
        return network

    def dropout(self, inputs, rate, layer_name="dropout"):
#        network = tl.layers.DropoutLayer(inputs, keep=rate, is_fix=True, is_train=True, name=layer_name)
#        network = tl.layers.DropoutLayer(inputs, keep=rate, is_fix=True, is_train=True, seed=None, name=layer_name)
#            self.dp3 = DropoutLayer(self.fc3, keep=0.6, is_fix=True, is_train=is_training, seed=None, name='dropout3')
        return inputs

    def relu(self, inputs):
        return tf.nn.relu(inputs)

    def average_pooling(self, inputs, pool_size=[2, 2], strides=2, padding='VALID', layer_name="pool"):
#        network = tl.layers.MeanPool2d(inputs, filter_size=pool_size, strides=strides, padding=padding, data_format=self.channel_pos, name=layer_name)
#        network = tl.layers.MeanPool2d(inputs, filter_size=pool_size, strides=strides, padding=padding, name=layer_name)
#        network = tl.layers.MeanPool2d(inputs, filter_size=pool_size, strides=strides, padding=(0,0), name=layer_name)
#        network = MaxPool2d(inputs, filter_size=pool_size, strides=(strides, strides), padding=(0,0), name=layer_name)
        network = tl.layers.MeanPool2d(inputs, filter_size=pool_size, strides=(strides, strides), padding=(0,0), name=layer_name)
        return network

    def concat(self, inputs, layer_name="concat"):
        network = tl.layers.ConcatLayer(inputs, self.channel_index, name=layer_name)
        return network

    def linear(self, inputs, units):
        return tf.layers.dense(inputs=inputs, units=units, name="linear")

    def bottleneck_layer(self, x, block_name="dense"):
        name = block_name
        x = self.batch_normlization(x, layer_name=name+"bn0")
        x = self.conv_layer(x, filters=4*self.growth_rate, kernel=[1, 1], layer_name=name+"depthwise")
        x = self.dropout(x, rate=self.dropout_rate, layer_name=name+"dropout0")

        x = self.batch_normlization(x, layer_name=name+"bn1")
        x = self.conv_layer(x, filters=self.growth_rate, kernel=[3, 3], padding=(1,1), layer_name=name+"conv")
        x = self.dropout(x, rate=self.dropout_rate, layer_name=name+"dropout1")

        return x

    def transition_layer(self, x, block_name="tran"):
        name = block_name
        x = self.batch_normlization(x, layer_name=name+"bn0")

        filter_size = x.outputs.shape[self.channel_index]
        transition_filter_size = int(int(filter_size) / 2)

#        tf.Print(transition_filter_size, [transition_filter_size], message='Debug message:',summarize=100)

        x = self.conv_layer(x, filters=transition_filter_size, kernel=[3, 3], padding=[1,1], layer_name=name+"conv")
        x = self.dropout(x, rate=self.dropout_rate, layer_name=name+"dropout1")
        x = self.average_pooling(x, pool_size=[2, 2], strides=2, layer_name=name+"pool")

        return x

    def dense_block(self, input_x, nb_layers, block_name="dense"):
        layers_concat = list()
        layers_concat.append(input_x)
        x = self.bottleneck_layer(input_x, block_name+"_"+str(0))
        layers_concat.append(x)

        for i in range(nb_layers - 1):
            x = self.concat(layers_concat, layer_name=block_name+str(i))
            x = self.bottleneck_layer(x, block_name+"_"+str(i+1))
            layers_concat.append(x)

        x = self.concat(layers_concat, layer_name=block_name+str(nb_layers))
        return x

    def max_pooling(self, inputs, pool_size=[3,3], stride=2, padding='VALID', layer_name="max_pool"):
        network = tl.layers.MaxPool2d(inputs, pool_size, strides=(stride, stride), padding=(0,0), name=layer_name)
#        network = tl.layers.MaxPool2d(inputs, pool_size, strides=(stride, stride), padding=(0,0), name=layer_name)
        return network

    def network(self, images):
        x = images
        x = self.conv_layer(x, filters=2*24, kernel=[5, 5], stride=2, layer_name="conv0")

        ## add max pooling
        x = self.max_pooling(x, padding = 'SAME')

        x = self.dense_block(input_x=x, nb_layers=6, block_name="dense0")
        x = self.transition_layer(x, block_name="tran0")

#        x = self.batch_normlization(x,"bn_out0")
        x = self.dense_block(input_x=x, nb_layers=6, block_name="dense1")
        x = self.transition_layer(x, block_name="tran1")

        x = self.dense_block(input_x=x, nb_layers=6, block_name="dense2")

        x = self.batch_normlization(x, "bn_out1")
        x = self.dropout(x, rate=self.dropout_rate, layer_name="dropout1")

#-----------------------------
#        x = self.global_average_pooling(x)
#        x = tl.layers.FlattenLayer(x, name="flatten")
#        x = FlattenLayer(x, name='flatten')
#        x = tl.layers.DenseLayer(x, self.num_classes, act=None) 
#        x = self.linear(x.outputs, self.num_classes)

        return x


    def build(self, rgb, is_training=True):
        """
        Build the resnet model using loaded weights
        Parameters
        ----------
        rgb: image batch tensor from tf.decode_png, uint8
            Image in rgb shap. Scaled to Intervall [0, 255]
        is_training: bool
            Whether to build train or inference graph
        """
        
        # change RGB [0, 255] to mean RGB [0, 1]
        rgb = tf.cast(rgb, tf.float32)
        self.rgb = tf.multiply(rgb, 1/255.0, name="data")
        
        with tf.variable_scope("simplenet", reuse=None):
            self.input = InputLayer(self.rgb, "input")
            self.netout = self.network(self.input)

#         self.conv1 = Conv2d(self.input, n_filter=32, filter_size=(3, 3), strides=(1, 1), act=activation, padding=(1,1), W_init=weights_init, b_init=bias_init, W_lr_and_decay=W_lr_and_decay, b_lr_and_decay=b_lr_and_decay, name="conv1")
#         self.pool1 = MaxPool2d(self.conv1, filter_size=(2, 2), strides=(2, 2), padding=(0,0), name='pool1')

#        self.conv2 = Conv2d(self.pool1, n_filter=32, filter_size=(3, 3), strides=(1, 1), act=activation, padding=(1,1), W_init=weights_init, b_init=bias_init, W_lr_and_decay=W_lr_and_decay, b_lr_and_decay=b_lr_and_decay, name="conv2")
#            self.pool2 = MaxPool2d(self.conv2, filter_size=(2, 2), strides=(2, 2), padding=(0,0), name='pool2')

            self.pool2 = self.netout
            self.flatten = FlattenLayer(self.pool2, name='flatten')
#            self.fc3 = DenseLayer(self.flatten, 32, act=None, name='fc3')
            # self.dp3 = DropoutLayer(self.fc3, keep=0.6, name='dropout3')
#            self.dp3 = DropoutLayer(self.fc3, keep=0.6, is_fix=True, is_train=is_training, seed=None, name='dropout3')

#            self.fc4 = DenseLayer(self.dp3, self.num_classes, act=None, name='fc4')
            predictions = slim.softmax(self.flatten.outputs)

        if is_training:
            # add backbone_params to gradient_ops
            # others add in self.ssd_multibox_layer()
            backbone_params = self.flatten.all_params
##            backbone_params = self.fc4.all_params
            print("------------------------")
            print("backbone_params:", backbone_params)
            for key in lr_mult.keys():
                if not lr_mult[key] ==0:
                    params = [x for x in backbone_params if key == x.name.split('/')[-1].split(':')[0]]

			
                    if lr_mult[key] not in self.gradient_ops:
#                    if not self.gradient_ops.has_key(lr_mult[key]):
                        self.gradient_ops[lr_mult[key]] = []
                    for param in params:
                        print(param, "->", key, lr_mult[key])
                        self.gradient_ops[lr_mult[key]].append(param)

#            return predictions, self.fc4.outputs
            return predictions, self.flatten.outputs
        else:
#            return predictions, self.fc4.outputs
            return predictions, self.flatten.outputs

if __name__ == '__main__':
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, [1, 227, 227, 3], name='data')
        
        m = Model()
        m.build(x, is_training=True)

        m.input.dump_all_layers_to_caffe_prototxt()
        print(str(m.input.get_caffe_model()))
