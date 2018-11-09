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
import random
import copy
import multiprocessing
import pdb

import numpy as np
import tensorflow as tf
import tensorlayer as tl

def conv_layer(inputs, filters, kernel, stride=1, layer_name="conv"):
    kernel_size = kernel[0]
    network = tl.layers.Conv2d(inputs, filters, kernel, strides=(stride,stride), act=None, name=layer_name)
    return network

def global_average_pooling(inputs):
    network = tl.layers.GlobalMeanPool2d(inputs)
    return network

def batch_normlization(inputs, layer_name="batchnorm"):
#    network = tl.layers.BatchNormLayer(inputs, decay=0.9, act=tf.nn.relu, is_train=self.is_training, name=layer_name)
    network = tl.layers.BatchNormLayer(inputs, decay=0.9, act=tf.nn.relu, is_train=True, name=layer_name)
    return network

def dropout(inputs, rate, layer_name="dropout"):
    return inputs

def relu(inputs):
    return tf.nn.relu(inputs)

def average_pooling(inputs, pool_size=[2, 2], strides=2, padding='VALID', layer_name="pool"):
    network = tl.layers.MeanPool2d(inputs, filter_size=pool_size, strides=strides, padding=padding, name=layer_name)
    return network

def concat(inputs, layer_name="concat"):
    network = tl.layers.ConcatLayer(inputs, 3, name=layer_name)
    return network

def linear(inputs, units):
    return tf.layers.dense(inputs=inputs, units=units, name="linear")

def bottleneck_layer(x, block_name="dense"):
    name = block_name
    x = batch_normlization(x, layer_name=name+"bn0")
#    x = self.conv_layer(x, filters=4*self.growth_rate, kernel=[1, 1], layer_name=name+"depthwise")
    x = conv_layer(x, filters=4*48, kernel=[1, 1], layer_name=name+"depthwise")
#    x = dropout(x, rate=self.dropout_rate, layer_name=name+"dropout0")
    x = dropout(x, rate=0.2, layer_name=name+"dropout0")

    x = batch_normlization(x, layer_name=name+"bn1")
#    x = self.conv_layer(x, filters=self.growth_rate, kernel=[3, 3], layer_name=name+"conv")
    x = conv_layer(x, filters=48, kernel=[3, 3], layer_name=name+"conv")
#    x = self.dropout(x, rate=dropout_rate, layer_name=name+"dropout1")
    x = dropout(x, rate=0.2, layer_name=name+"dropout1")
        
    return x
    
def transition_layer(x, block_name="tran"):
    name = block_name
    x = batch_normlization(x, layer_name=name+"bn0")
#    filter_size = x.outputs.shape[self.channel_index]
    filter_size = x.outputs.shape[3]
    transition_filter_size = int (int(filter_size) / 2)
#   tf.Print(transition_filter_size, [transition_filter_size], message='Debug message:',summarize=100)

    x = conv_layer(x, filters=transition_filter_size, kernel=[3, 3], layer_name=name+"conv")
#    x = self.dropout(x, rate=self.dropout_rate, layer_name=name+"dropout1")
    x = dropout(x, rate=0.2, layer_name=name+"dropout1")
    x = average_pooling(x, pool_size=[2, 2], strides=2, padding='SAME', layer_name=name+"pool")
    return x

def dense_block(input_x, nb_layers, block_name="dense"):
    layers_concat = list()
    layers_concat.append(input_x)
    x = bottleneck_layer(input_x, block_name+"_"+str(0))
    layers_concat.append(x)

    for i in range(nb_layers - 1):
        x = concat(layers_concat)
        x = bottleneck_layer(x, block_name+"_"+str(i+1))
        layers_concat.append(x)
        
    x = concat(layers_concat)
    return x

def max_pooling(inputs, pool_size=[3,3], stride=2, padding='VALID', layer_name="max_pool"):
    network = tl.layers.MaxPool2d(inputs, pool_size, strides=(stride, stride), padding=padding, name=layer_name)
#   network = tl.layers.MaxPool2d(inputs, pool_size, strides=(stride, stride), padding=(0,0), name=layer_name)
    return network

def add_inference(input_x):
    with tf.variable_scope("simplenet", reuse=None):
        x = tl.layers.InputLayer(input_x)
        x = conv_layer(x, filters=2*24, kernel=[5, 5], stride=2, layer_name="conv0")
        ## add max pooling
        x = max_pooling(x, padding = 'SAME')
        
        x = dense_block(input_x=x, nb_layers=6, block_name="dense0")
        x = transition_layer(x, block_name="tran0")

#        x = batch_normlization(x,"bn_out0")
        x = dense_block(input_x=x, nb_layers=6, block_name="dense1")
        x = transition_layer(x, block_name="tran1")

        x = dense_block(input_x=x, nb_layers=6, block_name="dense2")

        x = batch_normlization(x, "bn_out1")
#        x = dropout(x, rate=self.dropout_rate, layer_name="dropout1")
        x = dropout(x, rate=0.2, layer_name="dropout1")
#        x = global_average_pooling(x)
        x = tl.layers.FlattenLayer(x)


#        x = FlattenLayer(x)
#        x = tl.layers.DenseLayer(x, self.num_classes, act=None) 
#        x = self.linear(x.outputs, self.num_classes)

        return x.outputs

def _add_inference(images):
    if self.data_format == "NCHW":
        images = tf.transpose(images, [0, 3, 1, 2])
    logits = self.add_inference(images)
    return logits
