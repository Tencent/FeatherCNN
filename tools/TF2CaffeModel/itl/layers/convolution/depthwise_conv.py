#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from itl.layers.core import Layer
from itl.layers.core import LayersConfig
from itl.layers.core import l2_regularizer

from itl import logging

# from itl.decorators import deprecated_alias

__all__ = [
    'DepthwiseConv2d',
]


class DepthwiseConv2d(Layer):
    """Separable/Depthwise Convolutional 2D layer, see `tf.nn.depthwise_conv2d <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/depthwise_conv2d>`__.

    Input:
        4-D Tensor (batch, height, width, in_channels).
    Output:
        4-D Tensor (batch, new height, new width, in_channels * depth_multiplier).

    Parameters
    ------------
    prev_layer : :class:`Layer`
        Previous layer.
    filter_size : tuple of int
        The filter size (height, width).
    stride : tuple of int
        The stride step (height, width).
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    dilation_rate: tuple of 2 int
        The dilation rate in which we sample input values across the height and width dimensions in atrous convolution. If it is greater than 1, then all values of strides must be 1.
    depth_multiplier : int
        The number of channels to expand to.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip bias.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : str
        A unique layer name.

    Examples
    ---------
    >>> net = InputLayer(x, name='input')
    >>> net = Conv2d(net, 32, (3, 3), (2, 2), b_init=None, name='cin')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bnin')
    ...
    >>> net = DepthwiseConv2d(net, (3, 3), (1, 1), b_init=None, name='cdw1')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn11')
    >>> net = Conv2d(net, 64, (1, 1), (1, 1), b_init=None, name='c1')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn12')
    ...
    >>> net = DepthwiseConv2d(net, (3, 3), (2, 2), b_init=None, name='cdw2')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn21')
    >>> net = Conv2d(net, 128, (1, 1), (1, 1), b_init=None, name='c2')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn22')

    References
    -----------
    - tflearn's `grouped_conv_2d <https://github.com/tflearn/tflearn/blob/3e0c3298ff508394f3ef191bcd7d732eb8860b2e/tflearn/layers/conv.py>`__
    - keras's `separableconv2d <https://keras.io/layers/convolutional/#separableconv2d>`__

    """ # # https://zhuanlan.zhihu.com/p/31551004  https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/CNNs/MobileNet.py

    # @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            filter_size=(3, 3),
            strides=(1, 1),
            act=None,
            padding=(1, 1),
            dilation_rate=(1, 1),
            depth_multiplier=1,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_lr_and_decay=(1, 1),
            b_lr_and_decay=(2, 0),
            W_init_args=None,
            b_init_args=None,
            name='depthwise_conv2d',
    ):
        super(DepthwiseConv2d, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "DepthwiseConv2d %s: filter_size: %s strides: %s pad: %s act: %s" % (
                self.name, str(filter_size), str(strides), padding, self.act.__name__
                if self.act is not None else 'No Activation'
            )
        )

        self.strides = strides
        self.filter_size = filter_size
        self.padding = padding

        self.W_init = W_init
        self.b_init = b_init
        self.W_lr_and_decay = W_lr_and_decay
        self.b_lr_and_decay = b_lr_and_decay

        try:
            pre_channel = int(prev_layer.outputs.get_shape()[-1])
        except Exception:  # if pre_channel is ?, it happens when using Spatial Transformer Net
            pre_channel = 1
            logging.info("[warnings] unknown input channels, set to 1")

        self.shape = [filter_size[0], filter_size[1], pre_channel, depth_multiplier]

        if len(strides) == 2:
            strides = [1, strides[0], strides[1], 1]

        if len(strides) != 4:
            raise AssertionError("len(strides) should be 4.")

        self.W = None
        self.b = None

        with tf.variable_scope(name):

            self.W = tf.get_variable(
                name='depthwise_weights', shape=self.shape,
                regularizer=l2_regularizer(self.W_lr_and_decay[1]),
                initializer=W_init, dtype=LayersConfig.tf_dtype, **self.W_init_args
            )  # [filter_height, filter_width, in_channels, depth_multiplier]

            self.add_lr_to_gradient_operations(self.W_lr_and_decay[0], self.W)

            if padding == (0, 0):
                self.outputs = self.inputs
                padding_string = 'SAME'
            else:
                padding4d = [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]]
                self.outputs = tf.pad(self.inputs, paddings=padding4d)
                padding_string = 'VALID'

            self.outputs = tf.nn.depthwise_conv2d(self.outputs, self.W, strides=strides, padding=padding_string, rate=dilation_rate)

            if b_init:
                self.b = tf.get_variable(
                    name='depthwise_bias', shape=(pre_channel * depth_multiplier), initializer=b_init,
                    dtype=LayersConfig.tf_dtype, **self.b_init_args
                )

                self.add_lr_to_gradient_operations(self.b_lr_and_decay[0], self.b)

                self.outputs = tf.nn.bias_add(self.outputs, self.b, name='bias_add')

            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)

        if b_init:
            self._add_params([self.W, self.b])
        else:
            self._add_params(self.W)
    def get_layers_flops_a_size(self):
        Gflops = self.shape[-1]*self.shape[-2]*(self.inputs.shape[1]-self.shape[0]+self.padding[0]+1)/self.strides[0]*(self.inputs.shape[2]-self.shape[1]+self.padding[1]+1)/self.strides[1]*self.shape[0]*self.shape[1]
        Size = self.shape[0]*self.shape[1]*self.shape[2]*self.shape[3]
        return Gflops,Size

    def to_caffe_prototxt(self):
        depthwise_layer = self.create_caffe_layer()

        depthwise_layer.name = self.name
        depthwise_layer.type = 'Convolution'
        self.append_bottom_from_inputs(depthwise_layer)
        self.append_top_from_outputs(depthwise_layer)

        # ParamSpec
        if self.W is not None:
            param = depthwise_layer.param.add()
            param.lr_mult = self.W_lr_and_decay[0]
            param.decay_mult = self.W_lr_and_decay[1]

        if self.b is not None:
            param = depthwise_layer.param.add()
            param.lr_mult = self.b_lr_and_decay[0]
            param.decay_mult = self.b_lr_and_decay[1]

        # ConvolutionParameter
        depthwise_layer.convolution_param.num_output = self.shape[-2]
        depthwise_layer.convolution_param.bias_term = True if self.b is not None else False
        depthwise_layer.convolution_param.pad_h = self.padding[0]
        depthwise_layer.convolution_param.pad_w = self.padding[1]
        depthwise_layer.convolution_param.kernel_h = self.filter_size[0]
        depthwise_layer.convolution_param.kernel_w = self.filter_size[1]
        depthwise_layer.convolution_param.stride_h = self.strides[0]
        depthwise_layer.convolution_param.stride_w = self.strides[1]
        depthwise_layer.convolution_param.group = self.shape[-2]

        if isinstance(self.W_init, tf.random_normal_initializer):
            depthwise_layer.convolution_param.weight_filler.type = 'gaussian'
            depthwise_layer.convolution_param.weight_filler.mean = self.W_init.mean
            depthwise_layer.convolution_param.weight_filler.std = self.W_init.stddev
        elif isinstance(self.W_init, tf.constant_initializer):
            depthwise_layer.convolution_param.weight_filler.type = 'constant'
            depthwise_layer.convolution_param.weight_filler.value = self.W_init.value

        if self.b is not None:
            if isinstance(self.b_init, tf.random_normal_initializer):
                depthwise_layer.convolution_param.bias_filler.type = 'gaussian'
                depthwise_layer.convolution_param.bias_filler.mean = self.b_init.mean
                depthwise_layer.convolution_param.bias_filler.std = self.b_init.stddev
            elif isinstance(self.b_init, tf.constant_initializer):
                depthwise_layer.convolution_param.bias_filler.type = 'constant'
                depthwise_layer.convolution_param.bias_filler.value = self.b_init.value

        # blobs
        if self.W is not None:
            blob = depthwise_layer.blobs.add()

            name = [ord(c) for c in self.W.name]
            blob.data.extend(name)

        if self.b is not None:
            blob = depthwise_layer.blobs.add()

            name = [ord(c) for c in self.b.name]
            blob.data.extend(name)

        self.add_activation_layer()
