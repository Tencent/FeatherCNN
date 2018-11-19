#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from itl.layers.core import Layer
from itl.layers.core import LayersConfig
from itl.layers.core import l2_regularizer
from itl.layers.utils import get_collection_trainable

from itl import logging

# from itl.decorators import deprecated_alias

__all__ = [
    'Conv1d',
    'Conv2d',
]

class Conv1d(Layer):
    """Simplified version of :class:`Conv1dLayer`.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer
    n_filter : int
        The number of filters
    filter_size : int
        The filter size
    stride : int
        The stride step
    dilation_rate : int
        Specifying the dilation rate to use for dilated convolution.
    act : activation function
        The function that is applied to the layer activations
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        Default is 'NWC' as it is a 1D CNN.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer (deprecated).
    b_init_args : dictionary
        The arguments for the bias vector initializer (deprecated).
    name : str
        A unique layer name

    Examples
    ---------
    >>> x = tf.placeholder(tf.float32, (batch_size, width))
    >>> y_ = tf.placeholder(tf.int64, shape=(batch_size,))
    >>> n = InputLayer(x, name='in')
    >>> n = ReshapeLayer(n, (-1, width, 1), name='rs')
    >>> n = Conv1d(n, 64, 3, 1, act=tf.nn.relu, name='c1')
    >>> n = MaxPool1d(n, 2, 2, padding='valid', name='m1')
    >>> n = Conv1d(n, 128, 3, 1, act=tf.nn.relu, name='c2')
    >>> n = MaxPool1d(n, 2, 2, padding='valid', name='m2')
    >>> n = Conv1d(n, 128, 3, 1, act=tf.nn.relu, name='c3')
    >>> n = MaxPool1d(n, 2, 2, padding='valid', name='m3')
    >>> n = FlattenLayer(n, name='f')
    >>> n = DenseLayer(n, 500, tf.nn.relu, name='d1')
    >>> n = DenseLayer(n, 100, tf.nn.relu, name='d2')
    >>> n = DenseLayer(n, 2, None, name='o')

    """

    # @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, n_filter=32, filter_size=5, stride=1, dilation_rate=1, act=None, padding='SAME',
            data_format="channels_last", W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0), W_init_args=None, b_init_args=None, name='conv1d'
    ):
        super(Conv1d, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "Conv1d %s: n_filter: %d filter_size: %s stride: %d pad: %s act: %s dilation_rate: %d" % (
                self.name, n_filter, filter_size, stride, padding, self.act.__name__
                if self.act is not None else 'No Activation', dilation_rate
            )
        )

        _conv1d = tf.layers.Conv1D(
            filters=n_filter, kernel_size=filter_size, strides=stride, padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, activation=self.act, use_bias=(True if b_init else False),
            kernel_initializer=W_init, bias_initializer=b_init, name=name
        )

        # _conv1d.dtype = LayersConfig.tf_dtype   # unsupport, it will use the same dtype of inputs
        self.outputs = _conv1d(self.inputs)
        # new_variables = _conv1d.weights  # new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
        # new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=self.name)  #vs.name)
        new_variables = get_collection_trainable(self.name)

        self._add_layers(self.outputs)
        self._add_params(new_variables)


class Conv2d(Layer):
    """Simplified version of :class:`Conv2dLayer`.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    strides : tuple of int
        The sliding window strides of corresponding input dimensions (height, width).
    act : activation function
        The activation function of this layer.
    padding : tuple of int
        The padding value (height, width).
    W_init : initializer
        The initializer for the the weight matrix.
    b_init : initializer or None
        The initializer for the the bias vector. If None, skip biases.
    W_lr_and_decay: tuple of float
        The lr_mult and decay_mult for the weight matrix.
    b_lr_and_decay: tuple of float
        The lr_mult and decay_mult for the bias matrix.
    W_init_args : dictionary
        The arguments for the weight matrix initializer (for TF < 1.5).
    b_init_args : dictionary
        The arguments for the bias vector initializer (for TF < 1.5).
    use_cudnn_on_gpu : bool
        Default is False (for TF < 1.5).
    data_format : str
        "NHWC" or "NCHW", default is "NHWC" (for TF < 1.5).
    name : str
        A unique layer name.

    Returns
    -------
    :class:`Layer`
        A :class:`Conv2dLayer` object.

    Examples
    --------
    >>> x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    >>> net = InputLayer(x, name='inputs')
    >>> net = Conv2d(net, 64, (3, 3), act=tf.nn.relu, name='conv1_1')
    >>> net = Conv2d(net, 64, (3, 3), act=tf.nn.relu, name='conv1_2')
    >>> net = MaxPool2d(net, (2, 2), name='pool1')
    >>> net = Conv2d(net, 128, (3, 3), act=tf.nn.relu, name='conv2_1')
    >>> net = Conv2d(net, 128, (3, 3), act=tf.nn.relu, name='conv2_2')
    >>> net = MaxPool2d(net, (2, 2), name='pool2')

    """

    # @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            n_filter=32,
            filter_size=(3, 3),
            strides=(1, 1),
            act=None,
            padding=(0, 0),
            dilation_rate=(1, 1),
            W_init=tf.random_normal_initializer(stddev=0.01),
            b_init=tf.constant_initializer(value=0.0),
            W_lr_and_decay=(1, 1),
            b_lr_and_decay=(2, 0),
            W_init_args=None,
            b_init_args=None,
            use_cudnn_on_gpu=None,
            data_format=None,
            name='conv2d',
    ):
        super(Conv2d, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "Conv2d %s: n_filter: %d filter_size: %s strides: %s pad: %s act: %s" % (
                self.name, n_filter, str(filter_size), str(strides), str(padding), self.act.__name__
                if self.act is not None else 'No Activation'
            )
        )

        self.padding = padding
        self.strides = strides

        self.W_init = W_init
        self.b_init = b_init
        self.W_lr_and_decay = W_lr_and_decay
        self.b_lr_and_decay = b_lr_and_decay

        self.W = None
        self.b = None

        if len(strides) != 2:
            raise ValueError("len(strides) should be 2, Conv2d and Conv2dLayer are different.")

        try:
            pre_channel = int(prev_layer.outputs.get_shape()[-1])

        except Exception:  # if pre_channel is ?, it happens when using Spatial Transformer Net
            pre_channel = 1
            logging.info("[warnings] unknow input channels, set to 1")

        self.shape = [filter_size[0], filter_size[1], pre_channel, n_filter]

        with tf.variable_scope(name):

            self.W = tf.get_variable(
                name='weights', shape=self.shape, initializer=W_init,
#                name='kernel', shape=self.shape, initializer=W_init,
                regularizer=l2_regularizer(self.W_lr_and_decay[1]),
                dtype=LayersConfig.tf_dtype, **self.W_init_args
            )

            tf.summary.histogram(self.W.name.replace(':', '_'), self.W)
            self.add_lr_to_gradient_operations(self.W_lr_and_decay[0], self.W)

            # add caffe-style's zero padding
            padding4d = [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]]
            strides4d = [1, strides[0], strides[1], 1]

            if padding == (0, 0):
                self.outputs = self.inputs
                padding_string = 'SAME'
            else:
                self.outputs = tf.pad(self.inputs, paddings=padding4d)
                padding_string = 'VALID'

            self.outputs = tf.nn.conv2d(
                self.outputs, self.W,
                strides=strides4d, padding=padding_string,
                use_cudnn_on_gpu=use_cudnn_on_gpu,
                data_format=data_format
            )

            if b_init:
                self.b = tf.get_variable(
                    name='bias', shape=(self.shape[-1]), initializer=b_init,
                    dtype=LayersConfig.tf_dtype,
                    **self.b_init_args
                )

                tf.summary.histogram(self.b.name.replace(':', '_'), self.b)
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
        conv2d_layer = self.create_caffe_layer()

        conv2d_layer.name = self.name
        conv2d_layer.type = 'Convolution'
        self.append_bottom_from_inputs(conv2d_layer)
        self.append_top_from_outputs(conv2d_layer)

        # ParamSpec
        if self.W is not None:
            param = conv2d_layer.param.add()
            param.lr_mult = self.W_lr_and_decay[0]
            param.decay_mult = self.W_lr_and_decay[1]

        if self.b is not None:
            param = conv2d_layer.param.add()
            param.lr_mult = self.b_lr_and_decay[0]
            param.decay_mult = self.b_lr_and_decay[1]

        # ConvolotionParameter
        conv2d_layer.convolution_param.num_output = self.shape[-1]
        conv2d_layer.convolution_param.bias_term = True if self.b is not None else False
        conv2d_layer.convolution_param.pad_h = self.padding[0]
        conv2d_layer.convolution_param.pad_w = self.padding[1]
        conv2d_layer.convolution_param.kernel_h = self.shape[0]
        conv2d_layer.convolution_param.kernel_w = self.shape[1]
        conv2d_layer.convolution_param.stride_h = self.strides[0]
        conv2d_layer.convolution_param.stride_w = self.strides[1]

        if isinstance(self.W_init, tf.random_normal_initializer):
            conv2d_layer.convolution_param.weight_filler.type = 'gaussian'
            conv2d_layer.convolution_param.weight_filler.mean = self.W_init.mean
            conv2d_layer.convolution_param.weight_filler.std = self.W_init.stddev
        elif isinstance(self.W_init, tf.constant_initializer):
            conv2d_layer.convolution_param.weight_filler.type = 'constant'
            conv2d_layer.convolution_param.weight_filler.value = self.W_init.value

        if self.b is not None:
            if isinstance(self.b_init, tf.random_normal_initializer):
                conv2d_layer.convolution_param.bias_filler.type = 'gaussian'
                conv2d_layer.convolution_param.bias_filler.mean = self.b_init.mean
                conv2d_layer.convolution_param.bias_filler.std = self.b_init.stddev
            elif isinstance(self.b_init, tf.constant_initializer):
                conv2d_layer.convolution_param.bias_filler.type = 'constant'
                conv2d_layer.convolution_param.bias_filler.value = self.b_init.value

        # blobs
        if self.W is not None:
            blob = conv2d_layer.blobs.add()
            name = [ord(c) for c in self.W.name]
            blob.data.extend(name)

        if self.b is not None:
            blob = conv2d_layer.blobs.add()
            name = [ord(c) for c in self.b.name]
            blob.data.extend(name)

        self.add_activation_layer()
