#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from itl.layers.core import Layer
from itl.layers.core import LayersConfig

from itl import logging

# from itl.decorators import deprecated_alias

__all__ = [
    'DenseLayer',
]


class DenseLayer(Layer):
    """The :class:`DenseLayer` class is a fully connected layer.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    n_units : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : a str
        A unique layer name.

    Examples
    --------
    With itl

    >>> net = tl.layers.InputLayer(x, name='input')
    >>> net = tl.layers.DenseLayer(net, 800, act=tf.nn.relu, name='relu')

    Without native itl APIs, you can do as follow.

    >>> W = tf.Variable(
    ...     tf.random_uniform([n_in, n_units], -1.0, 1.0), name='W')
    >>> b = tf.Variable(tf.zeros(shape=[n_units]), name='b')
    >>> y = tf.nn.relu(tf.matmul(inputs, W) + b)

    Notes
    -----
    If the layer input has more than two axes, it needs to be flatten by using :class:`FlattenLayer`.

    """

    # @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            n_units=100,
            act=None,
            W_init=tf.random_normal_initializer(stddev=0.1),
            b_init=tf.constant_initializer(value=0.0),
            W_lr_and_decay=(1, 1),
            b_lr_and_decay=(2, 0),
            W_init_args=None,
            b_init_args=None,
            name='dense',
    ):

        super(DenseLayer, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "DenseLayer  %s: %d %s" %
            (self.name, n_units, self.act.__name__ if self.act is not None else 'No Activation')
        )

        self.n_units = n_units

        self.W_init = W_init
        self.b_init = b_init
        self.W_lr_and_decay = W_lr_and_decay
        self.b_lr_and_decay = b_lr_and_decay
        self.W = None
        self.b = None

        if self.inputs.get_shape().ndims != 2:
            raise AssertionError("The input dimension must be rank 2, please reshape or flatten it")

        n_in = int(self.inputs.get_shape()[-1])

        with tf.variable_scope(name):
            self.W = tf.get_variable(
                name='W', shape=(n_in, n_units), initializer=W_init, dtype=LayersConfig.tf_dtype, **self.W_init_args
            )

            self.outputs = tf.matmul(self.inputs, self.W)

            if b_init is not None:
                try:
                    self.b = tf.get_variable(
                        name='b', shape=(n_units), initializer=b_init, dtype=LayersConfig.tf_dtype, **self.b_init_args
                    )
                except Exception:  # If initializer is a constant, do not specify shape.
                    self.b = tf.get_variable(name='b', initializer=b_init, dtype=LayersConfig.tf_dtype, **self.b_init_args)

                self.outputs = tf.nn.bias_add(self.outputs, self.b, name='bias_add')

            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)
        if b_init is not None:
            self._add_params([self.W, self.b])
        else:
            self._add_params(self.W)

    def to_caffe_prototxt(self):
        dense_layer = self.create_caffe_layer()
        
        dense_layer.name = self.name
        dense_layer.type = 'InnerProduct'
        self.append_bottom_from_inputs(dense_layer)
        self.append_top_from_outputs(dense_layer)

        # ParamSpec
        if self.W is not None:
            param = dense_layer.param.add()
            param.lr_mult = self.W_lr_and_decay[0]
            param.decay_mult = self.W_lr_and_decay[1]
            
        if self.b is not None:
            param = dense_layer.param.add()
            param.lr_mult = self.b_lr_and_decay[0]
            param.decay_mult = self.b_lr_and_decay[1]

        #inner_product_param 
        dense_layer.inner_product_param.num_output = self.n_units
        if isinstance(self.W_init, tf.random_normal_initializer):
            dense_layer.inner_product_param.weight_filler.type = 'gaussian'
            dense_layer.inner_product_param.weight_filler.mean = self.W_init.mean
            dense_layer.inner_product_param.weight_filler.std = self.W_init.stddev
        elif isinstance(self.W_init, tf.constant_initializer):
            dense_layer.inner_product_param.weight_filler.type = 'constant'
            dense_layer.inner_product_param.weight_filler.value = self.W_init.value

        if self.b is not None:
            if isinstance(self.b_init, tf.random_normal_initializer):
                dense_layer.inner_product_param.bias_filler.type = 'gaussian'
                dense_layer.inner_product_param.bias_filler.mean = self.b_init.mean
                dense_layer.inner_product_param.bias_filler.std = self.b_init.stddev
            elif isinstance(self.b_init, tf.constant_initializer):
                dense_layer.inner_product_param.bias_filler.type = 'constant'
                dense_layer.inner_product_param.bias_filler.value = self.b_init.value

        # blobs
        if self.W is not None:
            blob = dense_layer.blobs.add()
            name = [ord(c) for c in self.W.name]
            blob.data.extend(name)
            
        if self.b is not None:
            blob = dense_layer.blobs.add()
            name = [ord(c) for c in self.b.name]
            blob.data.extend(name)
             
        self.add_activation_layer()

