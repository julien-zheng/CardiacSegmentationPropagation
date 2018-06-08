from __future__ import division

import six
import h5py
import numpy as np
import cv2
import math
import scipy.spatial

from scipy import (
    interpolate,
    ndimage
)

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    UpSampling2D
)
from keras.layers.convolutional import (
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import (
    add,
    Add,
    Concatenate
)
from keras.layers.normalization import BatchNormalization
from keras.layers.core import (
    Lambda,
    Reshape
)
from keras.layers.advanced_activations import LeakyReLU 
from keras.regularizers import l2
from keras.callbacks import Callback
from keras import backend as K

from image2 import array_to_img
from medpy.metric.binary import (
    hd,
    dc
)

import tensorflow as tf


try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

'''
def bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)
'''

def scalar_multiplication(**params):
    """A layer to multiply a tensor by a scalar
    """
    name = params.setdefault("name", "")
    scalar = params.setdefault("scalar", 1.0)
    def f(input):
        return Lambda(lambda x: tf.scalar_mul(scalar, x))(input)
    return f

def tensor_slice(dimension, start, end):
    def f(input):
        if dimension == 0:
            return input[start:end]
        if dimension == 1:
            return input[:, start:end]
        if dimension == 2:
            return input[:, :, start:end]
        if dimension == 3:
            return input[:, :, :, start:end]
    return Lambda(f)
    

def bn_relu(**params):
    """Helper to build a BN -> relu block
    """
    name = params.setdefault("name", "")
    name_bn = name + "_bn"
    name_relu = name + "_relu"
    def f(input):
        norm = BatchNormalization(axis=CHANNEL_AXIS,
                                  momentum=0.99,
                                  epsilon=1e-3,
                                  center=True,
                                  scale=True,
                                  beta_initializer='zeros',
                                  gamma_initializer='ones',
                                  moving_mean_initializer='zeros',
                                  moving_variance_initializer='ones',
                                  beta_regularizer=None,
                                  gamma_regularizer=None,
                                  beta_constraint=None,
                                  gamma_constraint=None,
                                  name=name_bn)(input)
        return Activation("relu", name=name_relu)(norm)

    return f


def bn_leakyrelu(**params):
    """Helper to build a BN -> leaky relu block
    """
    name = params.setdefault("name", "")
    alpha = params.setdefault("alpha", 0.1)
    name_bn = name + "_bn"
    name_relu = name + "_relu"
    def f(input):
        norm = BatchNormalization(axis=CHANNEL_AXIS,
                                  momentum=0.99,
                                  epsilon=1e-3,
                                  center=True,
                                  scale=True,
                                  beta_initializer='zeros',
                                  gamma_initializer='ones',
                                  moving_mean_initializer='zeros',
                                  moving_variance_initializer='ones',
                                  beta_regularizer=None,
                                  gamma_regularizer=None,
                                  beta_constraint=None,
                                  gamma_constraint=None,
                                  name=name_bn)(input)
        return LeakyReLU(alpha=alpha, name=name_relu)(norm)

    return f


def conv_relu(**conv_params):
    """Helper to build a conv -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    #kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)
    name = conv_params.setdefault("name", "conv")
    name_relu = name + "_relu"

    def f(input):
        conv = Conv2D(filters=filters, 
                      kernel_size=kernel_size,
                      strides=strides, 
                      padding=padding,
                      data_format=None,
                      dilation_rate=(1, 1),
                      activation=None,
                      use_bias=True,
                      kernel_initializer=kernel_initializer,
                      bias_initializer='zeros',
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=None,
                      activity_regularizer=None,
                      kernel_constraint=None, 
                      bias_constraint=None,
                      name=name)(input)
        return Activation("relu", name=name_relu)(conv)

    return f



def conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    #kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)
    name = conv_params.setdefault("name", "conv")

    def f(input):
        conv = Conv2D(filters=filters, 
                      kernel_size=kernel_size,
                      strides=strides, 
                      padding=padding,
                      data_format=None,
                      dilation_rate=(1, 1),
                      activation=None,
                      use_bias=True,
                      kernel_initializer=kernel_initializer,
                      bias_initializer='zeros',
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=None,
                      activity_regularizer=None,
                      kernel_constraint=None, 
                      bias_constraint=None,
                      name=name)(input)
        return bn_relu(name=name)(conv)

    return f


def conv_bn_leakyrelu(**conv_params):
    """Helper to build a conv -> BN -> leaky relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    #kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)
    alpha = conv_params.setdefault("alpha", 0.1)
    name = conv_params.setdefault("name", "conv")

    def f(input):
        conv = Conv2D(filters=filters, 
                      kernel_size=kernel_size,
                      strides=strides, 
                      padding=padding,
                      data_format=None,
                      dilation_rate=(1, 1),
                      activation=None,
                      use_bias=True,
                      kernel_initializer=kernel_initializer,
                      bias_initializer='zeros',
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=None,
                      activity_regularizer=None,
                      kernel_constraint=None, 
                      bias_constraint=None,
                      name=name)(input)
        return bn_relu(alpha=alpha, name=name)(conv)

    return f


def deconv_relu(**deconv_params):
    """Helper to build a deconv -> relu block
    """
    filters = deconv_params["filters"]
    kernel_size = deconv_params.setdefault("kernel_size", (2, 2))
    strides = deconv_params.setdefault("strides", (2, 2))
    kernel_initializer = deconv_params.setdefault("kernel_initializer", "he_normal")
    padding = deconv_params.setdefault("padding", "valid")
    #kernel_regularizer = deconv_params.setdefault("kernel_regularizer", l2(1.e-4))
    kernel_regularizer = deconv_params.setdefault("kernel_regularizer", None)
    name = deconv_params.setdefault("name", "deconv")
    name_relu = name + "_relu"

    def f(input):
        deconv = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name)(input)
        return Activation("relu", name=name_relu)(deconv)

    return f



def deconv_bn_relu(**deconv_params):
    """Helper to build a deconv -> BN -> relu block
    """
    filters = deconv_params["filters"]
    kernel_size = deconv_params.setdefault("kernel_size", (2, 2))
    strides = deconv_params.setdefault("strides", (2, 2))
    kernel_initializer = deconv_params.setdefault("kernel_initializer", "he_normal")
    padding = deconv_params.setdefault("padding", "valid")
    #kernel_regularizer = deconv_params.setdefault("kernel_regularizer", l2(1.e-4))
    kernel_regularizer = deconv_params.setdefault("kernel_regularizer", None)
    name = deconv_params.setdefault("name", "deconv")

    def f(input):
        deconv = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name)(input)
        return bn_relu(name=name)(deconv)

    return f


def bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in "Identity Mappings in Deep Residual         
    Networks" (http://arxiv.org/pdf/1603.05027v2.pdf)
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)

    def f(input):
        activation = bn_relu()(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)(activation)

    return f



def shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
            kernel_size=(1, 1),
            strides=(stride_width, stride_height),
            padding="valid",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in "Identity Mappings in Deep Residual         
    Networks" (http://arxiv.org/pdf/1603.05027v2.pdf)
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                strides=init_strides,
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = bn_relu_conv(filters=filters, kernel_size=(3, 3),
                strides=init_strides)(input)

        residual = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in "Identity Mappings in Deep Residual         
    Networks" (http://arxiv.org/pdf/1603.05027v2.pdf)
    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                strides=init_strides,
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = bn_relu_conv(filters=filters, kernel_size=(1, 1),
                strides=init_strides)(input)

        conv_3_3 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return shortcut(input, residual)

    return f

def conv_conv_relu(**conv_params):
    """Helper to build a conv-conv-relu network block
    """
    filters = conv_params["filters"]
    kernel_size1 = conv_params["kernel_size1"]
    kernel_size2 = conv_params["kernel_size2"]
    strides1 = conv_params.setdefault("strides1", (1, 1))
    strides2 = conv_params.setdefault("strides2", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)
    name = conv_params.setdefault("name", "ccr")

    def f(input):
        conv1 = Conv2D(filters=filters, kernel_size=kernel_size1,
            strides=strides1, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv1")(input)
        conv2 = Conv2D(filters=filters, kernel_size=kernel_size2,
            strides=strides2, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv2")(conv1)

        return Activation("relu", name=name+"_relu")(conv2)

    return f

def conv_conv_bn_relu(**conv_params):
    """Helper to build a conv-conv-bn-relu network block
    """
    filters = conv_params["filters"]
    kernel_size1 = conv_params["kernel_size1"]
    kernel_size2 = conv_params["kernel_size2"]
    strides1 = conv_params.setdefault("strides1", (1, 1))
    strides2 = conv_params.setdefault("strides2", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)
    name = conv_params.setdefault("name", "ccbr")

    def f(input):
        conv1 = Conv2D(filters=filters, kernel_size=kernel_size1,
            strides=strides1, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv1")(input)
        conv2 = Conv2D(filters=filters, kernel_size=kernel_size2,
            strides=strides2, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv2")(conv1)

        return bn_relu(name=name)(conv2)

    return f


def deconv_deconv_relu(**deconv_params):
    """Helper to build a deconv-deconv-relu block
    """
    filters = deconv_params["filters"]
    kernel_size1 = deconv_params["kernel_size1"]
    kernel_size2 = deconv_params["kernel_size2"]
    strides1 = deconv_params["strides1"]
    strides2 = deconv_params["strides2"]
    kernel_initializer = deconv_params.setdefault("kernel_initializer", "he_normal")
    padding = deconv_params.setdefault("padding", "same")
    #kernel_regularizer = deconv_params.setdefault("kernel_regularizer", l2(1.e-4))
    kernel_regularizer = deconv_params.setdefault("kernel_regularizer", None)
    name = deconv_params.setdefault("name", "ddbr")

    def f(input):
        deconv1 = Conv2DTranspose(filters=filters, kernel_size=kernel_size1,
            strides=strides1, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_deconv1")(input)
        deconv2 = Conv2DTranspose(filters=filters, kernel_size=kernel_size2,
            strides=strides2, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_deconv2")(deconv1)
        return Activation("relu", name=name+"_relu")(deconv2)

    return f


def deconv_deconv_bn_relu(**deconv_params):
    """Helper to build a deconv-deconv-bn-relu block
    """
    filters = deconv_params["filters"]
    kernel_size1 = deconv_params["kernel_size1"]
    kernel_size2 = deconv_params["kernel_size2"]
    strides1 = deconv_params["strides1"]
    strides2 = deconv_params["strides2"]
    kernel_initializer = deconv_params.setdefault("kernel_initializer", "he_normal")
    padding = deconv_params.setdefault("padding", "same")
    #kernel_regularizer = deconv_params.setdefault("kernel_regularizer", l2(1.e-4))
    kernel_regularizer = deconv_params.setdefault("kernel_regularizer", None)
    name = deconv_params.setdefault("name", "ddbr")

    def f(input):
        deconv1 = Conv2DTranspose(filters=filters, kernel_size=kernel_size1,
            strides=strides1, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_deconv1")(input)
        deconv2 = Conv2DTranspose(filters=filters, kernel_size=kernel_size2,
            strides=strides2, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_deconv2")(deconv1)
        return bn_relu(name=name)(deconv2)

    return f






def gcn_block(**conv_params):
    """Helper to build a global convolutional network block
    which is proposed in "Large Kernel Matters -- Improve Semantic Segmentation by Global 
    Convolutional Network" (https://arxiv.org/pdf/1703.02719.pdf)
    """
    filters = conv_params["filters"]
    kernel_size1 = conv_params["kernel_size1"]
    kernel_size2 = conv_params["kernel_size2"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)
    name = conv_params.setdefault("name", "gcn")

    def f(input):
        conv1_1 = Conv2D(filters=filters, kernel_size=kernel_size1,
            strides=strides, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv1_1")(input)
        conv1_2 = Conv2D(filters=filters, kernel_size=kernel_size2,
            strides=strides, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv1_2")(conv1_1)

        conv2_1 = Conv2D(filters=filters, kernel_size=kernel_size2,
            strides=strides, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv2_1")(input)
        conv2_2 = Conv2D(filters=filters, kernel_size=kernel_size1,
            strides=strides, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv2_2")(conv2_1)

        return add([conv1_2, conv2_2], name=name+"_add")

    return f


def boundary_refinement_block(**conv_params):
    """Helper to build a boundary refinement block
    which is proposed in "Large Kernel Matters -- Improve Semantic Segmentation by Global 
    Convolutional Network" (https://arxiv.org/pdf/1703.02719.pdf)
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", None)
    name = conv_params.setdefault("name", "br")

    def f(input):
        conv1 = Conv2D(filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv1")(input)
        relu = Activation("relu", name=name+"_relu")(conv1)
        conv2 = Conv2D(filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name+"_conv2")(relu)

        return add([input, conv2], name=name+"_add")

    return f

def conv_relu_repetition_block(filters, kernel_size, repetitions,                 
        first_layer_down_size=False, name="conv_block"):
    """Builds a block with repeating convolution + relu blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and first_layer_down_size:
                init_strides = (2, 2)
            if isinstance(kernel_size, list):
                kernel_size_i = kernel_size[i]
            else:
                kernel_size_i = kernel_size
            input = conv_relu(filters=filters, kernel_size=kernel_size_i, 
                strides=init_strides, name=name+"_conv"+str(i))(input)
        return input

    return f


def conv_bn_relu_repetition_block(filters, kernel_size, repetitions,                 
        first_layer_down_size=False, name="conv_block"):
    """Builds a block with repeating convolution + batch_normalization + relu blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and first_layer_down_size:
                init_strides = (2, 2)
            if isinstance(kernel_size, list):
                kernel_size_i = kernel_size[i]
            else:
                kernel_size_i = kernel_size

            input = conv_bn_relu(filters=filters, kernel_size=kernel_size_i, 
                strides=init_strides, name=name+"_conv"+str(i))(input)
        return input

    return f


def conv_bn_leakyrelu_repetition_block(filters, kernel_size, repetitions,                 
        first_layer_down_size=False, alpha=0.1, name="conv_block"):
    """Builds a block with repeating convolution + batch_normalization + relu blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and first_layer_down_size:
                init_strides = (2, 2)
            if isinstance(kernel_size, list):
                kernel_size_i = kernel_size[i]
            else:
                kernel_size_i = kernel_size

            input = conv_bn_leakyrelu(filters=filters, kernel_size=kernel_size_i, 
                strides=init_strides, alpha=alpha, name=name+"_conv"+str(i))(input)
        return input

    return f


def conv_bn_leakyrelu_res_repetition_block(filters, kernel_size, repetitions,                 
        first_layer_down_size=False, alpha=0.1, name="conv_block"):
    """Builds a block with repeating convolution + batch_normalization + relu blocks.
    """
    def f(input):
        init_strides = (1, 1)
        if first_layer_down_size:
            init_strides = (2, 2)
        if isinstance(kernel_size, list):
            kernel_size_0 = kernel_size[0]
        else:
            kernel_size_0 = kernel_size

        input = conv_bn_leakyrelu(filters=filters, kernel_size=kernel_size_0, 
            strides=init_strides, alpha=alpha, name=name+"_conv"+str(0))(input)

        for i in range(1, repetitions):
            init_strides = (1, 1)
            if isinstance(kernel_size, list):
                kernel_size_i = kernel_size[i]
            else:
                kernel_size_i = kernel_size

            if i == 1:
                input1 = conv_bn_leakyrelu(filters=filters, kernel_size=kernel_size_i, 
                    strides=init_strides, alpha=alpha, name=name+"_conv"+str(i))(input)
            else:
                input1 = conv_bn_leakyrelu(filters=filters, kernel_size=kernel_size_i, 
                    strides=init_strides, alpha=alpha, name=name+"_conv"+str(i))(input1)

        if repetitions > 1:
            return add([input, input1], name=name+"_add")
        else:
            return input

    return f


def conv_relu_repetition_residual_block(filters, kernel_size, repetitions,                 
        first_layer_down_size=False, name="conv_block"):
    """Builds a block with repeating convolution + relu blocks and
    a residual connection.
    """
    def f(input):
        init_strides = (1, 1)
        if first_layer_down_size:
            init_strides = (2, 2)
        input = conv_relu(filters=filters, kernel_size=kernel_size, 
            strides=init_strides, name=name+"_conv"+str(0))(input)
        if repetitions == 2:
            input1 = conv_relu(filters=filters, kernel_size=kernel_size, 
                strides=(1,1), name=name+"_conv"+str(1))(input)
            return add([input, input1], name=name+"_add")

        if repetitions == 3:
            input1 = conv_relu(filters=filters, kernel_size=kernel_size, 
                strides=(1,1), name=name+"_conv"+str(1))(input)
            input2 = conv_relu(filters=filters, kernel_size=kernel_size, 
                strides=(1,1), name=name+"_conv"+str(2))(input1)
            return add([input, input2], name=name+"_add")

    return f


def conv_bn_relu_repetition_residual_block(filters, kernel_size, repetitions,                 
        first_layer_down_size=False, name="conv_block"):
    """Builds a block with repeating convolution + batch_normalization + relu blocks and
    a residual connection.
    """
    def f(input):
        init_strides = (1, 1)
        if first_layer_down_size:
            init_strides = (2, 2)
        input = conv_bn_relu(filters=filters, kernel_size=kernel_size, 
            strides=init_strides, name=name+"_conv"+str(0))(input)
        if repetitions == 2:
            input1 = conv_bn_relu(filters=filters, kernel_size=kernel_size, 
                strides=(1,1), name=name+"_conv"+str(1))(input)
            return add([input, input1], name=name+"_add")

        if repetitions == 3:
            input1 = conv_bn_relu(filters=filters, kernel_size=kernel_size, 
                strides=(1,1), name=name+"_conv"+str(1))(input)
            input2 = conv_bn_relu(filters=filters, kernel_size=kernel_size, 
                strides=(1,1), name=name+"_conv"+str(2))(input1)
            return add([input, input2], name=name+"_add")

    return f


def deconv_conv_relu_repetition_block(filters, kernel_size, repetitions,
        name="deconv_block"):
    """Builds a block with first deconvolution + relu, then 
    concatenate with input2, and finally repeating convolution + batch_normalization + relu 
    blocks.
    """
    def f(input, input2):
        input = deconv_relu(filters=filters, name=name+"_deconv")(input)
        input = Concatenate(axis=CHANNEL_AXIS, name=name+"_concate")([input, input2])
        for i in range(repetitions):
            input = conv_relu(filters=filters, kernel_size=kernel_size, 
                name=name+"_conv"+str(i))(input)
        return input

    return f


def deconv_conv_bn_relu_repetition_block(filters, kernel_size, repetitions,
        name="deconv_block"):
    """Builds a block with first deconvolution + batch_normalization + relu, then 
    concatenate with input2, and finally repeating convolution + batch_normalization + relu 
    blocks.
    """
    def f(input, input2):
        input = deconv_bn_relu(filters=filters, name=name+"_deconv")(input)
        input = Concatenate(axis=CHANNEL_AXIS, name=name+"_concate")([input, input2])
        for i in range(repetitions):
            input = conv_bn_relu(filters=filters, kernel_size=kernel_size, 
                name=name+"_conv"+str(i))(input)
        return input

    return f


def up_conv_relu_repetition_block(filters, kernel_size, repetitions,
        name="up_conv_block"):
    """Builds a block with first upsampling, then 
    concatenate with input2, and finally repeating convolution + relu 
    blocks.
    """
    def f(*args):
        input = args[0]
        input = UpSampling2D(size=(2, 2), name=name+"_up")(input)
        if len(args) > 1:
            input2 = args[1]
            input = Concatenate(axis=CHANNEL_AXIS, name=name+"_concate")([input, input2])
        for i in range(repetitions):
            if isinstance(kernel_size, list):
                kernel_size_i = kernel_size[i]
            else:
                kernel_size_i = kernel_size
            input = conv_relu(filters=filters, kernel_size=kernel_size_i,
                name=name+"_conv"+str(i))(input)
        return input

    return f

def up_conv_bn_relu_repetition_block(filters, kernel_size, repetitions,
        name="up_conv_block"):
    """Builds a block with first upsampling, then 
    concatenate with input2, and finally repeating convolution + batch_normalization + relu 
    blocks.
    """
    def f(*args):
        input = args[0]
        input = UpSampling2D(size=(2, 2), name=name+"_up")(input)
        if len(args) > 1:
            input2 = args[1]
            input = Concatenate(axis=CHANNEL_AXIS, name=name+"_concate")([input, input2])
        for i in range(repetitions):
            if isinstance(kernel_size, list):
                kernel_size_i = kernel_size[i]
            else:
                kernel_size_i = kernel_size
            input = conv_bn_relu(filters=filters, kernel_size=kernel_size_i,
                name=name+"_conv"+str(i))(input)
        return input

    return f


def up_conv_relu_repetition_block2(filters, kernel_size, repetitions, up_filters_multiple=2,
        name="up_conv_block"):
    """Builds a block with first upsampling, then convolution + relu,
    concatenate with input2, and finally repeating convolution + relu 
    blocks.
    """
    def f(input, input2):
        input = UpSampling2D(size=(2, 2), name=name+"_up")(input)
        input = conv_relu(filters=up_filters_multiple*filters, kernel_size=kernel_size,
            name=name+"_0th_conv")(input)
        input = Concatenate(axis=CHANNEL_AXIS, name=name+"_concate")([input, input2])
        for i in range(repetitions):
            input = conv_relu(filters=filters, kernel_size=kernel_size,
                name=name+"_conv"+str(i))(input)
        return input

    return f


def up_conv_bn_relu_repetition_block2(filters, kernel_size, repetitions, up_filters_multiple=2,
        name="up_conv_block"):
    """Builds a block with first upsampling, then convolution + batch_normalization + relu,
    concatenate with input2, and finally repeating convolution + batch_normalization + relu 
    blocks.
    """
    def f(input, input2):
        input = UpSampling2D(size=(2, 2), name=name+"_up")(input)
        input = conv_bn_relu(filters=up_filters_multiple*filters, kernel_size=kernel_size,
            name=name+"_0th_conv")(input)
        input = Concatenate(axis=CHANNEL_AXIS, name=name+"_concate")([input, input2])
        for i in range(repetitions):
            input = conv_bn_relu(filters=filters, kernel_size=kernel_size,
                name=name+"_conv"+str(i))(input)
        return input

    return f

def up_conv_relu_repetition_residual_block2(filters, kernel_size, repetitions, up_filters_multiple=1,
        name="up_conv_block"):
    """Builds a block with first upsampling, then convolution + batch_normalization + relu,
    concatenate with input2, and then repeating convolution + batch_normalization + relu 
    blocks, and finally a residual connection.
    """
    def f(input, input2):
        input = UpSampling2D(size=(2, 2), name=name+"_up")(input)
        input = conv_relu(filters=up_filters_multiple*filters, kernel_size=kernel_size,
            name=name+"_0th_conv")(input)
        result = Concatenate(axis=CHANNEL_AXIS, name=name+"_concate")([input, input2])
        for i in range(repetitions):
            result = conv_relu(filters=filters, kernel_size=kernel_size,
                name=name+"_conv"+str(i))(result)
        return add([input, result], name=name+"_add")

    return f

def up_conv_bn_relu_repetition_residual_block2(filters, kernel_size, repetitions, up_filters_multiple=1,
        name="up_conv_block"):
    """Builds a block with first upsampling, then convolution + batch_normalization + relu,
    concatenate with input2, and then repeating convolution + batch_normalization + relu 
    blocks, and finally a residual connection.
    """
    def f(input, input2):
        input = UpSampling2D(size=(2, 2), name=name+"_up")(input)
        input = conv_bn_relu(filters=up_filters_multiple*filters, kernel_size=kernel_size,
            name=name+"_0th_conv")(input)
        result = Concatenate(axis=CHANNEL_AXIS, name=name+"_concate")([input, input2])
        for i in range(repetitions):
            result = conv_bn_relu(filters=filters, kernel_size=kernel_size,
                name=name+"_conv"+str(i))(result)
        return add([input, result], name=name+"_add")

    return f


def print_hdf5_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print("      {}: {}".format(p_name, param.shape))
    finally:
        f.close()


def dice_coef(y_true, y_pred, smooth=0.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred, smooth=0.0):
    return -dice_coef(y_true, y_pred, smooth)


def dice_coef2(y_true, y_pred, smooth=1.0):
    #y_true_f = K.flatten(y_true)
    y_true = tf.where(y_true > 0.5, K.ones_like(y_true), K.zeros_like(y_true))
    #y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    sum = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean((2. * intersection + smooth) / (sum + smooth), axis=0)


def dice_coef2_loss(y_true, y_pred, smooth=1.0):
    return -dice_coef2(y_true, y_pred, smooth)

def jaccard_coef2(y_true, y_pred, smooth=0.0):
    #y_true_f = K.flatten(y_true)
    y_true = tf.where(y_true > 0.5, K.ones_like(y_true), K.zeros_like(y_true))
    #y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    sum = K.sum(y_true * y_true, axis=[1,2,3]) + K.sum(y_pred * y_pred, axis=[1,2,3])
    return K.mean((1.0 * intersection + smooth) / (sum - intersection + smooth), axis=0)


def jaccard_coef2_loss(y_true, y_pred, smooth=0.0):
    return -jaccard_coef2(y_true, y_pred, smooth)


def dice_coef3(y_true, y_pred, smooth=0.0):
    #y_true_f = K.flatten(y_true)
    y_true0 = tf.where(K.equal(y_true, 0.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true1 = tf.where(K.equal(y_true, 1.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true2 = tf.where(K.equal(y_true, 2.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true3 = tf.where(K.equal(y_true, 3.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    

    y_pred0 = tf.slice(y_pred, [0,0,0,0], [-1,-1,-1,1])
    #y_pred_f0 = K.flatten(y_pred0)
    y_pred1 = tf.slice(y_pred, [0,0,0,1], [-1,-1,-1,1])
    #y_pred_f1 = K.flatten(y_pred1)
    y_pred2 = tf.slice(y_pred, [0,0,0,2], [-1,-1,-1,1])
    #y_pred_f2 = K.flatten(y_pred2)
    y_pred3 = tf.slice(y_pred, [0,0,0,3], [-1,-1,-1,1])
    #y_pred_f3 = K.flatten(y_pred3)

    #intersection0 = K.sum(y_true_f0 * y_pred_f0)
    intersection0 = K.sum(y_true0 * y_pred0, axis=[1,2,3])
    sum0 = K.sum(y_true0, axis=[1,2,3]) + K.sum(y_pred0, axis=[1,2,3])
    res0 = K.mean((2. * intersection0 + smooth) / (sum0 + smooth), axis=0)

    #intersection1 = K.sum(y_true_f1 * y_pred_f1)
    intersection1 = K.sum(y_true1 * y_pred1, axis=[1,2,3])
    sum1 = K.sum(y_true1, axis=[1,2,3]) + K.sum(y_pred1, axis=[1,2,3])
    res1 = K.mean((2. * intersection1 + smooth) / (sum1+ smooth), axis=0)

    #intersection2 = K.sum(y_true_f2 * y_pred_f2)
    intersection2 = K.sum(y_true2 * y_pred2, axis=[1,2,3])
    sum2 = K.sum(y_true2, axis=[1,2,3]) + K.sum(y_pred2, axis=[1,2,3])
    res2 = K.mean((2. * intersection2 + smooth) / (sum2 + smooth), axis=0)

    #intersection3 = K.sum(y_true_f3 * y_pred_f3)
    intersection3 = K.sum(y_true3 * y_pred3, axis=[1,2,3])
    sum3 = K.sum(y_true3, axis=[1,2,3]) + K.sum(y_pred3, axis=[1,2,3])
    res3 = K.mean((2. * intersection3 + smooth) / (sum3 + smooth), axis=0)

    return (res0 + res1 + res2 + res3) / 4.0
    #return (res1 + res2 + res3) / 3.0


def dice_coef3_loss(y_true, y_pred, smooth=0.0):
    return -dice_coef3(y_true, y_pred, smooth)


def dice_coef3_0(y_true, y_pred, smooth=1e-10):
    #y_true_f = K.flatten(y_true)
    y_true0 = tf.where(K.equal(y_true, 0.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))

    y_pred = K.argmax(y_pred, axis=-1)
    shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (shape[0], shape[1], shape[2], 1))
    y_pred = tf.to_float(y_pred)
    y_pred0 = tf.where(K.equal(y_pred, 0.0 * K.ones_like(y_pred)), 
                       K.ones_like(y_pred), K.zeros_like(y_pred))
    #y_pred_f0 = K.flatten(y_pred0)

    intersection0 = K.sum(y_true0 * y_pred0, axis=[1,2,3])
    sum0 = K.sum(y_true0, axis=[1,2,3]) + K.sum(y_pred0, axis=[1,2,3])
    res0 = K.mean((2. * intersection0 + smooth) / (sum0 + smooth), axis=0)

    return res0


def dice_coef3_1(y_true, y_pred, smooth=1e-10):
    #y_true_f = K.flatten(y_true)
    y_true1 = tf.where(K.equal(y_true, 1.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    
    y_pred = K.argmax(y_pred, axis=-1)
    shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (shape[0], shape[1], shape[2], 1))
    y_pred = tf.to_float(y_pred)
    y_pred1 = tf.where(K.equal(y_pred, 1.0 * K.ones_like(y_pred)), 
                       K.ones_like(y_pred), K.zeros_like(y_pred))
    #y_pred_f1 = K.flatten(y_pred1)

    intersection1 = K.sum(y_true1 * y_pred1, axis=[1,2,3])
    sum1 = K.sum(y_true1, axis=[1,2,3]) + K.sum(y_pred1, axis=[1,2,3])
    res1 = K.mean((2. * intersection1 + smooth) / (sum1+ smooth), axis=0)

    return res1


def dice_coef3_2(y_true, y_pred, smooth=1e-10):
    #y_true_f = K.flatten(y_true)
    y_true2 = tf.where(K.equal(y_true, 2.0 * K.ones_like(y_true)),
                         K.ones_like(y_true), K.zeros_like(y_true))

    y_pred = K.argmax(y_pred, axis=-1)
    shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (shape[0], shape[1], shape[2], 1))
    y_pred = tf.to_float(y_pred)
    y_pred2 = tf.where(K.equal(y_pred, 2.0 * K.ones_like(y_pred)), 
                       K.ones_like(y_pred), K.zeros_like(y_pred))
    #y_pred_f2 = K.flatten(y_pred2)

    intersection2 = K.sum(y_true2 * y_pred2, axis=[1,2,3])
    sum2 = K.sum(y_true2, axis=[1,2,3]) + K.sum(y_pred2, axis=[1,2,3])
    res2 = K.mean((2. * intersection2 + smooth) / (sum2 + smooth), axis=0)

    return res2

def dice_coef3_3(y_true, y_pred, smooth=1e-10):
    #y_true_f = K.flatten(y_true)
    y_pred = tf.to_float(y_pred)
    y_true3 = tf.where(K.equal(y_true, 3.0 * K.ones_like(y_true)),
                         K.ones_like(y_true), K.zeros_like(y_true))

    y_pred = K.argmax(y_pred, axis=-1)
    shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (shape[0], shape[1], shape[2], 1))
    y_pred = tf.to_float(y_pred)
    y_pred3 = tf.where(K.equal(y_pred, 3.0 * K.ones_like(y_pred)), 
                       K.ones_like(y_pred), K.zeros_like(y_pred))
    #y_pred_f3 = K.flatten(y_pred3)

    intersection3 = K.sum(y_true3 * y_pred3, axis=[1,2,3])
    sum3 = K.sum(y_true3, axis=[1,2,3]) + K.sum(y_pred3, axis=[1,2,3])
    res3 = K.mean((2. * intersection3 + smooth) / (sum3 + smooth), axis=0)

    return res3


def dice_coef4(y_true, y_pred, smooth=0.0):
    #y_true_f = K.flatten(y_true)
    y_true0 = tf.where(K.equal(y_true, 0.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true1 = tf.where(K.equal(y_true, 1.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true2 = tf.where(K.equal(y_true, 2.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true3 = tf.where(K.equal(y_true, 3.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    

    y_pred0 = tf.slice(y_pred, [0,0,0,0], [-1,-1,-1,1])
    #y_pred_f0 = K.flatten(y_pred0)
    y_pred1 = tf.slice(y_pred, [0,0,0,1], [-1,-1,-1,1])
    #y_pred_f1 = K.flatten(y_pred1)
    y_pred2 = tf.slice(y_pred, [0,0,0,2], [-1,-1,-1,1])
    #y_pred_f2 = K.flatten(y_pred2)
    y_pred3 = tf.slice(y_pred, [0,0,0,3], [-1,-1,-1,1])
    #y_pred_f3 = K.flatten(y_pred3)

    #intersection0 = K.sum(y_true_f0 * y_pred_f0)
    intersection0 = K.sum(y_true0 * y_pred0, axis=[1,2,3])
    sum0 = K.sum(y_true0 * y_true0, axis=[1,2,3]) + K.sum(y_pred0 * y_pred0, axis=[1,2,3])
    res0 = K.mean((2. * intersection0 + smooth) / (sum0 + smooth), axis=0)

    #intersection1 = K.sum(y_true_f1 * y_pred_f1)
    intersection1 = K.sum(y_true1 * y_pred1, axis=[1,2,3])
    sum1 = K.sum(y_true1 * y_true1, axis=[1,2,3]) + K.sum(y_pred1 * y_pred1, axis=[1,2,3])
    res1 = K.mean((2. * intersection1 + smooth) / (sum1+ smooth), axis=0)

    #intersection2 = K.sum(y_true_f2 * y_pred_f2)
    intersection2 = K.sum(y_true2 * y_pred2, axis=[1,2,3])
    sum2 = K.sum(y_true2 * y_true2, axis=[1,2,3]) + K.sum(y_pred2 * y_pred2, axis=[1,2,3])
    res2 = K.mean((2. * intersection2 + smooth) / (sum2 + smooth), axis=0)

    #intersection3 = K.sum(y_true_f3 * y_pred_f3)
    intersection3 = K.sum(y_true3 * y_pred3, axis=[1,2,3])
    sum3 = K.sum(y_true3 * y_true3, axis=[1,2,3]) + K.sum(y_pred3 * y_pred3, axis=[1,2,3])
    res3 = K.mean((2. * intersection3 + smooth) / (sum3 + smooth), axis=0)

    return (res0 + res1 + res2 + res3) / 4.0
    #return (res1 + res2 + res3) / 3.0


def dice_coef4_loss(y_true, y_pred, smooth=0.0):
    return -dice_coef4(y_true, y_pred, smooth)


def dice_coef5(y_true, y_pred, smooth=1.0):
    #y_true_f = K.flatten(y_true)
    y_true0 = tf.where(K.equal(y_true, 0.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true1 = tf.where(K.equal(y_true, 1.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true2 = tf.where(K.equal(y_true, 2.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true3 = tf.where(K.equal(y_true, 3.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    

    y_pred0 = tf.slice(y_pred, [0,0,0,0], [-1,-1,-1,1])
    #y_pred_f0 = K.flatten(y_pred0)
    y_pred1 = tf.slice(y_pred, [0,0,0,1], [-1,-1,-1,1])
    #y_pred_f1 = K.flatten(y_pred1)
    y_pred2 = tf.slice(y_pred, [0,0,0,2], [-1,-1,-1,1])
    #y_pred_f2 = K.flatten(y_pred2)
    y_pred3 = tf.slice(y_pred, [0,0,0,3], [-1,-1,-1,1])
    #y_pred_f3 = K.flatten(y_pred3)

    #intersection0 = K.sum(y_true_f0 * y_pred_f0)
    intersection0 = K.sum(y_true0 * y_pred0, axis=[1,2,3])
    sum0 = K.sum(y_true0, axis=[1,2,3]) + K.sum(y_pred0, axis=[1,2,3])
    res0 = K.mean((2. * intersection0 + smooth) / (sum0 + smooth), axis=0)

    #intersection1 = K.sum(y_true_f1 * y_pred_f1)
    intersection1 = K.sum(y_true1 * y_pred1, axis=[1,2,3])
    sum1 = K.sum(y_true1, axis=[1,2,3]) + K.sum(y_pred1, axis=[1,2,3])
    res1 = K.mean((2. * intersection1 + smooth) / (sum1+ smooth), axis=0)

    #intersection2 = K.sum(y_true_f2 * y_pred_f2)
    intersection2 = K.sum(y_true2 * y_pred2, axis=[1,2,3])
    sum2 = K.sum(y_true2, axis=[1,2,3]) + K.sum(y_pred2, axis=[1,2,3])
    res2 = K.mean((2. * intersection2 + smooth) / (sum2 + smooth), axis=0)

    #intersection3 = K.sum(y_true_f3 * y_pred_f3)
    intersection3 = K.sum(y_true3 * y_pred3, axis=[1,2,3])
    sum3 = K.sum(y_true3, axis=[1,2,3]) + K.sum(y_pred3, axis=[1,2,3])
    res3 = K.mean((2. * intersection3 + smooth) / (sum3 + smooth), axis=0)

    return (res0 + res1 + res2 + res3) / 4.0
    #return (res1 + res2 + res3) / 3.0


def dice_coef5_loss(y_true, y_pred, smooth=1.0):
    return -dice_coef5(y_true, y_pred, smooth)


def dice_coef5_0(y_true, y_pred, smooth=1.0):
    #y_true_f = K.flatten(y_true)
    y_true0 = tf.where(K.equal(y_true, 0.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))

    y_pred = K.argmax(y_pred, axis=-1)
    shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (shape[0], shape[1], shape[2], 1))
    y_pred = tf.to_float(y_pred)
    y_pred0 = tf.where(K.equal(y_pred, 0.0 * K.ones_like(y_pred)), 
                       K.ones_like(y_pred), K.zeros_like(y_pred))
    #y_pred_f0 = K.flatten(y_pred0)

    intersection0 = K.sum(y_true0 * y_pred0, axis=[1,2,3])
    sum0 = K.sum(y_true0, axis=[1,2,3]) + K.sum(y_pred0, axis=[1,2,3])
    res0 = K.mean((2. * intersection0 + smooth) / (sum0 + smooth), axis=0)

    return res0


def dice_coef5_1(y_true, y_pred, smooth=1.0):
    #y_true_f = K.flatten(y_true)
    y_true1 = tf.where(K.equal(y_true, 1.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    
    y_pred = K.argmax(y_pred, axis=-1)
    shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (shape[0], shape[1], shape[2], 1))
    y_pred = tf.to_float(y_pred)
    y_pred1 = tf.where(K.equal(y_pred, 1.0 * K.ones_like(y_pred)), 
                       K.ones_like(y_pred), K.zeros_like(y_pred))
    #y_pred_f1 = K.flatten(y_pred1)

    intersection1 = K.sum(y_true1 * y_pred1, axis=[1,2,3])
    sum1 = K.sum(y_true1, axis=[1,2,3]) + K.sum(y_pred1, axis=[1,2,3])
    res1 = K.mean((2. * intersection1 + smooth) / (sum1+ smooth), axis=0)

    return res1


def dice_coef5_2(y_true, y_pred, smooth=1.0):
    #y_true_f = K.flatten(y_true)
    y_true2 = tf.where(K.equal(y_true, 2.0 * K.ones_like(y_true)),
                         K.ones_like(y_true), K.zeros_like(y_true))

    y_pred = K.argmax(y_pred, axis=-1)
    shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (shape[0], shape[1], shape[2], 1))
    y_pred = tf.to_float(y_pred)
    y_pred2 = tf.where(K.equal(y_pred, 2.0 * K.ones_like(y_pred)), 
                       K.ones_like(y_pred), K.zeros_like(y_pred))
    #y_pred_f2 = K.flatten(y_pred2)

    intersection2 = K.sum(y_true2 * y_pred2, axis=[1,2,3])
    sum2 = K.sum(y_true2, axis=[1,2,3]) + K.sum(y_pred2, axis=[1,2,3])
    res2 = K.mean((2. * intersection2 + smooth) / (sum2 + smooth), axis=0)

    return res2

def dice_coef5_3(y_true, y_pred, smooth=1.0):
    #y_true_f = K.flatten(y_true)
    y_pred = tf.to_float(y_pred)
    y_true3 = tf.where(K.equal(y_true, 3.0 * K.ones_like(y_true)),
                         K.ones_like(y_true), K.zeros_like(y_true))

    y_pred = K.argmax(y_pred, axis=-1)
    shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (shape[0], shape[1], shape[2], 1))
    y_pred = tf.to_float(y_pred)
    y_pred3 = tf.where(K.equal(y_pred, 3.0 * K.ones_like(y_pred)), 
                       K.ones_like(y_pred), K.zeros_like(y_pred))
    #y_pred_f3 = K.flatten(y_pred3)

    intersection3 = K.sum(y_true3 * y_pred3, axis=[1,2,3])
    sum3 = K.sum(y_true3, axis=[1,2,3]) + K.sum(y_pred3, axis=[1,2,3])
    res3 = K.mean((2. * intersection3 + smooth) / (sum3 + smooth), axis=0)

    return res3



def dice_coef6(y_true, y_pred, smooth=1.0):
    #y_true_f = K.flatten(y_true)
    y_true0 = tf.where(K.equal(y_true, 0.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true1 = tf.where(K.equal(y_true, 1.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true2 = tf.where(K.equal(y_true, 2.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    

    y_pred0 = tf.slice(y_pred, [0,0,0,0], [-1,-1,-1,1])
    #y_pred_f0 = K.flatten(y_pred0)
    y_pred1 = tf.slice(y_pred, [0,0,0,1], [-1,-1,-1,1])
    #y_pred_f1 = K.flatten(y_pred1)
    y_pred2 = tf.slice(y_pred, [0,0,0,2], [-1,-1,-1,1])
    #y_pred_f2 = K.flatten(y_pred2)

    #intersection0 = K.sum(y_true_f0 * y_pred_f0)
    intersection0 = K.sum(y_true0 * y_pred0, axis=[1,2,3])
    sum0 = K.sum(y_true0, axis=[1,2,3]) + K.sum(y_pred0, axis=[1,2,3])
    res0 = K.mean((2. * intersection0 + smooth) / (sum0 + smooth), axis=0)

    #intersection1 = K.sum(y_true_f1 * y_pred_f1)
    intersection1 = K.sum(y_true1 * y_pred1, axis=[1,2,3])
    sum1 = K.sum(y_true1, axis=[1,2,3]) + K.sum(y_pred1, axis=[1,2,3])
    res1 = K.mean((2. * intersection1 + smooth) / (sum1+ smooth), axis=0)

    #intersection2 = K.sum(y_true_f2 * y_pred_f2)
    intersection2 = K.sum(y_true2 * y_pred2, axis=[1,2,3])
    sum2 = K.sum(y_true2, axis=[1,2,3]) + K.sum(y_pred2, axis=[1,2,3])
    res2 = K.mean((2. * intersection2 + smooth) / (sum2 + smooth), axis=0)

    return (res0 + res1 + res2) / 3.0


def dice_coef6_loss(y_true, y_pred, smooth=1.0):
    return -dice_coef6(y_true, y_pred, smooth)




def jaccard_coef3_1(y_true, y_pred, smooth=0.0):
    '''
    y_true0 = tf.where(K.equal(y_true, 0.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true1 = tf.where(K.equal(y_true, 1.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true2 = tf.where(K.equal(y_true, 2.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))
    y_true3 = tf.where(K.equal(y_true, 3.0 * K.ones_like(y_true)), 
                         K.ones_like(y_true), K.zeros_like(y_true))

    shape = tf.shape(y_true)
    y_true0 = K.reshape(y_true0, (shape[0], shape[1], shape[2]))
    y_true1 = K.reshape(y_true1, (shape[0], shape[1], shape[2]))
    y_true2 = K.reshape(y_true2, (shape[0], shape[1], shape[2]))
    y_true3 = K.reshape(y_true3, (shape[0], shape[1], shape[2]))

    y_pred0 = tf.slice(y_pred, [0,0,0,0], [-1,-1,-1,1])
    y_pred1 = tf.slice(y_pred, [0,0,0,1], [-1,-1,-1,1])
    y_pred2 = tf.slice(y_pred, [0,0,0,2], [-1,-1,-1,1])
    y_pred3 = tf.slice(y_pred, [0,0,0,3], [-1,-1,-1,1])

    y_pred0 = K.reshape(y_pred0, (shape[0], shape[1], shape[2]))
    y_pred1 = K.reshape(y_pred1, (shape[0], shape[1], shape[2]))
    y_pred2 = K.reshape(y_pred2, (shape[0], shape[1], shape[2]))
    y_pred3 = K.reshape(y_pred3, (shape[0], shape[1], shape[2]))

    intersection0 = tf.norm(y_true0 * y_pred0, axis=[1,2])
    sum0 =  K.sum(y_true0 * y_true0, axis=[1,2]) + K.sum(y_pred0 * y_pred0, axis=[1,2])
    res0 = (1.0 * intersection0 + smooth) / (sum0 - intersection0 + smooth)

    intersection1 = tf.norm(y_true1 * y_pred1, axis=[1,2])
    sum1 =  K.sum(y_true1 * y_true1, axis=[1,2]) + K.sum(y_pred1 * y_pred1, axis=[1,2])
    res1 = (1.0 * intersection1 + smooth) / (sum1 - intersection1 + smooth)

    intersection2 = tf.norm(y_true2 * y_pred2, axis=[1,2])
    sum2 =  K.sum(y_true2 * y_true2, axis=[1,2]) + K.sum(y_pred2 * y_pred2, axis=[1,2])
    res2 = (1.0 * intersection2 + smooth) / (sum2 - intersection2 + smooth)

    intersection3 = tf.norm(y_true3 * y_pred3, axis=[1,2])
    sum3 =  K.sum(y_true3 * y_true3, axis=[1,2]) + K.sum(y_pred3 * y_pred3, axis=[1,2])
    res3 = (1.0 * intersection3 + smooth) / (sum3 - intersection3 + smooth)
    '''


    #y_true_f = K.flatten(y_true)
    #y_true0 = tf.where(K.equal(y_true, 0.0 * K.ones_like(y_true)), 
    #             K.ones_like(y_true), K.zeros_like(y_true))
    y_true1 = tf.where(K.equal(y_true, 1.0 * K.ones_like(y_true)), 
                     K.ones_like(y_true), K.zeros_like(y_true))
    y_true2 = tf.where(K.equal(y_true, 2.0 * K.ones_like(y_true)), 
                     K.ones_like(y_true), K.zeros_like(y_true))
    y_true3 = tf.where(K.equal(y_true, 3.0 * K.ones_like(y_true)), 
                     K.ones_like(y_true), K.zeros_like(y_true))
    

    #y_pred0 = tf.slice(y_pred, [0,0,0,0], [-1,-1,-1,1])
    #y_pred_f0 = K.flatten(y_pred0)
    y_pred1 = tf.slice(y_pred, [0,0,0,1], [-1,-1,-1,1])
    #y_pred_f1 = K.flatten(y_pred1)
    y_pred2 = tf.slice(y_pred, [0,0,0,2], [-1,-1,-1,1])
    #y_pred_f2 = K.flatten(y_pred2)
    y_pred3 = tf.slice(y_pred, [0,0,0,3], [-1,-1,-1,1])
    #y_pred_f3 = K.flatten(y_pred3)

    #intersection0 = K.sum(y_true_f0 * y_pred_f0)
    #intersection0 = K.sum(y_true0 * y_pred0, axis=[1,2,3])
    #sum0 = K.sum(y_true0 * y_true0, axis=[1,2,3]) + K.sum(y_pred0 * y_pred0, axis=[1,2,3])
    #res0 = K.mean((1.0 * intersection0 + smooth) / (sum0 - intersection0 + smooth), axis=0)

    #intersection1 = K.sum(y_true_f1 * y_pred_f1)
    intersection1 = K.sum(y_true1 * y_pred1, axis=[1,2,3])
    sum1 = K.sum(y_true1 * y_true1, axis=[1,2,3]) + K.sum(y_pred1 * y_pred1, axis=[1,2,3])
    res1 = K.mean((1.0 * intersection1 + smooth) / (sum1 - intersection1 + smooth), axis=0)

    #intersection2 = K.sum(y_true_f2 * y_pred_f2)
    intersection2 = K.sum(y_true2 * y_pred2, axis=[1,2,3])
    sum2 = K.sum(y_true2 * y_true2, axis=[1,2,3]) + K.sum(y_pred2 * y_pred2, axis=[1,2,3])
    res2 = K.mean((1.0 * intersection2 + smooth) / (sum2 - intersection2 + smooth), axis=0)

    #intersection3 = K.sum(y_true_f3 * y_pred_f3)
    intersection3 = K.sum(y_true3 * y_pred3, axis=[1,2,3])
    sum3 = K.sum(y_true3 * y_true3, axis=[1,2,3]) + K.sum(y_pred3 * y_pred3, axis=[1,2,3])
    res3 = K.mean((1.0 * intersection3 + smooth) / (sum3 - intersection3 + smooth), axis=0)

 
    #return (res0 + res1 + res2 + res3) / 4.0
    return (res1 + res2 + res3) / 3.0



def jaccard_coef3_2(y_true, y_pred, smooth=0.0):
    
    y_true0 = K.ones_like(y_true)
    y_pred0 = tf.slice(y_pred, [0,0,0,0], [-1,-1,-1,1])

    intersection0 = K.sum(y_true0 * y_pred0, axis=[1,2,3])
    sum0 = K.sum(y_true0 * y_true0, axis=[1,2,3]) + K.sum(y_pred0 * y_pred0, axis=[1,2,3])
    res0 = K.mean((1.0 * intersection0 + smooth) / (sum0 - intersection0 + smooth), axis=0)

    return res0


def jaccard_coef3(y_true, y_pred, smooth=0.0):

    return tf.cond(tf.reduce_max(y_true) > 0., 
                   lambda: jaccard_coef3_1(y_true, y_pred, smooth),
                   lambda: jaccard_coef3_2(y_true, y_pred, smooth))


def jaccard_coef3_loss(y_true, y_pred, smooth=0.0):
    return -jaccard_coef3(y_true, y_pred, smooth)


def jaccard_coef4(y_true, y_pred, smooth=0.0):

    #y_true_f = K.flatten(y_true)
    y_true0 = tf.where(K.equal(y_true, 0.0 * K.ones_like(y_true)), 
                     K.ones_like(y_true), K.zeros_like(y_true))
    y_true1 = tf.where(K.equal(y_true, 1.0 * K.ones_like(y_true)), 
                     K.ones_like(y_true), K.zeros_like(y_true))
    y_true2 = tf.where(K.equal(y_true, 2.0 * K.ones_like(y_true)), 
                     K.ones_like(y_true), K.zeros_like(y_true))
    y_true3 = tf.where(K.equal(y_true, 3.0 * K.ones_like(y_true)), 
                     K.ones_like(y_true), K.zeros_like(y_true))
    

    y_pred0 = tf.slice(y_pred, [0,0,0,0], [-1,-1,-1,1])
    #y_pred_f0 = K.flatten(y_pred0)
    y_pred1 = tf.slice(y_pred, [0,0,0,1], [-1,-1,-1,1])
    #y_pred_f1 = K.flatten(y_pred1)
    y_pred2 = tf.slice(y_pred, [0,0,0,2], [-1,-1,-1,1])
    #y_pred_f2 = K.flatten(y_pred2)
    y_pred3 = tf.slice(y_pred, [0,0,0,3], [-1,-1,-1,1])
    #y_pred_f3 = K.flatten(y_pred3)

    #intersection0 = K.sum(y_true_f0 * y_pred_f0)
    intersection0 = K.sum(y_true0 * y_pred0, axis=[1,2,3])
    sum0 = K.sum(y_true0 * y_true0, axis=[1,2,3]) + K.sum(y_pred0 * y_pred0, axis=[1,2,3])
    res0 = K.mean((1.0 * intersection0 + smooth) / (sum0 - intersection0 + smooth), axis=0)

    #intersection1 = K.sum(y_true_f1 * y_pred_f1)
    intersection1 = K.sum(y_true1 * y_pred1, axis=[1,2,3])
    sum1 = K.sum(y_true1 * y_true1, axis=[1,2,3]) + K.sum(y_pred1 * y_pred1, axis=[1,2,3])
    res1 = K.mean((1.0 * intersection1 + smooth) / (sum1 - intersection1 + smooth), axis=0)

    #intersection2 = K.sum(y_true_f2 * y_pred_f2)
    intersection2 = K.sum(y_true2 * y_pred2, axis=[1,2,3])
    sum2 = K.sum(y_true2 * y_true2, axis=[1,2,3]) + K.sum(y_pred2 * y_pred2, axis=[1,2,3])
    res2 = K.mean((1.0 * intersection2 + smooth) / (sum2 - intersection2 + smooth), axis=0)

    #intersection3 = K.sum(y_true_f3 * y_pred_f3)
    intersection3 = K.sum(y_true3 * y_pred3, axis=[1,2,3])
    sum3 = K.sum(y_true3 * y_true3, axis=[1,2,3]) + K.sum(y_pred3 * y_pred3, axis=[1,2,3])
    res3 = K.mean((1.0 * intersection3 + smooth) / (sum3 - intersection3 + smooth), axis=0)

 
    return (res0 + res1 + res2 + res3) / 4.0
    #return (res1 + res2 + res3) / 3.0


def jaccard_coef4_loss(y_true, y_pred, smooth=0.0):
    return -jaccard_coef4(y_true, y_pred, smooth)

def base_slice_euclidean_distance_loss(y_true, y_pred):
    y_pred_reduced = tf.reduce_max(y_pred, axis=[1,2,3])
    
    max_values = tf.reduce_max(y_true, axis=[1,2,3])
    
    labels = tf.where(K.equal(max_values, 3.0 * K.ones_like(max_values)), 
        K.ones_like(max_values), K.zeros_like(max_values))
    
    return tf.reduce_sum(tf.square(tf.subtract(y_pred_reduced, labels)))


def depth_softmax(matrix):
    sigmoid = lambda x: 1.0 / (1 + K.exp(-x))
    sigmoided_matrix = sigmoid(matrix)
    softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=2)
    return softmax_matrix


def mean_variance_normalization(array):
    mean = np.mean(array)
    std = np.std(array)
    adjusted_std = max(std, 1.0/np.sqrt(array.size))
    return (array - mean)/adjusted_std

def mean_variance_normalization2(array):

    percentile1 = np.percentile(array, 10)
    percentile2 = np.percentile(array, 90)
    array2 = array[np.logical_and(array > percentile1, array < percentile2)]
    mean = np.mean(array2)
    std = np.std(array2)
    '''
    percentile1 = np.percentile(array, 5)
    percentile2 = np.percentile(array, 95)
    array[array <= percentile1] = percentile1
    array[array >= percentile2] = percentile2
    mean = np.mean(array)
    std = np.std(array)
    '''
    adjusted_std = max(std, 1.0/np.sqrt(array.size))
    return 1.0 * (array - mean)/adjusted_std

def mean_variance_normalization3(array):
    percentile1 = np.percentile(array, 5)
    percentile2 = np.percentile(array, 95)
    array[array <= percentile1] = percentile1
    array[array >= percentile2] = percentile2

    shape = array.shape
    array = np.reshape(array, (shape[0], shape[1]))
    array = array.astype('uint16')
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    array = clahe.apply(array)
    array = array.astype(float)
    array = np.reshape(array, shape)

    mean = np.mean(array)
    std = np.std(array)
    adjusted_std = max(std, 1.0/np.sqrt(array.size))
    return 1.0 * (array - mean)/adjusted_std

def mean_variance_normalization4(array):

    array2 = array[array > array.min()]
    if (array2.size > 0):
        mean = np.mean(array2)
        std = np.std(array2)
        adjusted_std = max(std, 1.0/np.sqrt(array.size))
        return 1.0 * (array - mean)/adjusted_std
    else:
        return (array - array.min())


def mean_variance_normalization5(array):

    percentile1 = np.percentile(array, 5)
    percentile2 = np.percentile(array, 95)

    array2 = array[np.logical_and(array > percentile1, array < percentile2)]
    if (array2.size > 0):
        mean = np.mean(array2)
        std = np.std(array2)
        adjusted_std = max(std, 1.0/np.sqrt(array.size))
        return 1.0 * (array - mean)/adjusted_std
    else:
        return (array - array.min())

def elementwise_multiplication(array):
    return (0.02 * array)

def elementwise_multiplication2(array):
    array2 = 0.02 * array
    array2[array2 == 3.0] = 0.0
    return array2

def one_hot(indices, num_classes):
    res = []
    for i in range(num_classes):
        res += [tf.where(K.equal(indices, i * K.ones_like(indices)), 
                         K.ones_like(indices), K.zeros_like(indices))]
    return K.concatenate(res, axis=-1)
    

def mask_to_contour(mask):
    results = cv2.findContours(np.uint8(mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if cv2.__version__[:2] == '3.':
        coords = results[1]
    else:
        coords = results[0]
    #coords, hierarchy = cv2.findContours(np.uint8(mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(coords) > 1:
        print('Multiple contours detected')
        lengths = []
        for coord in coords:
            lengths.append(len(coord))
        coords = [coords[np.argmax(lengths)]]
    if len(coords) > 0:
        coord = coords[0]
        coord = np.squeeze(coord, axis=(1,))
        coord = np.append(coord, coord[:1], axis=0)
    else:
        coord = np.empty(0)
    return coord

def hausdorff_distance(coord1, coord2, pixel_spacing):
    max_of_min1 = scipy.spatial.distance.directed_hausdorff(coord1, coord2)[0]
    max_of_min2 = scipy.spatial.distance.directed_hausdorff(coord2, coord1)[0]
    return max(max_of_min1, max_of_min2) * pixel_spacing


def extract_2D_mask_boundary(m):
    w = m.shape[0]
    h = m.shape[1]
    mm = m
    for wi in range(1, w-1):
        for hi in range(1, h-1):
            if m[wi, hi] > 0 and m[wi-1, hi] > 0 and m[wi+1, hi] > 0 and m[wi, hi-1] > 0 and m[wi, hi+1] > 0:
                mm[wi, hi] = 0
    return mm

def volume_Dice(p_volume, m_volume, min_v, max_v):
    p = np.where(np.logical_and(p_volume>=min_v, p_volume<=max_v), 
        np.ones_like(p_volume), np.zeros_like(p_volume))
    m = np.where(np.logical_and(m_volume>=min_v, m_volume<=max_v), 
        np.ones_like(m_volume), np.zeros_like(m_volume))
    Dices = []
    slices = p.shape[0]
    for s in range(slices):
        p_s = p[s, :, :].flatten()
        m_s = m[s, :, :].flatten()
        if (np.sum(m_s) > 0):
            Dice = 2.0 * np.sum(p_s * m_s) / (np.sum(p_s) + np.sum(m_s))
        else:
            Dice = -1
        Dices.append(Dice)
    return Dices

def volume_Dice_3D(p_volume, m_volume, min_v, max_v):
    p = np.where(np.logical_and(p_volume>=min_v, p_volume<=max_v), 
        np.ones_like(p_volume), np.zeros_like(p_volume))
    m = np.where(np.logical_and(m_volume>=min_v, m_volume<=max_v), 
        np.ones_like(m_volume), np.zeros_like(m_volume))
    
    return dc(m, p)

def volume_APD(p_volume, m_volume, min_v, max_v, pixel_spacing, eng):
    p = np.where(np.logical_and(p_volume>=min_v, p_volume<=max_v), 
        np.ones_like(p_volume), np.zeros_like(p_volume))
    m = np.where(np.logical_and(m_volume>=min_v, m_volume<=max_v), 
        np.ones_like(m_volume), np.zeros_like(m_volume))
    APDs = []
    slices = p.shape[0]
    for s in range(slices):
        p_s = p[s, :, :]
        m_s = m[s, :, :]
        p_s_ctr = mask_to_contour(p_s * 255)
        m_s_ctr = mask_to_contour(m_s * 255)
        if len(p_s_ctr.shape) == 2:
            try:
                APD = eng.average_perpendicular_distance(p_s_ctr[:, 0], p_s_ctr[:, 1], 
                    m_s_ctr[:, 0], m_s_ctr[:, 1], p_s.shape[0], p_s.shape[1], pixel_spacing)
            except Exception as e:
                print(e)
                APD = -2
        else:
            APD = -1
        APDs.append(APD)
    return APDs

def volume_APD2(p_volume, m_txt, min_v, max_v, pixel_spacing, to_original_x, to_original_y, eng):
    p = np.where(np.logical_and(p_volume>=min_v, p_volume<=max_v), 
        np.ones_like(p_volume), np.zeros_like(p_volume))
    APDs = []
    slices = p.shape[0]
    for s in range(slices):
        p_s = p[s, :, :]
        p_s_ctr = mask_to_contour(p_s * 255)
        m_s_ctr = np.loadtxt(m_txt[s])
        if len(p_s_ctr.shape) == 2:
            p_s_ctr[:, 0] += to_original_x
            p_s_ctr[:, 1] += to_original_y
            try:
                APD = eng.average_perpendicular_distance(p_s_ctr[:, 0].tolist(), p_s_ctr[:, 1].tolist(), 
                    m_s_ctr[:, 0].tolist(), m_s_ctr[:, 1].tolist(), int(p_s.shape[0]), int(p_s.shape[1]), pixel_spacing)
            except Exception as e:
                print(e)
                APD = -2
        else:
            APD = -1
        APDs.append(APD)
    return APDs


def volume_hausdorff_distance(p_volume, m_volume, min_v, max_v, pixel_spacing, to_contours):
    p = np.where(np.logical_and(p_volume>=min_v, p_volume<=max_v), 
        np.ones_like(p_volume), np.zeros_like(p_volume))
    m = np.where(np.logical_and(m_volume>=min_v, m_volume<=max_v), 
        np.ones_like(m_volume), np.zeros_like(m_volume))
    HDs = []
    slices = p.shape[0]
    for s in range(slices):
        p_s = p[s, :, :]
        m_s = m[s, :, :]
        if (np.sum(m_s.flatten()) > 0):
            if (np.sum(p_s.flatten()) > 0):
                if to_contours:
                    p_s_ctr = mask_to_contour(p_s * 255)
                    m_s_ctr = mask_to_contour(m_s * 255)
                    HD = hausdorff_distance(p_s_ctr, m_s_ctr, pixel_spacing)
                else:
                    p_s_b = extract_2D_mask_boundary(p_s)
                    m_s_b = extract_2D_mask_boundary(m_s)
                    p_s_coord = np.argwhere(np.array(p_s_b, dtype=bool))
                    m_s_coord = np.argwhere(np.array(m_s_b, dtype=bool))
                    HD = hausdorff_distance(p_s_coord, m_s_coord, pixel_spacing)
            else:
                HD = -2
        else:
            HD = -1
        HDs.append(HD)
    return HDs


def volume_hausdorff_distance2(p_volume, m_txt, min_v, max_v, pixel_spacing, 
    to_original_x, to_original_y, to_contours):
    p = np.where(np.logical_and(p_volume>=min_v, p_volume<=max_v), 
        np.ones_like(p_volume), np.zeros_like(p_volume))
    HDs = []
    slices = p.shape[0]
    for s in range(slices):
        p_s = p[s, :, :]
        m_s_ctr = np.loadtxt(m_txt[s])
        if (np.sum(p_s.flatten()) > 0):
            try:  
                if to_contours:
                    p_s_ctr = mask_to_contour(p_s * 255)
                    p_s_ctr[:, 0] += to_original_x
                    p_s_ctr[:, 1] += to_original_y
                    HD = hausdorff_distance(p_s_ctr, m_s_ctr, pixel_spacing)
                else:
                    p_s_b = extract_2D_mask_boundary(p_s)
                    p_s_coord = np.argwhere(np.array(p_s_b, dtype=bool))
                    p_s_coord[:, 0] += to_original_x
                    p_s_coord[:, 1] += to_original_y
                    HD = hausdorff_distance(p_s_coord, m_s_ctr, pixel_spacing)
            except Exception as e:
                print(e)
                HD = -2
        else:
            HD = -2
        HDs.append(HD)
    return HDs

def volume_hausdorff_distance_3D(p_volume, m_volume, min_v, max_v, pixel_spacing):
    p = np.where(np.logical_and(p_volume>=min_v, p_volume<=max_v), 
        np.ones_like(p_volume), np.zeros_like(p_volume))
    m = np.where(np.logical_and(m_volume>=min_v, m_volume<=max_v), 
        np.ones_like(m_volume), np.zeros_like(m_volume))
    
    return hd(m, p, pixel_spacing)



def mean_of_positive_elements(l):
    return 1.0 * sum([x for x in l if x >= 0]) / max(len([x for x in l if x >= 0]), 1)





def number_of_components(array, value):
    v_mask = np.where(array==value, np.ones_like(array), np.zeros_like(array))
    connected_components, num_connected_components = ndimage.label(v_mask)
    return num_connected_components

def keep_largest_components(array, keep_values=[1, 2, 3], values=[1, 2, 3]):
    output = np.zeros_like(array)
    for v in values:
        v_mask = np.where(array==v, np.ones_like(array), np.zeros_like(array))
        connected_components, num_connected_components = ndimage.label(v_mask)
        if (num_connected_components > 1) and (v in keep_values):
            unique, counts = np.unique(connected_components, return_counts=True)
            max_idx = np.where(counts == max(counts[1:]))[0][0]
            v_mask = v_mask * (connected_components == max_idx)
        output = output + v * v_mask
    return output


def second_largest_component_ratio(array, value):
    v_mask = np.where(array==value, np.ones_like(array), np.zeros_like(array))
    connected_components, num_connected_components = ndimage.label(v_mask)
    if (num_connected_components > 1):
        unique, counts = np.unique(connected_components, return_counts=True)
        max_idx = np.where(counts == max(counts[1:]))[0][0]
        max_count = (connected_components == max_idx).sum()
        second_tmp_idx = counts[1:].argsort()[-2]
        second_max_idx = np.where(counts == counts[1:][second_tmp_idx])[0][0]
        second_max_count = (connected_components == second_max_idx).sum()
        return float(second_max_count) / max_count
    else:
        return 0.0


def v1_touch_v2(array, size_x, size_y, v1, v2, threshold=10):
    touch_count = 0
    for p in range(size_x * size_y):
        r = p // size_x
        c = p % size_x
        if array[r, c] == v1:
            up_touch = (r != 0) and (array[r-1, c] == v2)
            down_touch = (r != size_y-1) and (array[r+1, c] == v2)
            left_touch = (c != 0) and (array[r, c-1] == v2)
            right_touch = (c != size_x-1) and (array[r, c+1] == v2)

            touch_count += (up_touch + down_touch + left_touch + right_touch)
    return touch_count >= threshold


def touch_length_count(array, size_x, size_y, v1, v2):
    touch_count = 0
    for p in range(size_x * size_y):
        r = p // size_x
        c = p % size_x
        if array[r, c] == v1:
            up_touch = (r != 0) and (array[r-1, c] == v2)
            down_touch = (r != size_y-1) and (array[r+1, c] == v2)
            left_touch = (c != 0) and (array[r, c-1] == v2)
            right_touch = (c != size_x-1) and (array[r, c+1] == v2)

            touch_count += (up_touch + down_touch + left_touch + right_touch)
    return touch_count

def area_boundary_ratio(array, size_x, size_y, v):
    area = 0
    boundary = 0
    for p in range(size_x * size_y):
        r = p // size_x
        c = p % size_x
        if array[r, c] == v:
            up_touch = (r != 0) and (array[r-1, c] != v)
            down_touch = (r != size_y-1) and (array[r+1, c] != v)
            left_touch = (c != 0) and (array[r, c-1] != v)
            right_touch = (c != size_x-1) and (array[r, c+1] != v)

            area += 1
            boundary += (up_touch + down_touch + left_touch + right_touch)
    return float(area)/(boundary*boundary)

def change_neighbor_value(array, size_x, size_y, v0, v1, v2):
    output = array
    for p in range(size_x * size_y):
        r = p // size_x
        c = p % size_x
        if array[r, c] == v0:
            if (r != 0) and (array[r-1, c] == v1):
                output[r-1, c] = v2
            if (r != size_y-1) and (array[r+1, c] == v1):
                output[r+1, c] = v2
            if (c != 0) and (array[r, c-1] == v1):
                output[r, c-1] = v2
            if (c != size_x-1) and (array[r, c+1] == v1):
                output[r, c+1] = v2
    return output



def save_layer_output(model, data, layer_name='output', save_path_prefix='output'):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(data)
    batch_size = intermediate_output.shape[0]
    for i in range(batch_size):
        data = intermediate_output[i]
        if (data.shape[2] == 1):
            img = array_to_img(data, data_format=None, scale=True)
            img.save(save_path_prefix + str(i).zfill(2) + ".png")
            np.savetxt(save_path_prefix + str(i).zfill(2) + ".txt", data, fmt='%.6f')
        else:
            for j in range(data.shape[2]):
                data_j = data[:,:,j:(j+1)]
                img_j = array_to_img(data_j, data_format=None, scale=True)
                img_j.save(save_path_prefix + str(i).zfill(2) + "_" + str(j).zfill(2) + ".png")
                np.savetxt(save_path_prefix + str(i).zfill(2) + "_" + str(j).zfill(2) + ".txt", data_j, fmt='%.6f')

def print_model_weights(model):
    layers = model.layers[-10:]
    all_weights = []
    for layer in layers:
        all_weights += layer.weights
    evaluated_weights = K.batch_get_value(all_weights)
    for layer in layers:
        print("\nlayer #{} {}:{}".format(model.layers.index(layer), layer.__class__.__name__, layer.name) )
        print("input shape: {}".format(layer.input_shape) )
        print("output shape: {}".format(layer.output_shape) )
        for j, w in enumerate(layer.trainable_weights):
            w = evaluated_weights[all_weights.index(w)]
            wf = w.flatten()
            s = w.size
            print(" weights #{} {}: {} {} {} {}".format(j, w.shape,\
                wf[int(s//5)], wf[int(2*s//5)], wf[int(3*s//5)], wf[int(4*s//5)]) )


def print_model_gradients(model, batch_img, batch_mask):
    layers = model.layers[-10:]
    all_trainable_weights = []
    for layer in layers:
        all_trainable_weights += layer.trainable_weights
    gradient_list = model.optimizer.get_gradients(model.total_loss,
                                                  all_trainable_weights)
    '''
    get_gradient_list = K.function(
        inputs=[model.inputs[0], model.targets[0], model.sample_weights[0],
                K.learning_phase()],
        outputs=gradient_list,
        updates=None)
    evaluated_gradient_list = get_gradient_list(
        [batch_img, batch_mask, [1.0]*(batch_img.shape[0]), 1])
    '''
    evaluated_gradient_list = K.get_session().run(
         gradient_list, 
         feed_dict={model.inputs[0]: batch_img,
                    model.targets[0]: batch_mask,
                    model.sample_weights[0]: [1.0]*(batch_img.shape[0]),
                    K.learning_phase(): 1})

    for layer in layers:
        print("\nlayer #{} {}:{}".format(model.layers.index(layer), layer.__class__.__name__, layer.name) )
        print("input shape: {}".format(layer.input_shape) )
        print("output shape: {}".format(layer.output_shape) )

        for j, w in enumerate(layer.trainable_weights):
            g = evaluated_gradient_list[all_trainable_weights.index(w)]
            gf = g.flatten()
            s = g.size
            print(" gradients #{} {}: {} {} {} {}".format(j, g.shape,\
                gf[int(s//5)], gf[int(2*s//5)], gf[int(3*s//5)], gf[int(4*s//5)]) )




def print_model_weights_gradients(model, batch_img, batch_mask):
    layers = model.layers[-10:]

    all_weights = []
    all_trainable_weights = []
    for layer in layers:
        all_weights += layer.weights
        all_trainable_weights += layer.trainable_weights
    gradient_list = model.optimizer.get_gradients(model.total_loss,
                                                  all_trainable_weights)
    weights_len = len(all_weights)
    gradient_len = len(gradient_list)
    if (weights_len + gradient_len > 0):
        if not isinstance(batch_img, list):
            evaluated = K.get_session().run(
                all_weights + gradient_list, 
                feed_dict={model.inputs[0]: batch_img,
                           model.targets[0]: batch_mask,
                           model.sample_weights[0]: [1.0]*(batch_img.shape[0]),
                           K.learning_phase(): 1})

        elif not isinstance(batch_mask, list):
            if len(batch_img) == 2:
                evaluated = K.get_session().run(
                    all_weights + gradient_list, 
                    feed_dict={model.inputs[0]: batch_img[0],
                               model.inputs[1]: batch_img[1],
                               model.targets[0]: batch_mask,
                               model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                               K.learning_phase(): 1})
            elif len(batch_img) == 3:
                evaluated = K.get_session().run(
                    all_weights + gradient_list, 
                    feed_dict={model.inputs[0]: batch_img[0],
                               model.inputs[1]: batch_img[1],
                               model.inputs[2]: batch_img[2],
                               model.targets[0]: batch_mask,
                               model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                               K.learning_phase(): 1})
            elif len(batch_img) == 4:
                evaluated = K.get_session().run(
                    all_weights + gradient_list, 
                    feed_dict={model.inputs[0]: batch_img[0],
                               model.inputs[1]: batch_img[1],
                               model.inputs[2]: batch_img[2],
                               model.inputs[3]: batch_img[3],
                               model.targets[0]: batch_mask,
                               model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                               K.learning_phase(): 1})
        else:
            if len(batch_img) == 2 and len(batch_mask) == 2:
                evaluated = K.get_session().run(
                    all_weights + gradient_list, 
                    feed_dict={model.inputs[0]: batch_img[0],
                               model.inputs[1]: batch_img[1],
                               model.targets[0]: batch_mask[0],
                               model.targets[1]: batch_mask[1],
                               model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                               model.sample_weights[1]: [1.0]*(batch_img[0].shape[0]),
                               K.learning_phase(): 1})
            elif len(batch_img) == 3 and len(batch_mask) == 2:
                evaluated = K.get_session().run(
                    all_weights + gradient_list, 
                    feed_dict={model.inputs[0]: batch_img[0],
                               model.inputs[1]: batch_img[1],
                               model.inputs[2]: batch_img[2],
                               model.targets[0]: batch_mask[0],
                               model.targets[1]: batch_mask[1],
                               model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                               model.sample_weights[1]: [1.0]*(batch_img[0].shape[0]),
                               K.learning_phase(): 1})
            elif len(batch_img) == 3 and len(batch_mask) == 3:
                evaluated = K.get_session().run(
                    all_weights + gradient_list, 
                    feed_dict={model.inputs[0]: batch_img[0],
                               model.inputs[1]: batch_img[1],
                               model.inputs[2]: batch_img[2],
                               model.targets[0]: batch_mask[0],
                               model.targets[1]: batch_mask[1],
                               model.targets[2]: batch_mask[2],
                               model.sample_weights[0]: [1.0]*(batch_img[0].shape[0]),
                               model.sample_weights[1]: [1.0]*(batch_img[0].shape[0]),
                               model.sample_weights[2]: [1.0]*(batch_img[0].shape[0]),
                               K.learning_phase(): 1})
    else:
        evaluated = []

    evaluated_all_weights = evaluated[:weights_len]
    evaluated_gradient_list = evaluated[weights_len:]

    for layer in layers:
        print("\nlayer #{} {}:{}".format(model.layers.index(layer), layer.__class__.__name__, layer.name) )
        print("input shape: {}".format(layer.input_shape) )
        print("output shape: {}".format(layer.output_shape) )

        for j, wt in enumerate(layer.trainable_weights):
            w = evaluated_all_weights[all_weights.index(wt)]
            wf = w.flatten()
            s = w.size
            print(" t_weights #{} {}: {} {} {} {}".format(j, w.shape,\
                wf[int(s//5)], wf[int(2*s//5)], wf[int(3*s//5)], wf[int(4*s//5)]) )

            g = evaluated_gradient_list[all_trainable_weights.index(wt)]
            gf = g.flatten()
            s = g.size
            print(" gradients #{} {}: {} {} {} {}".format(j, g.shape,\
                gf[int(s//5)], gf[int(2*s//5)], gf[int(3*s//5)], gf[int(4*s//5)]) )

        for j, wt in enumerate(layer.non_trainable_weights):
            w = evaluated_all_weights[all_weights.index(wt)]
            wf = w.flatten()
            s = w.size
            print(" nont_weights #{} {}: {} {} {} {}".format(j, w.shape,\
                wf[int(s//5)], wf[int(2*s//5)], wf[int(3*s//5)], wf[int(4*s//5)]) )




def handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


