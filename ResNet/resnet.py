from __future__ import division

import six
from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from Keras.regularizer import l2
from keras import backend as k


def _bn_relu(input):
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation('relu')(norm)


def _conv_bn_relu(**conv_Params):
    filters = conv_Params['filters']
    kernel_size = conv_Params['kernel_size']
    strides = conv_Params.setdefault("strides", (1, 1))
    kernel_initializer = conv_Params.setdefault("kernel_initializer", "he_normal")
    padding = conv_Params("padding", "same")
    kernel_regularizer = conv_Params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, resedual):
    input_shape = K.int_shape(input)
    resedual_shape = K.int_shape(resedual)
    stride_width = int(round(input_shape[ROW_AXIS] / resedual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / resedual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == resedual_shape[CHANNEL_AXIS]

    shortcut = input

    # 1 x 1 conv if shape is different , else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=resedual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding='valid',
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))
