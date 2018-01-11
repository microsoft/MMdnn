# -*- coding: utf-8 -*-
"""Inception V3 model for Keras.

Note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function is also different (same as Xception).

# Reference

- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)

"""
from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

import keras
from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D, Concatenate
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.preprocessing import image


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    Arguments:
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def InceptionV3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    """Instantiates the Inception v3 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input image size for this model is 299x299.

    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
        A Keras model instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
                
    input_1         = Input(shape = (299, 299, 3,), dtype = "float32")
    conv2d_1        = Conv2D(filters = 32, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', use_bias = False)(input_1)
    batch_normalization_1 = BatchNormalization(name = 'batch_normalization_1', axis = 3, scale = False)(conv2d_1)
    activation_1    = Activation('relu')(batch_normalization_1)
    conv2d_2        = Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1), padding = 'valid', use_bias = False)(activation_1)
    batch_normalization_2 = BatchNormalization(name = 'batch_normalization_2', axis = 3, scale = False)(conv2d_2)
    activation_2    = Activation('relu')(batch_normalization_2)
    conv2d_3        = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = False)(activation_2)
    batch_normalization_3 = BatchNormalization(name = 'batch_normalization_3', axis = 3, scale = False)(conv2d_3)
    activation_3    = Activation('relu')(batch_normalization_3)
    max_pooling2d_1 = MaxPooling2D(name = 'max_pooling2d_1', pool_size = (3, 3), strides = (2, 2), padding = 'valid')(activation_3)
    conv2d_4        = Conv2D(filters = 80, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = False)(max_pooling2d_1)
    batch_normalization_4 = BatchNormalization(name = 'batch_normalization_4', axis = 3, scale = False)(conv2d_4)
    activation_4    = Activation('relu')(batch_normalization_4)
    conv2d_5        = Conv2D(filters = 192, kernel_size = (3, 3), strides = (1, 1), padding = 'valid', use_bias = False)(activation_4)
    batch_normalization_5 = BatchNormalization(name = 'batch_normalization_5', axis = 3, scale = False)(conv2d_5)
    activation_5    = Activation('relu')(batch_normalization_5)
    max_pooling2d_2 = MaxPooling2D(name = 'max_pooling2d_2', pool_size = (3, 3), strides = (2, 2), padding = 'valid')(activation_5)
    conv2d_9        = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(max_pooling2d_2)
    batch_normalization_9 = BatchNormalization(name = 'batch_normalization_9', axis = 3, scale = False)(conv2d_9)
    activation_9    = Activation('relu')(batch_normalization_9)
    conv2d_10       = Conv2D(filters = 96, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = False)(activation_9)
    batch_normalization_10 = BatchNormalization(name = 'batch_normalization_10', axis = 3, scale = False)(conv2d_10)
    activation_10   = Activation('relu')(batch_normalization_10)
    conv2d_11       = Conv2D(filters = 96, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = False)(activation_10)
    batch_normalization_11 = BatchNormalization(name = 'batch_normalization_11', axis = 3, scale = False)(conv2d_11)
    activation_11   = Activation('relu')(batch_normalization_11)
    conv2d_7        = Conv2D(filters = 48, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(max_pooling2d_2)
    batch_normalization_7 = BatchNormalization(name = 'batch_normalization_7', axis = 3, scale = False)(conv2d_7)
    activation_7    = Activation('relu')(batch_normalization_7)
    conv2d_8        = Conv2D(filters = 64, kernel_size = (5, 5), strides = (1, 1), padding = 'same', use_bias = False)(activation_7)
    batch_normalization_8 = BatchNormalization(name = 'batch_normalization_8', axis = 3, scale = False)(conv2d_8)
    activation_8    = Activation('relu')(batch_normalization_8)
    average_pooling2d_1 = AveragePooling2D(name = 'average_pooling2d_1', pool_size = (3, 3), strides = (1, 1), padding = 'same')(max_pooling2d_2)
    conv2d_12       = Conv2D(filters = 32, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(average_pooling2d_1)
    batch_normalization_12 = BatchNormalization(name = 'batch_normalization_12', axis = 3, scale = False)(conv2d_12)
    activation_12   = Activation('relu')(batch_normalization_12)
    conv2d_6        = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(max_pooling2d_2)
    batch_normalization_6 = BatchNormalization(name = 'batch_normalization_6', axis = 3, scale = False)(conv2d_6)
    activation_6    = Activation('relu')(batch_normalization_6)
    mixed0          = layers.concatenate(name = 'mixed0', inputs = [activation_6, activation_8, activation_11, activation_12])
    conv2d_16       = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed0)
    batch_normalization_16 = BatchNormalization(name = 'batch_normalization_16', axis = 3, scale = False)(conv2d_16)
    activation_16   = Activation('relu')(batch_normalization_16)
    conv2d_17       = Conv2D(filters = 96, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = False)(activation_16)
    batch_normalization_17 = BatchNormalization(name = 'batch_normalization_17', axis = 3, scale = False)(conv2d_17)
    activation_17   = Activation('relu')(batch_normalization_17)
    conv2d_18       = Conv2D(filters = 96, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = False)(activation_17)
    batch_normalization_18 = BatchNormalization(name = 'batch_normalization_18', axis = 3, scale = False)(conv2d_18)
    activation_18   = Activation('relu')(batch_normalization_18)
    conv2d_14       = Conv2D(filters = 48, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed0)
    batch_normalization_14 = BatchNormalization(name = 'batch_normalization_14', axis = 3, scale = False)(conv2d_14)
    activation_14   = Activation('relu')(batch_normalization_14)
    conv2d_15       = Conv2D(filters = 64, kernel_size = (5, 5), strides = (1, 1), padding = 'same', use_bias = False)(activation_14)
    batch_normalization_15 = BatchNormalization(name = 'batch_normalization_15', axis = 3, scale = False)(conv2d_15)
    activation_15   = Activation('relu')(batch_normalization_15)
    average_pooling2d_2 = AveragePooling2D(name = 'average_pooling2d_2', pool_size = (3, 3), strides = (1, 1), padding = 'same')(mixed0)
    conv2d_19       = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(average_pooling2d_2)
    batch_normalization_19 = BatchNormalization(name = 'batch_normalization_19', axis = 3, scale = False)(conv2d_19)
    activation_19   = Activation('relu')(batch_normalization_19)
    conv2d_13       = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed0)
    batch_normalization_13 = BatchNormalization(name = 'batch_normalization_13', axis = 3, scale = False)(conv2d_13)
    activation_13   = Activation('relu')(batch_normalization_13)
    mixed1          = layers.concatenate(name = 'mixed1', inputs = [activation_13, activation_15, activation_18, activation_19])
    conv2d_23       = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed1)
    batch_normalization_23 = BatchNormalization(name = 'batch_normalization_23', axis = 3, scale = False)(conv2d_23)
    activation_23   = Activation('relu')(batch_normalization_23)
    conv2d_24       = Conv2D(filters = 96, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = False)(activation_23)
    batch_normalization_24 = BatchNormalization(name = 'batch_normalization_24', axis = 3, scale = False)(conv2d_24)
    activation_24   = Activation('relu')(batch_normalization_24)
    conv2d_25       = Conv2D(filters = 96, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = False)(activation_24)
    batch_normalization_25 = BatchNormalization(name = 'batch_normalization_25', axis = 3, scale = False)(conv2d_25)
    activation_25   = Activation('relu')(batch_normalization_25)
    conv2d_21       = Conv2D(filters = 48, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed1)
    batch_normalization_21 = BatchNormalization(name = 'batch_normalization_21', axis = 3, scale = False)(conv2d_21)
    activation_21   = Activation('relu')(batch_normalization_21)
    conv2d_22       = Conv2D(filters = 64, kernel_size = (5, 5), strides = (1, 1), padding = 'same', use_bias = False)(activation_21)
    batch_normalization_22 = BatchNormalization(name = 'batch_normalization_22', axis = 3, scale = False)(conv2d_22)
    activation_22   = Activation('relu')(batch_normalization_22)
    average_pooling2d_3 = AveragePooling2D(name = 'average_pooling2d_3', pool_size = (3, 3), strides = (1, 1), padding = 'same')(mixed1)
    conv2d_26       = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(average_pooling2d_3)
    batch_normalization_26 = BatchNormalization(name = 'batch_normalization_26', axis = 3, scale = False)(conv2d_26)
    activation_26   = Activation('relu')(batch_normalization_26)
    conv2d_20       = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed1)
    batch_normalization_20 = BatchNormalization(name = 'batch_normalization_20', axis = 3, scale = False)(conv2d_20)
    activation_20   = Activation('relu')(batch_normalization_20)
    mixed2          = layers.concatenate(name = 'mixed2', inputs = [activation_20, activation_22, activation_25, activation_26])
    conv2d_28       = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed2)
    batch_normalization_28 = BatchNormalization(name = 'batch_normalization_28', axis = 3, scale = False)(conv2d_28)
    activation_28   = Activation('relu')(batch_normalization_28)
    conv2d_29       = Conv2D(filters = 96, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = False)(activation_28)
    batch_normalization_29 = BatchNormalization(name = 'batch_normalization_29', axis = 3, scale = False)(conv2d_29)
    activation_29   = Activation('relu')(batch_normalization_29)
    conv2d_30       = Conv2D(filters = 96, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', use_bias = False)(activation_29)
    batch_normalization_30 = BatchNormalization(name = 'batch_normalization_30', axis = 3, scale = False)(conv2d_30)
    activation_30   = Activation('relu')(batch_normalization_30)
    conv2d_27       = Conv2D(filters = 384, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', use_bias = False)(mixed2)
    batch_normalization_27 = BatchNormalization(name = 'batch_normalization_27', axis = 3, scale = False)(conv2d_27)
    activation_27   = Activation('relu')(batch_normalization_27)
    max_pooling2d_3 = MaxPooling2D(name = 'max_pooling2d_3', pool_size = (3, 3), strides = (2, 2), padding = 'valid')(mixed2)
    mixed3          = layers.concatenate(name = 'mixed3', inputs = [activation_27, activation_30, max_pooling2d_3])
    conv2d_35       = Conv2D(filters = 128, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed3)
    batch_normalization_35 = BatchNormalization(name = 'batch_normalization_35', axis = 3, scale = False)(conv2d_35)
    activation_35   = Activation('relu')(batch_normalization_35)
    conv2d_36       = Conv2D(filters = 128, kernel_size = (7, 1), strides = (1, 1), padding = 'same', use_bias = False)(activation_35)
    batch_normalization_36 = BatchNormalization(name = 'batch_normalization_36', axis = 3, scale = False)(conv2d_36)
    activation_36   = Activation('relu')(batch_normalization_36)
    conv2d_37       = Conv2D(filters = 128, kernel_size = (1, 7), strides = (1, 1), padding = 'same', use_bias = False)(activation_36)
    batch_normalization_37 = BatchNormalization(name = 'batch_normalization_37', axis = 3, scale = False)(conv2d_37)
    activation_37   = Activation('relu')(batch_normalization_37)
    conv2d_38       = Conv2D(filters = 128, kernel_size = (7, 1), strides = (1, 1), padding = 'same', use_bias = False)(activation_37)
    batch_normalization_38 = BatchNormalization(name = 'batch_normalization_38', axis = 3, scale = False)(conv2d_38)
    activation_38   = Activation('relu')(batch_normalization_38)
    conv2d_39       = Conv2D(filters = 192, kernel_size = (1, 7), strides = (1, 1), padding = 'same', use_bias = False)(activation_38)
    batch_normalization_39 = BatchNormalization(name = 'batch_normalization_39', axis = 3, scale = False)(conv2d_39)
    activation_39   = Activation('relu')(batch_normalization_39)
    conv2d_32       = Conv2D(filters = 128, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed3)
    batch_normalization_32 = BatchNormalization(name = 'batch_normalization_32', axis = 3, scale = False)(conv2d_32)
    activation_32   = Activation('relu')(batch_normalization_32)
    conv2d_33       = Conv2D(filters = 128, kernel_size = (1, 7), strides = (1, 1), padding = 'same', use_bias = False)(activation_32)
    batch_normalization_33 = BatchNormalization(name = 'batch_normalization_33', axis = 3, scale = False)(conv2d_33)
    activation_33   = Activation('relu')(batch_normalization_33)
    conv2d_34       = Conv2D(filters = 192, kernel_size = (7, 1), strides = (1, 1), padding = 'same', use_bias = False)(activation_33)
    batch_normalization_34 = BatchNormalization(name = 'batch_normalization_34', axis = 3, scale = False)(conv2d_34)
    activation_34   = Activation('relu')(batch_normalization_34)
    average_pooling2d_4 = AveragePooling2D(name = 'average_pooling2d_4', pool_size = (3, 3), strides = (1, 1), padding = 'same')(mixed3)
    conv2d_40       = Conv2D(filters = 192, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(average_pooling2d_4)
    batch_normalization_40 = BatchNormalization(name = 'batch_normalization_40', axis = 3, scale = False)(conv2d_40)
    activation_40   = Activation('relu')(batch_normalization_40)
    conv2d_31       = Conv2D(filters = 192, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed3)
    batch_normalization_31 = BatchNormalization(name = 'batch_normalization_31', axis = 3, scale = False)(conv2d_31)
    activation_31   = Activation('relu')(batch_normalization_31)
    mixed4          = layers.concatenate(name = 'mixed4', inputs = [activation_31, activation_34, activation_39, activation_40])
    conv2d_45       = Conv2D(filters = 160, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed4)
    batch_normalization_45 = BatchNormalization(name = 'batch_normalization_45', axis = 3, scale = False)(conv2d_45)
    activation_45   = Activation('relu')(batch_normalization_45)
    conv2d_46       = Conv2D(filters = 160, kernel_size = (7, 1), strides = (1, 1), padding = 'same', use_bias = False)(activation_45)
    batch_normalization_46 = BatchNormalization(name = 'batch_normalization_46', axis = 3, scale = False)(conv2d_46)
    activation_46   = Activation('relu')(batch_normalization_46)
    conv2d_47       = Conv2D(filters = 160, kernel_size = (1, 7), strides = (1, 1), padding = 'same', use_bias = False)(activation_46)
    batch_normalization_47 = BatchNormalization(name = 'batch_normalization_47', axis = 3, scale = False)(conv2d_47)
    activation_47   = Activation('relu')(batch_normalization_47)
    conv2d_48       = Conv2D(filters = 160, kernel_size = (7, 1), strides = (1, 1), padding = 'same', use_bias = False)(activation_47)
    batch_normalization_48 = BatchNormalization(name = 'batch_normalization_48', axis = 3, scale = False)(conv2d_48)
    activation_48   = Activation('relu')(batch_normalization_48)
    conv2d_49       = Conv2D(filters = 192, kernel_size = (1, 7), strides = (1, 1), padding = 'same', use_bias = False)(activation_48)
    batch_normalization_49 = BatchNormalization(name = 'batch_normalization_49', axis = 3, scale = False)(conv2d_49)
    activation_49   = Activation('relu')(batch_normalization_49)
    conv2d_42       = Conv2D(filters = 160, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed4)
    batch_normalization_42 = BatchNormalization(name = 'batch_normalization_42', axis = 3, scale = False)(conv2d_42)
    activation_42   = Activation('relu')(batch_normalization_42)
    conv2d_43       = Conv2D(filters = 160, kernel_size = (1, 7), strides = (1, 1), padding = 'same', use_bias = False)(activation_42)
    batch_normalization_43 = BatchNormalization(name = 'batch_normalization_43', axis = 3, scale = False)(conv2d_43)
    activation_43   = Activation('relu')(batch_normalization_43)
    conv2d_44       = Conv2D(filters = 192, kernel_size = (7, 1), strides = (1, 1), padding = 'same', use_bias = False)(activation_43)
    batch_normalization_44 = BatchNormalization(name = 'batch_normalization_44', axis = 3, scale = False)(conv2d_44)
    activation_44   = Activation('relu')(batch_normalization_44)
    average_pooling2d_5 = AveragePooling2D(name = 'average_pooling2d_5', pool_size = (3, 3), strides = (1, 1), padding = 'same')(mixed4)
    conv2d_50       = Conv2D(filters = 192, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(average_pooling2d_5)
    batch_normalization_50 = BatchNormalization(name = 'batch_normalization_50', axis = 3, scale = False)(conv2d_50)
    activation_50   = Activation('relu')(batch_normalization_50)
    conv2d_41       = Conv2D(filters = 192, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed4)
    batch_normalization_41 = BatchNormalization(name = 'batch_normalization_41', axis = 3, scale = False)(conv2d_41)
    activation_41   = Activation('relu')(batch_normalization_41)
    mixed5          = layers.concatenate(name = 'mixed5', inputs = [activation_41, activation_44, activation_49, activation_50])
    conv2d_55       = Conv2D(filters = 160, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed5)
    batch_normalization_55 = BatchNormalization(name = 'batch_normalization_55', axis = 3, scale = False)(conv2d_55)
    activation_55   = Activation('relu')(batch_normalization_55)
    conv2d_56       = Conv2D(filters = 160, kernel_size = (7, 1), strides = (1, 1), padding = 'same', use_bias = False)(activation_55)
    batch_normalization_56 = BatchNormalization(name = 'batch_normalization_56', axis = 3, scale = False)(conv2d_56)
    activation_56   = Activation('relu')(batch_normalization_56)
    conv2d_57       = Conv2D(filters = 160, kernel_size = (1, 7), strides = (1, 1), padding = 'same', use_bias = False)(activation_56)
    batch_normalization_57 = BatchNormalization(name = 'batch_normalization_57', axis = 3, scale = False)(conv2d_57)
    activation_57   = Activation('relu')(batch_normalization_57)
    conv2d_58       = Conv2D(filters = 160, kernel_size = (7, 1), strides = (1, 1), padding = 'same', use_bias = False)(activation_57)
    batch_normalization_58 = BatchNormalization(name = 'batch_normalization_58', axis = 3, scale = False)(conv2d_58)
    activation_58   = Activation('relu')(batch_normalization_58)
    conv2d_59       = Conv2D(filters = 192, kernel_size = (1, 7), strides = (1, 1), padding = 'same', use_bias = False)(activation_58)
    batch_normalization_59 = BatchNormalization(name = 'batch_normalization_59', axis = 3, scale = False)(conv2d_59)
    activation_59   = Activation('relu')(batch_normalization_59)
    conv2d_52       = Conv2D(filters = 160, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed5)
    batch_normalization_52 = BatchNormalization(name = 'batch_normalization_52', axis = 3, scale = False)(conv2d_52)
    activation_52   = Activation('relu')(batch_normalization_52)
    conv2d_53       = Conv2D(filters = 160, kernel_size = (1, 7), strides = (1, 1), padding = 'same', use_bias = False)(activation_52)
    batch_normalization_53 = BatchNormalization(name = 'batch_normalization_53', axis = 3, scale = False)(conv2d_53)
    activation_53   = Activation('relu')(batch_normalization_53)
    conv2d_54       = Conv2D(filters = 192, kernel_size = (7, 1), strides = (1, 1), padding = 'same', use_bias = False)(activation_53)
    batch_normalization_54 = BatchNormalization(name = 'batch_normalization_54', axis = 3, scale = False)(conv2d_54)
    activation_54   = Activation('relu')(batch_normalization_54)
    average_pooling2d_6 = AveragePooling2D(name = 'average_pooling2d_6', pool_size = (3, 3), strides = (1, 1), padding = 'same')(mixed5)
    conv2d_60       = Conv2D(filters = 192, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(average_pooling2d_6)
    batch_normalization_60 = BatchNormalization(name = 'batch_normalization_60', axis = 3, scale = False)(conv2d_60)
    activation_60   = Activation('relu')(batch_normalization_60)
    conv2d_51       = Conv2D(filters = 192, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed5)
    batch_normalization_51 = BatchNormalization(name = 'batch_normalization_51', axis = 3, scale = False)(conv2d_51)
    activation_51   = Activation('relu')(batch_normalization_51)
    mixed6          = layers.concatenate(name = 'mixed6', inputs = [activation_51, activation_54, activation_59, activation_60])
    conv2d_65       = Conv2D(filters = 192, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed6)
    batch_normalization_65 = BatchNormalization(name = 'batch_normalization_65', axis = 3, scale = False)(conv2d_65)
    activation_65   = Activation('relu')(batch_normalization_65)
    conv2d_66       = Conv2D(filters = 192, kernel_size = (7, 1), strides = (1, 1), padding = 'same', use_bias = False)(activation_65)
    batch_normalization_66 = BatchNormalization(name = 'batch_normalization_66', axis = 3, scale = False)(conv2d_66)
    activation_66   = Activation('relu')(batch_normalization_66)
    conv2d_67       = Conv2D(filters = 192, kernel_size = (1, 7), strides = (1, 1), padding = 'same', use_bias = False)(activation_66)
    batch_normalization_67 = BatchNormalization(name = 'batch_normalization_67', axis = 3, scale = False)(conv2d_67)
    activation_67   = Activation('relu')(batch_normalization_67)
    conv2d_68       = Conv2D(filters = 192, kernel_size = (7, 1), strides = (1, 1), padding = 'same', use_bias = False)(activation_67)
    batch_normalization_68 = BatchNormalization(name = 'batch_normalization_68', axis = 3, scale = False)(conv2d_68)
    activation_68   = Activation('relu')(batch_normalization_68)
    conv2d_69       = Conv2D(filters = 192, kernel_size = (1, 7), strides = (1, 1), padding = 'same', use_bias = False)(activation_68)
    batch_normalization_69 = BatchNormalization(name = 'batch_normalization_69', axis = 3, scale = False)(conv2d_69)
    activation_69   = Activation('relu')(batch_normalization_69)
    conv2d_62       = Conv2D(filters = 192, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed6)
    batch_normalization_62 = BatchNormalization(name = 'batch_normalization_62', axis = 3, scale = False)(conv2d_62)
    activation_62   = Activation('relu')(batch_normalization_62)
    conv2d_63       = Conv2D(filters = 192, kernel_size = (1, 7), strides = (1, 1), padding = 'same', use_bias = False)(activation_62)
    batch_normalization_63 = BatchNormalization(name = 'batch_normalization_63', axis = 3, scale = False)(conv2d_63)
    activation_63   = Activation('relu')(batch_normalization_63)
    conv2d_64       = Conv2D(filters = 192, kernel_size = (7, 1), strides = (1, 1), padding = 'same', use_bias = False)(activation_63)
    batch_normalization_64 = BatchNormalization(name = 'batch_normalization_64', axis = 3, scale = False)(conv2d_64)
    activation_64   = Activation('relu')(batch_normalization_64)
    average_pooling2d_7 = AveragePooling2D(name = 'average_pooling2d_7', pool_size = (3, 3), strides = (1, 1), padding = 'same')(mixed6)
    conv2d_70       = Conv2D(filters = 192, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(average_pooling2d_7)
    batch_normalization_70 = BatchNormalization(name = 'batch_normalization_70', axis = 3, scale = False)(conv2d_70)
    activation_70   = Activation('relu')(batch_normalization_70)
    conv2d_61       = Conv2D(filters = 192, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed6)
    batch_normalization_61 = BatchNormalization(name = 'batch_normalization_61', axis = 3, scale = False)(conv2d_61)
    activation_61   = Activation('relu')(batch_normalization_61)
    mixed7          = layers.concatenate(name = 'mixed7', inputs = [activation_61, activation_64, activation_69, activation_70])
    conv2d_73       = Conv2D(filters = 192, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed7)
    batch_normalization_73 = BatchNormalization(name = 'batch_normalization_73', axis = 3, scale = False)(conv2d_73)
    activation_73   = Activation('relu')(batch_normalization_73)
    conv2d_74       = Conv2D(filters = 192, kernel_size = (1, 7), strides = (1, 1), padding = 'same', use_bias = False)(activation_73)
    batch_normalization_74 = BatchNormalization(name = 'batch_normalization_74', axis = 3, scale = False)(conv2d_74)
    activation_74   = Activation('relu')(batch_normalization_74)
    conv2d_75       = Conv2D(filters = 192, kernel_size = (7, 1), strides = (1, 1), padding = 'same', use_bias = False)(activation_74)
    batch_normalization_75 = BatchNormalization(name = 'batch_normalization_75', axis = 3, scale = False)(conv2d_75)
    activation_75   = Activation('relu')(batch_normalization_75)
    conv2d_76       = Conv2D(filters = 192, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', use_bias = False)(activation_75)
    batch_normalization_76 = BatchNormalization(name = 'batch_normalization_76', axis = 3, scale = False)(conv2d_76)
    activation_76   = Activation('relu')(batch_normalization_76)
    conv2d_71       = Conv2D(filters = 192, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed7)
    batch_normalization_71 = BatchNormalization(name = 'batch_normalization_71', axis = 3, scale = False)(conv2d_71)
    activation_71   = Activation('relu')(batch_normalization_71)
    conv2d_72       = Conv2D(filters = 320, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', use_bias = False)(activation_71)
    batch_normalization_72 = BatchNormalization(name = 'batch_normalization_72', axis = 3, scale = False)(conv2d_72)
    activation_72   = Activation('relu')(batch_normalization_72)
    max_pooling2d_4 = MaxPooling2D(name = 'max_pooling2d_4', pool_size = (3, 3), strides = (2, 2), padding = 'valid')(mixed7)
    mixed8          = layers.concatenate(name = 'mixed8', inputs = [activation_72, activation_76, max_pooling2d_4])
    conv2d_81       = Conv2D(filters = 448, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed8)
    batch_normalization_81 = BatchNormalization(name = 'batch_normalization_81', axis = 3, scale = False)(conv2d_81)
    activation_81   = Activation('relu')(batch_normalization_81)
    conv2d_82       = Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = False)(activation_81)
    batch_normalization_82 = BatchNormalization(name = 'batch_normalization_82', axis = 3, scale = False)(conv2d_82)
    activation_82   = Activation('relu')(batch_normalization_82)
    conv2d_83       = Conv2D(filters = 384, kernel_size = (1, 3), strides = (1, 1), padding = 'same', use_bias = False)(activation_82)
    batch_normalization_83 = BatchNormalization(name = 'batch_normalization_83', axis = 3, scale = False)(conv2d_83)
    activation_83   = Activation('relu')(batch_normalization_83)
    conv2d_84       = Conv2D(filters = 384, kernel_size = (3, 1), strides = (1, 1), padding = 'same', use_bias = False)(activation_82)
    batch_normalization_84 = BatchNormalization(name = 'batch_normalization_84', axis = 3, scale = False)(conv2d_84)
    activation_84   = Activation('relu')(batch_normalization_84)
    concatenate_1   = layers.concatenate(name = 'concatenate_1', inputs = [activation_83, activation_84])
    conv2d_78       = Conv2D(filters = 384, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed8)
    batch_normalization_78 = BatchNormalization(name = 'batch_normalization_78', axis = 3, scale = False)(conv2d_78)
    activation_78   = Activation('relu')(batch_normalization_78)
    conv2d_79       = Conv2D(filters = 384, kernel_size = (1, 3), strides = (1, 1), padding = 'same', use_bias = False)(activation_78)
    batch_normalization_79 = BatchNormalization(name = 'batch_normalization_79', axis = 3, scale = False)(conv2d_79)
    activation_79   = Activation('relu')(batch_normalization_79)
    conv2d_80       = Conv2D(filters = 384, kernel_size = (3, 1), strides = (1, 1), padding = 'same', use_bias = False)(activation_78)
    batch_normalization_80 = BatchNormalization(name = 'batch_normalization_80', axis = 3, scale = False)(conv2d_80)
    activation_80   = Activation('relu')(batch_normalization_80)
    mixed9_0        = layers.concatenate(name = 'mixed9_0', inputs = [activation_79, activation_80])
    average_pooling2d_8 = AveragePooling2D(name = 'average_pooling2d_8', pool_size = (3, 3), strides = (1, 1), padding = 'same')(mixed8)
    conv2d_85       = Conv2D(filters = 192, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(average_pooling2d_8)
    batch_normalization_85 = BatchNormalization(name = 'batch_normalization_85', axis = 3, scale = False)(conv2d_85)
    activation_85   = Activation('relu')(batch_normalization_85)
    conv2d_77       = Conv2D(filters = 320, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed8)
    batch_normalization_77 = BatchNormalization(name = 'batch_normalization_77', axis = 3, scale = False)(conv2d_77)
    activation_77   = Activation('relu')(batch_normalization_77)
    mixed9          = layers.concatenate(name = 'mixed9', inputs = [activation_77, mixed9_0, concatenate_1, activation_85])
    conv2d_90       = Conv2D(filters = 448, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed9)
    batch_normalization_90 = BatchNormalization(name = 'batch_normalization_90', axis = 3, scale = False)(conv2d_90)
    activation_90   = Activation('relu')(batch_normalization_90)
    conv2d_91       = Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = False)(activation_90)
    batch_normalization_91 = BatchNormalization(name = 'batch_normalization_91', axis = 3, scale = False)(conv2d_91)
    activation_91   = Activation('relu')(batch_normalization_91)
    conv2d_92       = Conv2D(filters = 384, kernel_size = (1, 3), strides = (1, 1), padding = 'same', use_bias = False)(activation_91)
    batch_normalization_92 = BatchNormalization(name = 'batch_normalization_92', axis = 3, scale = False)(conv2d_92)
    activation_92   = Activation('relu')(batch_normalization_92)
    conv2d_93       = Conv2D(filters = 384, kernel_size = (3, 1), strides = (1, 1), padding = 'same', use_bias = False)(activation_91)
    batch_normalization_93 = BatchNormalization(name = 'batch_normalization_93', axis = 3, scale = False)(conv2d_93)
    activation_93   = Activation('relu')(batch_normalization_93)
    concatenate_2   = layers.concatenate(name = 'concatenate_2', inputs = [activation_92, activation_93])
    conv2d_87       = Conv2D(filters = 384, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed9)
    batch_normalization_87 = BatchNormalization(name = 'batch_normalization_87', axis = 3, scale = False)(conv2d_87)
    activation_87   = Activation('relu')(batch_normalization_87)
    conv2d_88       = Conv2D(filters = 384, kernel_size = (1, 3), strides = (1, 1), padding = 'same', use_bias = False)(activation_87)
    batch_normalization_88 = BatchNormalization(name = 'batch_normalization_88', axis = 3, scale = False)(conv2d_88)
    activation_88   = Activation('relu')(batch_normalization_88)
    conv2d_89       = Conv2D(filters = 384, kernel_size = (3, 1), strides = (1, 1), padding = 'same', use_bias = False)(activation_87)
    batch_normalization_89 = BatchNormalization(name = 'batch_normalization_89', axis = 3, scale = False)(conv2d_89)
    activation_89   = Activation('relu')(batch_normalization_89)
    mixed9_1        = layers.concatenate(name = 'mixed9_1', inputs = [activation_88, activation_89])
    average_pooling2d_9 = AveragePooling2D(name = 'average_pooling2d_9', pool_size = (3, 3), strides = (1, 1), padding = 'same')(mixed9)
    conv2d_94       = Conv2D(filters = 192, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(average_pooling2d_9)
    batch_normalization_94 = BatchNormalization(name = 'batch_normalization_94', axis = 3, scale = False)(conv2d_94)
    activation_94   = Activation('relu')(batch_normalization_94)
    conv2d_86       = Conv2D(filters = 320, kernel_size = (1, 1), strides = (1, 1), padding = 'same', use_bias = False)(mixed9)
    batch_normalization_86 = BatchNormalization(name = 'batch_normalization_86', axis = 3, scale = False)(conv2d_86)
    activation_86   = Activation('relu')(batch_normalization_86)
    mixed10         = layers.concatenate(name = 'mixed10', inputs = [activation_86, mixed9_1, concatenate_2, activation_94])
    avg_pool        = GlobalAveragePooling2D()(mixed10)
    predictions     = Dense(units = 1000, use_bias = True)(avg_pool)
    predictions_activation = Activation('softmax')(predictions)
    model           = Model(inputs = [input_1], outputs = [predictions_activation])


    # load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
        if include_top:
            weights_path = get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='9a0d58056eeedaa3f26cb7ebd46da564')
        else:
            weights_path = get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='bcbd6486424b2319ff4ef7d526e38f63')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            convert_all_kernels_in_model(model)

    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    model = keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet')

    # save as JSON
    json_string = model.to_json()
    with open("inception_v3.json", "w") as of:
        of.write(json_string)

    preds = model.predict(x)
    print('Src Predicted:', decode_predictions(preds))
    
    kit_model = InceptionV3(include_top=True, weights='imagenet') 
    preds = kit_model.predict(x)
    print('Kit Predicted:', decode_predictions(preds))
