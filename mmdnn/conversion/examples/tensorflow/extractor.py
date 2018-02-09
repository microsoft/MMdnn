#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import absolute_import

import tensorflow as tf

from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_v2
from mmdnn.conversion.examples.tensorflow.models import inception_resnet_v2
from mmdnn.conversion.examples.tensorflow.models import mobilenet_v1
from mmdnn.conversion.examples.tensorflow.models import nasnet
slim = tf.contrib.slim

from mmdnn.conversion.examples.imagenet_test import TestKit
from mmdnn.conversion.examples.extractor import base_extractor
from mmdnn.conversion.common.utils import download_file


class tensorflow_extractor(base_extractor):

    architecture_map = {
        'vgg19' : {
            'url'         : 'http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz',
            'filename'    : 'vgg_19.ckpt',
            'builder'     : lambda : vgg.vgg_19,
            'arg_scope'   : vgg.vgg_arg_scope,
            'input'       : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 224, 224, 3]),
            'num_classes' : 1000,
        },
        'inception_v1' : {
            'url'         : 'http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz',
            'filename'    : 'inception_v1.ckpt',
            'builder'     : lambda : inception.inception_v1,
            'arg_scope'   : inception.inception_v3_arg_scope,
            'input'       : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 224, 224, 3]),
            'num_classes' : 1001,
        },
        'inception_v3' : {
            'url'         : 'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz',
            'filename'    : 'inception_v3.ckpt',
            'builder'     : lambda : inception.inception_v3,
            'arg_scope'   : inception.inception_v3_arg_scope,
            'input'       : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 299, 299, 3]),
            'num_classes' : 1001,
        },
        'resnet_v1_50' : {
            'url'         : 'http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz',
            'filename'    : 'resnet_v1_50.ckpt',
            'builder'     : lambda : resnet_v1.resnet_v1_50,
            'arg_scope'   : resnet_v2.resnet_arg_scope,
            'input'       : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 224, 224, 3]),
            'num_classes' : 1000,
        },
        'resnet_v1_152' : {
            'url'         : 'http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz',
            'filename'    : 'resnet_v1_152.ckpt',
            'builder'     : lambda : resnet_v1.resnet_v1_152,
            'arg_scope'   : resnet_v2.resnet_arg_scope,
            'input'       : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 224, 224, 3]),
            'num_classes' : 1000,
        },
        'resnet_v2_50' : {
            'url'         : 'http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz',
            'filename'    : 'resnet_v2_50.ckpt',
            'builder'     : lambda : resnet_v2.resnet_v2_50,
            'arg_scope'   : resnet_v2.resnet_arg_scope,
            'input'       : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 299, 299, 3]),
            'num_classes' : 1001,
        },
        'resnet_v2_152' : {
            'url'         : 'http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz',
            'filename'    : 'resnet_v2_152.ckpt',
            'builder'     : lambda : resnet_v2.resnet_v2_152,
            'arg_scope'   : resnet_v2.resnet_arg_scope,
            'input'       : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 299, 299, 3]),
            'num_classes' : 1001,
        },
        'resnet_v2_200' : {
            'url'         : 'http://download.tensorflow.org/models/resnet_v2_200_2017_04_14.tar.gz',
            'filename'    : 'resnet_v2_200.ckpt',
            'builder'     : lambda : resnet_v2.resnet_v2_200,
            'arg_scope'   : resnet_v2.resnet_arg_scope,
            'input'       : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 299, 299, 3]),
            'num_classes' : 1001,
        },
        'mobilenet_v1_1.0' : {
            'url'         : 'http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz',
            'filename'    : 'mobilenet_v1_1.0_224.ckpt',
            'builder'     : lambda : mobilenet_v1.mobilenet_v1,
            'arg_scope'   : mobilenet_v1.mobilenet_v1_arg_scope,
            'input'       : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 224, 224, 3]),
            'num_classes' : 1001,
        },
        'inception_resnet_v2' : {
            'url'         : 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz',
            'filename'    : 'inception_resnet_v2_2016_08_30.ckpt',
            'builder'     : lambda : inception_resnet_v2.inception_resnet_v2,
            'arg_scope'   : inception_resnet_v2.inception_resnet_v2_arg_scope,
            'input'       : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 299, 299, 3]),
            'num_classes' : 1001,
        },
        'nasnet-a_large' : {
            'url'         : 'https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz',
            'filename'    : 'model.ckpt',
            'builder'     : lambda : nasnet.build_nasnet_large,
            'arg_scope'   : nasnet.nasnet_large_arg_scope,
            'input'       : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 331, 331, 3]),
            'num_classes' : 1001,
        },
    }


    @classmethod
    def handle_checkpoint(cls, architecture, path):
        with slim.arg_scope(cls.architecture_map[architecture]['arg_scope']()):
            data_input = cls.architecture_map[architecture]['input']()
            logits, endpoints = cls.architecture_map[architecture]['builder']()(
                data_input,
                num_classes=cls.architecture_map[architecture]['num_classes'],
                is_training=False)
            labels = tf.squeeze(logits, name='MMdnn_Output')

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./graphs', sess.graph)
            writer.close()
            sess.run(init)
            saver = tf.train.Saver()
            saver.restore(sess, path + cls.architecture_map[architecture]['filename'])
            save_path = saver.save(sess, path + "imagenet_{}.ckpt".format(architecture))
            print("Model saved in file: %s" % save_path)

        import tensorflow.contrib.keras as keras
        keras.backend.clear_session()


    @classmethod
    def handle_frozen_graph(cls, architecture, path):
        raise NotImplementedError()


    @classmethod
    def download(cls, architecture, path="./"):
        if cls.sanity_check(architecture):
            architecture_file = download_file(cls.architecture_map[architecture]['url'], directory=path, auto_unzip=True)
            if not architecture_file:
                return None

            if cls.architecture_map[architecture]['filename'].endswith('ckpt'):
                cls.handle_checkpoint(architecture, path)

            elif cls.architecture_map[architecture]['filename'].endswith('pb'):
                cls.handle_frozen_graph(architecture, path)

            else:
                raise ValueError("Unknown file name [{}].".format(cls.architecture_map[architecture]['filename']))

            return architecture_file

        else:
            return None


    @classmethod
    def inference(cls, architecture, path, image_path):
        if cls.download(architecture, path):
            import numpy as np
            func = TestKit.preprocess_func['tensorflow'][architecture]
            img = func(image_path)
            img = np.expand_dims(img, axis=0)

            with slim.arg_scope(cls.architecture_map[architecture]['arg_scope']()):
                data_input = cls.architecture_map[architecture]['input']()
                logits, endpoints = cls.architecture_map[architecture]['builder']()(
                    data_input,
                    num_classes=cls.architecture_map[architecture]['num_classes'],
                    is_training=False)
                labels = tf.squeeze(logits)

            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                saver = tf.train.Saver()
                saver.restore(sess, path + cls.architecture_map[architecture]['filename'])
                predict = sess.run(logits, feed_dict = {data_input : img})

            import tensorflow.contrib.keras as keras
            keras.backend.clear_session()

            predict = np.squeeze(predict)
            return predict

        else:
            return None
