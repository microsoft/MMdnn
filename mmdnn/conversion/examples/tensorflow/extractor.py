#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import absolute_import

import os
import tensorflow as tf

from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_v2
from mmdnn.conversion.examples.tensorflow.models import inception_resnet_v2
from mmdnn.conversion.examples.tensorflow.models import mobilenet_v1
from mmdnn.conversion.examples.tensorflow.models import nasnet
from mmdnn.conversion.examples.tensorflow.models.mobilenet import mobilenet_v2
slim = tf.contrib.slim

from mmdnn.conversion.examples.imagenet_test import TestKit
from mmdnn.conversion.examples.extractor import base_extractor
from mmdnn.conversion.common.utils import download_file


class tensorflow_extractor(base_extractor):

    architecture_map = {
        'vgg16' : {
            'url'         : 'http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz',
            'filename'    : 'vgg_16.ckpt',
            'builder'     : lambda : vgg.vgg_16,
            'arg_scope'   : vgg.vgg_arg_scope,
            'input'       : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 224, 224, 3]),
            'num_classes' : 1000,
        },
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
        'inception_v1_frozen' : {
            'url'         : 'https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz',
            'filename'    : 'inception_v1_2016_08_28_frozen.pb',
            'tensor_out'  : 'InceptionV1/Logits/Predictions/Reshape_1:0',
            'tensor_in'   : 'input:0',
            'input_shape' : [224, 224, 3],
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
        'inception_v3_frozen' : {
            'url'         : 'https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz',
            'filename'    : 'inception_v3_2016_08_28_frozen.pb',
            'tensor_out'  : 'InceptionV3/Predictions/Softmax:0',
            'tensor_in'   : 'input:0',
            'input_shape' : [299, 299, 3],
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
        'mobilenet_v1_1.0_frozen' : {
            'url'         : 'https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz',
            'filename'    : 'mobilenet_v1_1.0_224/frozen_graph.pb',
            'tensor_out'  : 'MobilenetV1/Predictions/Softmax:0',
            'tensor_in'   : 'input:0',
            'input_shape' : [224, 224, 3],
            'num_classes' : 1001,
        },
        'mobilenet_v2_1.0_224':{
            'url'         : 'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz',
            'filename'    : 'mobilenet_v2_1.0_224.ckpt',
            'builder'     : lambda : mobilenet_v2.mobilenet,
            'arg_scope'   : mobilenet_v2.training_scope,
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

            if logits.op.type == 'Squeeze':
                labels = tf.identity(logits, name='MMdnn_Output')
            else:
                labels = tf.squeeze(logits, name='MMdnn_Output')


        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # tf.train.export_meta_graph("kit.meta", as_text=True)
            # writer = tf.summary.FileWriter('./graphs', sess.graph)
            # writer.close()
            sess.run(init)
            saver = tf.train.Saver()
            saver.restore(sess, path + cls.architecture_map[architecture]['filename'])
            save_path = saver.save(sess, path + "imagenet_{}.ckpt".format(architecture))
            print("Model saved in file: %s" % save_path)

        import tensorflow.contrib.keras as keras
        keras.backend.clear_session()


    @classmethod
    def handle_frozen_graph(cls, architecture, path):
        return
        # raise NotImplementedError()

    @classmethod
    def get_frozen_para(cls, architecture):
        frozenname = architecture + '_frozen'
        return cls.architecture_map[frozenname]['filename'], cls.architecture_map[frozenname]['input_shape'], cls.architecture_map[frozenname]['tensor_in'], cls.architecture_map[frozenname]['tensor_out']

    @classmethod
    def download(cls, architecture, path="./"):
        if cls.sanity_check(architecture):
            architecture_file = download_file(cls.architecture_map[architecture]['url'], directory=path, auto_unzip=True)
            if not architecture_file:
                return None

            tf.reset_default_graph()

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
    def inference(cls, architecture, files, path, image_path, is_frozen=False):
        if is_frozen:
            architecture_ = architecture + "_frozen"
        else:
            architecture_ = architecture

        if cls.download(architecture_, path):
            import numpy as np
            func = TestKit.preprocess_func['tensorflow'][architecture]
            img = func(image_path)
            img = np.expand_dims(img, axis=0)

            if is_frozen:
                tf_model_path = cls.architecture_map[architecture_]['filename']
                with open(path + tf_model_path, 'rb') as f:
                    serialized = f.read()
                tf.reset_default_graph()
                original_gdef = tf.GraphDef()
                original_gdef.ParseFromString(serialized)
                tf_output_name =  cls.architecture_map[architecture_]['tensor_out']
                tf_input_name =  cls.architecture_map[architecture_]['tensor_in']

                with tf.Graph().as_default() as g:
                    tf.import_graph_def(original_gdef, name='')
                with tf.Session(graph = g) as sess:
                    tf_out = sess.run(tf_output_name, feed_dict={tf_input_name: img})
                predict = np.squeeze(tf_out)
                return predict

            else:
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

