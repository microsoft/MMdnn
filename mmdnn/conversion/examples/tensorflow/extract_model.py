#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import argparse
from six import text_type as _text_type
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import vgg
from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from mmdnn.conversion.examples.imagenet_test import TestKit

slim = tf.contrib.slim

input_layer_map = {
    'vgg16'         : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 224, 224, 3]),
    'vgg19'         : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 224, 224, 3]),
    'inception_v1'  : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 224, 224, 3]),
    'inception_v2'  : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 299, 299, 3]),
    'inception_v3'  : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 299, 299, 3]),
    'resnet50'      : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 299, 299, 3]),
    'resnet_v1_101' : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 224, 224, 3]),
    'resnet101'     : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 299, 299, 3]),
    'resnet152'     : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 299, 299, 3]),
    'resnet200'     : lambda : tf.placeholder(name='input', dtype=tf.float32, shape=[None, 299, 299, 3]),
}

arg_scopes_map = {
    'vgg16'         : vgg.vgg_arg_scope,
    'vgg19'         : vgg.vgg_arg_scope,
    'inception_v1'  : inception.inception_v3_arg_scope,
    'inception_v2'  : inception.inception_v3_arg_scope,
    'inception_v3'  : inception.inception_v3_arg_scope,
    'resnet50'      : resnet_v2.resnet_arg_scope,
    'resnet_v1_101' : resnet_v2.resnet_arg_scope,
    'resnet101'     : resnet_v2.resnet_arg_scope,
    'resnet152'     : resnet_v2.resnet_arg_scope,
    'resnet200'     : resnet_v2.resnet_arg_scope,
    # 'mobilenet_v1': mobilenet_v1.mobilenet_v1_arg_scope,
}

networks_map = {
    'vgg16'         : lambda : vgg.vgg_16,
    'vgg19'         : lambda : vgg.vgg_19,
    'inception_v1'  : lambda : inception.inception_v1,
    'inception_v2'  : lambda : inception.inception_v2,
    'inception_v3'  : lambda : inception.inception_v3,
    'resnet_v1_101' : lambda : resnet_v1.resnet_v1_101,
    'resnet50'      : lambda : resnet_v2.resnet_v2_50,
    'resnet101'     : lambda : resnet_v2.resnet_v2_101,
    'resnet152'     : lambda : resnet_v2.resnet_v2_152,
    'resnet200'     : lambda : resnet_v2.resnet_v2_200,
    #'mobilenet_v1' : mobilenet_v1.mobilenet_v1,
}

def _main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--network', type=_text_type, help='Model Type', required=True,
        choices = input_layer_map.keys())

    parser.add_argument('-i', '--image',
        type=_text_type, help='Test Image Path')

    parser.add_argument('-ckpt', '--checkpoint',
        type=_text_type, help='Tensorflow Checkpoint file name', required=True)

    args = parser.parse_args()

    num_classes = 1000 if args.network in ('vgg16', 'vgg19', 'resnet_v1_101') else 1001

    with slim.arg_scope(arg_scopes_map[args.network]()):
        data_input = input_layer_map[args.network]()
        logits, endpoints = networks_map[args.network]()(data_input, num_classes=num_classes, is_training=False)
        labels = tf.squeeze(logits)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        writer.close()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, args.checkpoint)
        save_path = saver.save(sess, "./imagenet_{}.ckpt".format(args.network))
        print("Model saved in file: %s" % save_path)

        if args.image:
            import numpy as np
            func = TestKit.preprocess_func['tensorflow'][args.network]
            img = func(args.image)
            img = np.expand_dims(img, axis = 0)
            predict = sess.run(logits, feed_dict = {data_input : img})
            predict = np.squeeze(predict)
            top_indices = predict.argsort()[-5:][::-1]
            result = [(i, predict[i]) for i in top_indices]
            print (result)


if __name__=='__main__':
    _main()
