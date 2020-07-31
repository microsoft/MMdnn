#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import absolute_import
import os
import keras
from keras import backend as K
from mmdnn.conversion.examples.imagenet_test import TestKit
from mmdnn.conversion.examples.extractor import base_extractor
from mmdnn.conversion.common.utils import download_file


class keras_extractor(base_extractor):

    MMDNN_BASE_URL = 'http://mmdnn.eastasia.cloudapp.azure.com:89/models/'

    architecture_map = {
        'inception_v3'        : lambda : keras.applications.inception_v3.InceptionV3(input_shape=(299, 299, 3)),
        'vgg16'               : lambda : keras.applications.vgg16.VGG16(),
        'vgg19'               : lambda : keras.applications.vgg19.VGG19(),
        'resnet50'            : lambda : keras.applications.resnet50.ResNet50(),
        'mobilenet'           : lambda : keras.applications.mobilenet.MobileNet(),
        'xception'            : lambda : keras.applications.xception.Xception(input_shape=(299, 299, 3)),
        'inception_resnet_v2' : lambda : keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=(299, 299, 3)),
        'densenet'            : lambda : keras.applications.densenet.DenseNet201(),
        'nasnet'              : lambda : keras.applications.nasnet.NASNetLarge(),
    }

    thirdparty_map = {
        'yolo2'    : MMDNN_BASE_URL + 'keras/yolo2.h5',
    }

    image_size = {
        'inception_v3'      : 299,
        'vgg16'             : 224,
        'vgg19'             : 224,
        'resnet'            : 224,
        'mobilenet'         : 224,
        'xception'          : 299,
        'inception_resnet'  : 299,
        'densenet'          : 224,
        'nasnet'            : 331,
    }

    @classmethod
    def help(cls):
        print('Supported models: {}'.format(set().union(cls.architecture_map.keys(), cls.thirdparty_map.keys())))


    @classmethod
    def download(cls, architecture, path="./"):
        if architecture in cls.thirdparty_map:
            weight_file = download_file(cls.thirdparty_map[architecture], directory=path)
            return weight_file

        elif cls.sanity_check(architecture):
            output_filename = path + 'imagenet_{}.h5'.format(architecture)
            if os.path.exists(output_filename) == False:
                model = cls.architecture_map[architecture]()
                model.save(output_filename)
                print("Keras model {} is saved in [{}]".format(architecture, output_filename))
                K.clear_session()
                del model
                return output_filename

            else:
                print("File [{}] existed, skip download.".format(output_filename))
                return output_filename

        else:
            return None


    @classmethod
    def inference(cls, architecture, files, path, image_path):
        if architecture in cls.thirdparty_map:
            model = keras.models.load_model(files)

        elif cls.sanity_check(architecture):
            model = cls.architecture_map[architecture]()

        else:
            model = None

        if model:
            import numpy as np
            func = TestKit.preprocess_func['keras'][architecture]
            img = func(image_path)
            img = np.expand_dims(img, axis=0)
            predict = model.predict(img)
            predict = np.squeeze(predict)
            K.clear_session()
            del model
            return predict

        else:
            return None
