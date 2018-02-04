#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import absolute_import
import keras
from mmdnn.conversion.examples.imagenet_test import TestKit
from mmdnn.conversion.examples.extractor import base_extractor


class keras_extractor(base_extractor):

    architecture_map = {
        'inception_v3'        : lambda : keras.applications.inception_v3.InceptionV3(input_shape=(299, 299, 3)),
        'vgg16'               : lambda : keras.applications.vgg16.VGG16(),
        'vgg19'               : lambda : keras.applications.vgg19.VGG19(),
        'resnet50'            : lambda : keras.applications.resnet50.ResNet50(),
        'mobilenet'           : lambda : keras.applications.mobilenet.MobileNet(),
        'xception'            : lambda : keras.applications.xception.Xception(input_shape=(299, 299, 3)),
        'inception_resnet_v2' : lambda : keras.applications.inception_resnet_v2.InceptionResNetV2(),
        'densenet'            : lambda : keras.applications.densenet.DenseNet201(),
        'nasnet'              : lambda : keras.applications.nasnet.NASNetLarge()
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
    def download(cls, architecture):
        if cls.sanity_check(architecture):
            model = cls.architecture_map[architecture]()

            model.save('imagenet_{}.h5'.format(architecture))
            print("Keras model {} is saved as [imagenet_{}.h5]".format(architecture, architecture))

            # # save network structure as JSON
            # json_string = model.to_json()
            # with open("imagenet_{}.json".format(architecture), "w") as of:
            #     of.write(json_string)
            # print("Network structure is saved as [imagenet_{}.json].".format(architecture))

            # model.save_weights('imagenet_{}.h5'.format(architecture))
            # print("Network weights are saved as [imagenet_{}.h5].".format(architecture))

        else:
            return False


    @classmethod
    def inference(cls, architecture, image_path):
        if cls.sanity_check(architecture):
            model = cls.architecture_map[architecture]()
            import numpy as np
            func = TestKit.preprocess_func['keras'][architecture]
            img = func(image_path)
            img = np.expand_dims(img, axis=0)
            predict = model.predict(img)
            predict = np.squeeze(predict)
            return predict

        else:
            return None
