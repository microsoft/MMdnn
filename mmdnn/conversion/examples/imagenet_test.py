#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import argparse
import numpy as np
import sys
import os
from six import text_type as _text_type
from tensorflow.contrib.keras.python.keras.preprocessing import image
import tensorflow as tf


class TestKit(object):

    truth = {
        'caffe' : {
            'vgg19'          : [(386, 0.78324348), (101, 0.19303614), (385, 0.018230435), (347, 0.0021320845), (348, 0.0013114288)],
            'inception_v1'   : [(386, 0.74765885), (101, 0.21649569), (385, 0.035726305), (343, 2.2025948e-05), (346, 1.609625e-05)]
        },
        'tf' : {
            'vgg19'             : [(386, 21.274265), (101, 20.239922), (385, 18.051809), (347, 15.064944), (354, 14.070037)],
            'resnet'            : [(387, 14.552186), (102, 11.523594), (386, 7.2283664), (500, 4.6292458), (899, 2.8113561)],
            'inception_v3'      : [(387, 10.452494), (102, 7.0714035), (386, 4.9622779), (341, 1.9631921), (685, 1.6739436)],
            'mobilenet'         : [(387, 22.832821), (102, 21.173042), (386, 16.660761), (349, 13.075641), (350, 10.205788)]
        },
        'keras' : {
            'vgg19'          : [(386, 0.78324348), (101, 0.19303614), (385, 0.018230435), (347, 0.0021320845), (348, 0.0013114288)],
            'inception_v3'   : [(386, 0.94166446), (101, 0.029935108), (385, 0.0025184087), (340, 0.00017132676), (684, 0.00014733514)],
            'xception'       : [(386, 0.62978429), (101, 0.25602135), (385, 0.015696181), (340, 0.00043122924), (615, 0.00037536205)],
            'mobilenet'      : [(386, 0.83868736), (101, 0.15950277), (385, 0.0017502838), (348, 4.8541253e-05), (349, 2.7526441e-06)],
            'resnet'         : [(386, 0.49285111), (101, 0.46214512), (385, 0.034884922), (354, 0.0036463451), (348, 0.003200819)]
        },
        'mxnet' : {
            'vgg19'          : [(386, 0.71452385), (101, 0.25398338), (385, 0.028478812), (347, 0.0014366163), (354, 0.00053119892)],
            'resnet'         : [(386, 0.68158048), (101, 0.27469227), (385, 0.038434178), (347, 0.0027639084), (348, 0.00042860108)],
            'squeezenet'     : [(386, 0.98810065), (101, 0.0090002166), (385, 0.0027751704), (354, 8.5944812e-05), (348, 1.8317563e-05)],
            'inception_bn'   : [(386, 0.59537756), (101, 0.36962193), (385, 0.034420349), (354, 0.00017646443), (347, 0.00015946048)]
        }
    }

    preprocess_func = {
        'caffe' : {
            'vgg19'         : lambda path : TestKit.ZeroCenter(path, 224, True),
            'inception_v1'  : lambda path : TestKit.ZeroCenter(path, 224, True)
        },

        'tf' : {
            'vgg19'         : lambda path : TestKit.ZeroCenter(path, 224, False),
            'inception_v3'  : lambda path : TestKit.Standard(path, 299),
            'resnet'        : lambda path : TestKit.Standard(path, 299),
            'mobilenet'     : lambda path : TestKit.Standard(path, 224)
        },

        'keras' : {
            'vgg19'         : lambda path : TestKit.ZeroCenter(path, 224, True),
            'inception_v3'  : lambda path : TestKit.Standard(path, 299),
            'resnet'        : lambda path : TestKit.ZeroCenter(path, 224, True),
            'xception'      : lambda path : TestKit.Standard(path, 299),
            'mobilenet'     : lambda path : TestKit.Standard(path, 224),
        },

        'mxnet' : {
            'vgg19'         : lambda path : TestKit.ZeroCenter(path, 224, False),
            'resnet'        : lambda path : TestKit.Identity(path, 224)
        }
    }

    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('-p', '--preprocess',
            type = _text_type, help='Model Preprocess Type')

        parser.add_argument('-n',
            type = _text_type, default = 'kit_imagenet', help = 'Network structure file name.')

        parser.add_argument('-s',
            type = _text_type, choices = ["caffe", "tf", "keras", "cntk", "mxnet"], help = 'Source Framework Type')

        parser.add_argument('-w',
            type = _text_type, help = 'Network weights file name', required = True)

        parser.add_argument('--image', '-i',
            type = _text_type,
            default = "mmdnn/conversion/examples/data/elephant.jpg",
            help = 'Test image path.'
        )
        
        parser.add_argument('--dump',
            type = _text_type,
            default = None,
            help = 'Target model path.')
        
        self.args = parser.parse_args()
        if self.args.n.endswith('.py'):
            self.args.n = self.args.n[:-3]            
        self.MainModel = __import__(self.args.n)


    @staticmethod
    def ZeroCenter(path, size, BGRTranspose = False):
        img = image.load_img(path, target_size = (size, size))
        x = image.img_to_array(img)
        if BGRTranspose == True:
            x = x[..., ::-1]
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68
        return x


    @staticmethod
    def Standard(path, size):
        img = image.load_img(path, target_size = (size, size))
        x = image.img_to_array(img)
        x /= 255.0
        x -= 0.5
        x *= 2.0
        return x

    
    @staticmethod
    def Identity(path, size):
        img = image.load_img(path, target_size = (size, size))
        x = image.img_to_array(img)
        return x


    def preprocess(self, image_path):        
        func = self.preprocess_func[self.args.s][self.args.preprocess]
        return func(image_path)

    
    def print_result(self, predict):
        predict = np.squeeze(predict)
        top_indices = predict.argsort()[-5:][::-1]
        self.result = [(i, predict[i]) for i in top_indices]
        print (self.result)

    
    def print_intermediate_result(self, intermediate_output, if_transpose = False):                        
        intermediate_output = np.squeeze(intermediate_output)
        
        if if_transpose == True:
            intermediate_output = np.transpose(intermediate_output, [2, 0, 1])
        
        print (intermediate_output)
        print (intermediate_output.shape)


    def test_truth(self):
        this_truth = self.truth[self.args.s][self.args.preprocess]
        for index, i in enumerate(self.result):            
            assert this_truth[index][0] == i[0]
            assert np.isclose(this_truth[index][1], i[1], atol = 1e-6)

        print ("Test model [{}] from [{}] passed.".format(
            self.args.preprocess,
            self.args.s
        ))
    
    
    def inference(self, image_path):
        self.preprocess(image_path)                
        self.print_result()

    
    def dump(self, path = None):
        raise NotImplementError()


'''
if __name__=='__main__':    
    tester = TestKit()
    tester.inference('examples/data/elephant.jpg')
'''
