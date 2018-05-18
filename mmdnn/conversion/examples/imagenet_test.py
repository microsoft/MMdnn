#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import absolute_import
import argparse
import numpy as np
from six import text_type as _text_type
from tensorflow.contrib.keras.api.keras.preprocessing import image


class TestKit(object):

    truth = {
        'caffe' : {
            'alexnet'        : [(821, 0.25088307), (657, 0.20857951), (744, 0.096812263), (595, 0.066312768), (847, 0.053720973)],
            'vgg19'          : [(21, 0.37522122), (144, 0.28500062), (23, 0.099720284), (134, 0.036305398), (22, 0.033559237)],
            'inception_v1'   : [(21, 0.93591732), (23, 0.037170019), (22, 0.014315935), (128, 0.005050648), (749, 0.001965977)],
            'resnet152'      : [(144, 0.93159181), (23, 0.033074539), (21, 0.028599562), (99, 0.001878676), (146, 0.001557963)],
            'squeezenet'     : [(21, 0.5285601), (128, 0.071685813), (144, 0.064104252), (416, 0.050044473), (22, 0.049522042)]
        },
        'tensorflow' : {
            'vgg19'             : [(21, 11.285443), (144, 10.240093), (23, 9.1792336), (22, 8.1113129), (128, 8.1065922)],
            'resnet'            : [(22, 11.756789), (147, 8.5718527), (24, 6.1751032), (88, 4.3121386), (141, 4.1778097)],
            'resnet_v1_101'     : [(21, 14.384739), (23, 14.262486), (144, 14.068737), (94, 12.17205), (134, 12.064575)],
            'resnet_v2_152'     : [(22, 13.370557), (147, 8.807369), (24, 5.702235), (90, 5.6126657), (95, 4.8026266)],
            'inception_v3'      : [(22, 9.4921198), (24, 4.0932288), (25, 3.700398), (23, 3.3715961), (147, 3.3620636)],
            'mobilenet_v1_1.0'         : [(22, 16.223597), (24, 14.54775), (147, 13.173758), (145, 11.36431), (728, 11.083847)],
            'mobilenet_v2_1.0_224' : [(22, 9.384777), (147, 5.865254), (23, 5.5761757), (750, 5.0572333), (132, 4.865659)]

        },
        'keras' : {
            'vgg16'             : [(21, 0.81199354), (562, 0.019326132), (23, 0.018279659), (144, 0.012460723), (22, 0.012429929)],
            'vgg19'             : [(21, 0.37522098), (144, 0.28500044), (23, 0.099720411), (134, 0.036305476), (22, 0.033559218)],
            'inception_v3'      : [(21, 0.91967654), (23, 0.0029040477), (24, 0.0020232804), (146, 0.0019062747), (22, 0.0017500133)],
            'xception'          : [(21, 0.67462814), (23, 0.063138723), (87, 0.028424012), (89, 0.02484037), (88, 0.0062591862)],
            'mobilenet'         : [(21, 0.7869994), (23, 0.14728773), (146, 0.037277445), (144, 0.0061039869), (727, 0.0046111974)],
            'resnet'            : [(144, 0.80301273), (23, 0.067478567), (21, 0.046560187), (562, 0.037413299), (146, 0.015967956)],
            'inception_resnet_v2'  : [(21, 0.93837249), (87, 0.0021177295), (146, 0.0019775454), (23, 0.00072135136), (24, 0.00056668324)],
            'densenet'          : [(21, 0.86279225), (146, 0.051543437), (23, 0.030489875), (144, 0.028583106), (141, 0.003564599)],
            'nasnet'            : [(21, 0.8541155), (22, 0.0030572189), (146, 0.0026522065), (23, 0.0020259875), (88, 0.0020091296)]
        },
        'mxnet' : {
            'vgg19'                         : [(21, 0.54552644), (144, 0.19179004), (23, 0.066389613), (22, 0.022819581), (128, 0.02271222)],
            'resnet'                        : [(21, 0.84012794), (144, 0.097428247), (23, 0.039757393), (146, 0.010432643), (99, 0.0023797606)],
            'squeezenet'                    : [(21, 0.36026478), (128, 0.084114805), (835, 0.07940048), (144, 0.057378717), (749, 0.053491514)],
            'inception_bn'                  : [(21, 0.84332663), (144, 0.041747514), (677, 0.021810319), (973, 0.02054958), (115, 0.008529461)],
            'resnet152-11k'                 : [(1278, 0.49073416), (1277, 0.21393695), (282, 0.12980066), (1282, 0.0663582), (1224, 0.022041745)],
            'imagenet1k-resnext-101-64x4d'  : [(21, 0.587986), (23, 0.29983738), (862, 0.044453762), (596, 0.00983246), (80, 0.00465048)],
            'imagenet1k-resnext-50'         : [(396, 0.7104751), (398, 0.122665755), (438, 0.06391319), (440, 0.029796895), (417, 0.019492012)],
            'resnext'                       : [(21, 0.58798772), (23, 0.29983655), (862, 0.044453178), (596, 0.0098323636), (80, 0.0046504852)]
        },
        'pytorch' : {
            'resnet18'  : [(394, 10.310125), (395, 9.2285385), (21, 8.9611788), (144, 8.3729601), (749, 7.9692998)],
            'resnet152' : [(21, 13.080057), (141, 12.32998), (94, 9.8761454), (146, 9.3761511), (143, 8.9194641)],
            'vgg19'     : [(821, 8.4734678), (562, 8.3472366), (835, 8.2712851), (749, 7.792901), (807, 6.6604013)],
        },

        'cntk' : {
            'alexnet'       : [(836, 7.5413785), (837, 7.076382), (84, 6.9632936), (148, 6.90293), (416, 6.571906)],
            'resnet18'      : [(21, 8.2490816), (22, 7.7600741), (23, 7.4341722), (148, 7.1398726), (144, 6.9187264)],
            'resnet152'     : [(21, 12.461424), (99, 12.38283), (144, 11.1572275), (94, 10.569823), (146, 10.096423)],
            'inception_v3'  : [(21, 15.558625), (22, 9.7712708), (23, 9.6847782), (146, 9.188818), (144, 8.0436306)]
        },
        'coreml' : {
            'mobilenet' : [],
        },

        'darknet' : {
            'yolov3'        :[],
        },

    }

    preprocess_func = {
        'caffe' : {
            'alexnet'       : lambda path : TestKit.ZeroCenter(path, 227, True),
            'vgg19'         : lambda path : TestKit.ZeroCenter(path, 224, True),
            'inception_v1'  : lambda path : TestKit.ZeroCenter(path, 224, True),
            'resnet152'     : lambda path : TestKit.ZeroCenter(path, 224, True),
            'squeezenet'    : lambda path : TestKit.ZeroCenter(path, 227),
            'inception_v4'  : lambda path : TestKit.Standard(path, 299, True),
            'xception'      : lambda path : TestKit.Standard(path, 299, True),
            'voc-fcn8s'     : lambda path : TestKit.ZeroCenter(path, 500, True),
            'voc-fcn16s'    : lambda path : TestKit.ZeroCenter(path, 500, True),
            'voc-fcn32s'    : lambda path : TestKit.ZeroCenter(path, 500, True),
        },

        'tensorflow' : {
            'vgg16'         : lambda path : TestKit.ZeroCenter(path, 224),
            'vgg19'         : lambda path : TestKit.ZeroCenter(path, 224),
            'inception_v1'  : lambda path : TestKit.Standard(path, 224),
            'inception_v3'  : lambda path : TestKit.Standard(path, 299),
            'resnet'        : lambda path : TestKit.Standard(path, 299),
            'resnet_v1_50'  : lambda path : TestKit.ZeroCenter(path, 224),
            'resnet_v1_101' : lambda path : TestKit.ZeroCenter(path, 224),
            'resnet_v1_152' : lambda path : TestKit.ZeroCenter(path, 224),
            'resnet_v2_50'  : lambda path : TestKit.Standard(path, 299),
            'resnet_v2_152' : lambda path : TestKit.Standard(path, 299),
            'resnet_v2_200' : lambda path : TestKit.Standard(path, 299),
            'resnet152'     : lambda path : TestKit.Standard(path, 299),
            'mobilenet_v1_1.0'  : lambda path : TestKit.Standard(path, 224),
            'mobilenet_v1_0.50' : lambda path : TestKit.Standard(path, 224),
            'mobilenet_v1_0.25' : lambda path : TestKit.Standard(path, 224),
            'mobilenet'     : lambda path : TestKit.Standard(path, 224),
            'mobilenet_v2_1.0_224'  : lambda path : TestKit.Standard(path, 224),
            'nasnet-a_large'     : lambda path : TestKit.Standard(path, 331),
            'inception_resnet_v2' : lambda path : TestKit.Standard(path, 299),
        },

        'keras' : {
            'vgg16'                : lambda path : TestKit.ZeroCenter(path, 224, True),
            'vgg19'                : lambda path : TestKit.ZeroCenter(path, 224, True),
            'inception_v3'         : lambda path : TestKit.Standard(path, 299),
            'resnet50'             : lambda path : TestKit.ZeroCenter(path, 224, True),
            'xception'             : lambda path : TestKit.Standard(path, 299),
            'mobilenet'            : lambda path : TestKit.Standard(path, 224),
            'inception_resnet_v2'  : lambda path : TestKit.Standard(path, 299),
            'densenet'             : lambda path : TestKit.Standard(path, 224),
            'nasnet'               : lambda path : TestKit.Standard(path, 331),
            'yolo2-tiny'           : lambda path : TestKit.Identity(path, 416),
            'yolo2'                : lambda path : TestKit.Identity(path, 416),
        },

        'mxnet' : {
            'vgg16'                         : lambda path : TestKit.ZeroCenter(path, 224, False),
            'vgg19'                         : lambda path : TestKit.ZeroCenter(path, 224, False),
            'resnet'                        : lambda path : TestKit.Identity(path, 224, True),
            'squeezenet_v1.0'               : lambda path : TestKit.ZeroCenter(path, 224, False),
            'squeezenet_v1.1'               : lambda path : TestKit.ZeroCenter(path, 224, False),
            'imagenet1k-inception-bn'       : lambda path : TestKit.Identity(path, 224, False),
            'imagenet1k-resnet-18'          : lambda path : TestKit.Identity(path, 224, True),
            'imagenet1k-resnet-152'         : lambda path : TestKit.Identity(path, 224, True),
            'resnext'                       : lambda path : TestKit.Identity(path, 224, False),
            'imagenet1k-resnext-50'         : lambda path : TestKit.Identity(path, 224, False),
            'imagenet1k-resnext-101-64x4d'  : lambda path : TestKit.Identity(path, 224, False),
        },

        'pytorch' : {
            'alexnet'       : lambda path : TestKit.Standard(path, 227),
            'densenet121'   : lambda path : TestKit.Standard(path, 224),
            'densenet169'   : lambda path : TestKit.Standard(path, 224),
            'densenet161'   : lambda path : TestKit.Standard(path, 224),
            'densenet201'   : lambda path : TestKit.Standard(path, 224),
            'vgg11'         : lambda path : TestKit.Standard(path, 224),
            'vgg13'         : lambda path : TestKit.Standard(path, 224),
            'vgg16'         : lambda path : TestKit.Standard(path, 224),
            'vgg19'         : lambda path : TestKit.Standard(path, 224),
            'vgg11_bn'         : lambda path : TestKit.Standard(path, 224),
            'vgg13_bn'         : lambda path : TestKit.Standard(path, 224),
            'vgg16_bn'         : lambda path : TestKit.Standard(path, 224),
            'vgg19_bn'         : lambda path : TestKit.Standard(path, 224),
            'resnet18'      : lambda path : TestKit.Standard(path, 224),
            'resnet34'      : lambda path : TestKit.Standard(path, 224),
            'resnet50'      : lambda path : TestKit.Standard(path, 224),
            'resnet101'      : lambda path : TestKit.Standard(path, 224),
            'resnet152'     : lambda path : TestKit.Standard(path, 224),
            'squeezenet1_0' : lambda path : TestKit.Standard(path, 224),
            'inception_v3'  : lambda path : TestKit.Standard(path, 299),
        },

        'cntk' : {
            'alexnet'       : lambda path : TestKit.Identity(path, 227),
            'resnet18'      : lambda path : TestKit.Identity(path, 224),
            'resnet152'     : lambda path : TestKit.Identity(path, 224),
            'inception_v3'  : lambda path : TestKit.Identity(path, 299),
        },


        'darknet' : {
             'yolov3'        : lambda path : TestKit.Identity(path, 416),
             'yolov2'        : lambda path : TestKit.Identity(path, 416),
        },


        'coreml' : {
            'mobilenet'         : lambda path :  TestKit.Normalize(path, 224, 0.0170000009239, [-2.10256004333, -1.98526000977, -1.76698005199], [1.0, 1.0, 1.0], True),
            'inception_v3'      : lambda path : TestKit.Standard(path, 299),
            'vgg16'             : lambda path : TestKit.ZeroCenter(path, 224, True),
            'resnet50'          : lambda path : TestKit.ZeroCenter(path, 224, True),
            'tinyyolo'          : lambda path : TestKit.Normalize(path, 416, 0.00392156863, [0, 0, 0], [1.0, 1.0, 1.0], False),
        }

    }

    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('-p', '--preprocess', type=_text_type, help='Model Preprocess Type')

        parser.add_argument('-n', type=_text_type, default='kit_imagenet',
                            help='Network structure file name.')

        parser.add_argument('-s', type=_text_type, help='Source Framework Type',
                            choices=self.truth.keys())

        parser.add_argument('-w', type=_text_type, required=True,
                            help='Network weights file name')

        parser.add_argument('--image', '-i',
                            type=_text_type, help='Test image path.',
                            default="mmdnn/conversion/examples/data/seagull.jpg"
        )

        parser.add_argument('-l', '--label',
                            type=_text_type,
                            default='mmdnn/conversion/examples/data/imagenet_1000.txt',
                            help='Path of label.')

        parser.add_argument('--dump',
            type=_text_type,
            default=None,
            help='Target model path.')

        parser.add_argument('--detect',
            type=_text_type,
            default=None,
            help='Model detection result path.')


        self.args = parser.parse_args()
        if self.args.n.endswith('.py'):
            self.args.n = self.args.n[:-3]
        self.MainModel = __import__(self.args.n)


    @staticmethod
    def ZeroCenter(path, size, BGRTranspose=False):
        img = image.load_img(path, target_size = (size, size))
        x = image.img_to_array(img)

        # Reference: 1) Keras image preprocess: https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
        #            2) tensorflow github issue: https://github.com/tensorflow/models/issues/517
        # R-G-B for Imagenet === [123.68, 116.78, 103.94]


        x[..., 0] -= 123.68
        x[..., 1] -= 116.779
        x[..., 2] -= 103.939

        if BGRTranspose == True:
            x = x[..., ::-1]

        return x


    @staticmethod
    def Normalize(path, size=224, scale=0.0392156863 ,mean=[-0.485, -0.456, -0.406], std=[0.229, 0.224, 0.225], BGRTranspose = False):
        img = image.load_img(path, target_size=(size, size))
        x = image.img_to_array(img)
        x *= scale
        for i in range(0, 3):
            x[..., i] += mean[i]
            x[..., i] /= std[i]
        if BGRTranspose == True:
            x = x[..., ::-1]
        return x


    @staticmethod
    def Standard(path, size, BGRTranspose=False):
        img = image.load_img(path, target_size = (size, size))
        x = image.img_to_array(img)
        x /= 255.0
        x -= 0.5
        x *= 2.0
        if BGRTranspose == True:
            x = x[..., ::-1]
        return x


    @staticmethod
    def Identity(path, size, BGRTranspose=False):
        img = image.load_img(path, target_size = (size, size))
        x = image.img_to_array(img)
        if BGRTranspose == True:
            x = x[..., ::-1]
        return x


    def preprocess(self, image_path):
        func = self.preprocess_func[self.args.s][self.args.preprocess]
        return func(image_path)


    def print_result(self, predict):
        predict = np.squeeze(predict)
        if predict.ndim == 1:
            top_indices = predict.argsort()[-5:][::-1]
            if predict.shape[0] == 1001 or predict.shape[0] == 1000:
                if predict.shape[0] == 1000:
                    offset = 0
                else:
                    offset = 1

                import os
                if os.path.exists(self.args.label):
                    with open(self.args.label, 'r') as f:
                        labels = [l.rstrip() for l in f]

                for i in top_indices:
                    print (labels[i - offset], i, predict[i])

            self.result = [(i, predict[i]) for i in top_indices]

        else:
            self.result = predict
            print (self.result)


    @staticmethod
    def print_intermediate_result(intermediate_output, if_transpose=False):
        intermediate_output = np.squeeze(intermediate_output)

        if if_transpose == True:
            intermediate_output = np.transpose(intermediate_output, [2, 0, 1])

        print (intermediate_output)
        print (intermediate_output.shape)
        print ("Sum = %.30f" % np.sum(intermediate_output))
        print ("Std = %.30f" % np.std(intermediate_output))


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
        raise NotImplementedError()


'''
if __name__=='__main__':
    tester = TestKit()
    tester.inference('examples/data/seagull.jpg')
'''
