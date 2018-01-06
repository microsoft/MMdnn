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
            'resnet152'      : [(144, 0.93159181), (23, 0.033074539), (21, 0.028599562), (99, 0.001878676), (146, 0.001557963)]
        },
        'tensorflow' : {
            'vgg19'             : [(21, 11.285443), (144, 10.240093), (23, 9.1792336), (22, 8.1113129), (128, 8.1065922)],
            'resnet'            : [(22, 11.756789), (147, 8.5718527), (24, 6.1751032), (88, 4.3121386), (141, 4.1778097)],
            'resnet_v1_101'     : [(21, 14.384739), (23, 14.262486), (144, 14.068737), (94, 12.17205), (134, 12.064575)],
            'inception_v3'      : [(22, 9.4921198), (24, 4.0932288), (25, 3.700398), (23, 3.3715961), (147, 3.3620636)],
            'mobilenet'         : [(22, 16.223597), (24, 14.54775), (147, 13.173758), (145, 11.36431), (728, 11.083847)]
        },
        'keras' : {
            'vgg16'             : [(21, 0.81199354), (562, 0.019326132), (23, 0.018279659), (144, 0.012460723), (22, 0.012429929)],
            'vgg19'             : [(21, 0.37522098), (144, 0.28500044), (23, 0.099720411), (134, 0.036305476), (22, 0.033559218)],
            'inception_v3'      : [(21, 0.91967654), (23, 0.0029040477), (24, 0.0020232804), (146, 0.0019062747), (22, 0.0017500133)],
            'xception'          : [(21, 0.67462814), (23, 0.063138723), (87, 0.028424012), (89, 0.02484037), (88, 0.0062591862)],
            'mobilenet'         : [(21, 0.7869994), (23, 0.14728773), (146, 0.037277445), (144, 0.0061039869), (727, 0.0046111974)],
            'resnet'            : [(144, 0.80301273), (23, 0.067478567), (21, 0.046560187), (562, 0.037413299), (146, 0.015967956)],
            'inception_resnet'  : [(21, 0.93837249), (87, 0.0021177295), (146, 0.0019775454), (23, 0.00072135136), (24, 0.00056668324)]
        },
        'mxnet' : {
            'vgg19'          : [(21, 0.54552644), (144, 0.19179004), (23, 0.066389613), (22, 0.022819581), (128, 0.02271222)],
            'resnet'         : [(21, 0.84012794), (144, 0.097428247), (23, 0.039757393), (146, 0.010432643), (99, 0.0023797606)],
            'squeezenet'     : [(21, 0.36026478), (128, 0.084114805), (835, 0.07940048), (144, 0.057378717), (749, 0.053491514)],
            'inception_bn'   : [(21, 0.84332663), (144, 0.041747514), (677, 0.021810319), (973, 0.02054958), (115, 0.008529461)],
            'resnet152-11k'  : [(1278, 0.49073416), (1277, 0.21393695), (282, 0.12980066), (1282, 0.0663582), (1224, 0.022041745)]
        },
        'pytorch' :{
            'resnet152' : [(21, 13.080057), (141, 12.32998), (94, 9.8761454), (146, 9.3761511), (143, 8.9194641)],
            'vgg19'     : [(821, 8.4734678), (562, 8.3472366), (835, 8.2712851), (749, 7.792901), (807, 6.6604013)],
        }
    }

    preprocess_func = {
        'caffe' : {
            'alexnet'       : lambda path : TestKit.ZeroCenter(path, 227, True),
            'vgg19'         : lambda path : TestKit.ZeroCenter(path, 224, True),
            'inception_v1'  : lambda path : TestKit.ZeroCenter(path, 224, True),
            'resnet152'     : lambda path : TestKit.ZeroCenter(path, 224, True)
        },

        'tensorflow' : {
            'vgg19'         : lambda path : TestKit.ZeroCenter(path, 224, False),
            'inception_v3'  : lambda path : TestKit.Standard(path, 299),
            'resnet'        : lambda path : TestKit.Standard(path, 299),
            'resnet_v1_101' : lambda path : TestKit.ZeroCenter(path, 224, False),
            'resnet152'     : lambda path : TestKit.Standard(path, 299),
            'mobilenet'     : lambda path : TestKit.Standard(path, 224)
        },

        'keras' : {
            'vgg16'             : lambda path : TestKit.ZeroCenter(path, 224, True),
            'vgg19'             : lambda path : TestKit.ZeroCenter(path, 224, True),
            'inception_v3'      : lambda path : TestKit.Standard(path, 299),
            'resnet'            : lambda path : TestKit.ZeroCenter(path, 224, True),
            'xception'          : lambda path : TestKit.Standard(path, 299),
            'mobilenet'         : lambda path : TestKit.Standard(path, 224),
            'inception_resnet'  : lambda path : TestKit.Standard(path, 299)
        },

        'mxnet' : {
            'vgg19'         : lambda path : TestKit.ZeroCenter(path, 224, False),
            'resnet'        : lambda path : TestKit.Identity(path, 224, True),
            'squeezenet'    : lambda path : TestKit.ZeroCenter(path, 224, False),
            'inception_bn'  : lambda path : TestKit.Identity(path, 224, False),
            'resnet152-11k' : lambda path : TestKit.Identity(path, 224, True)
        },

        'pytorch' : {
            'vgg19'         : lambda path : TestKit.Normalize(path),
            'resnet152'     : lambda path : TestKit.Normalize(path),
            'inception_v3'  : lambda path : TestKit.Normalize(path),
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

        parser.add_argument('--dump',
            type = _text_type,
            default = None,
            help = 'Target model path.')

        self.args = parser.parse_args()
        if self.args.n.endswith('.py'):
            self.args.n = self.args.n[:-3]
        self.MainModel = __import__(self.args.n)


    @staticmethod
    def ZeroCenter(path, size, BGRTranspose=False):
        img = image.load_img(path, target_size = (size, size))
        x = image.img_to_array(img)
        if BGRTranspose == True:
            x = x[..., ::-1]
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68
        return x


    @staticmethod
    def Normalize(path, size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        img = image.load_img(path, target_size=(size, size))
        x = image.img_to_array(img)
        x /= 255.0
        for i in range(0, 3):
            x[..., i] -= mean[i]
            x[..., i] /= std[i]
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
        top_indices = predict.argsort()[-5:][::-1]
        self.result = [(i, predict[i]) for i in top_indices]
        print (self.result)


    def print_intermediate_result(self, intermediate_output, if_transpose = False):
        intermediate_output = np.squeeze(intermediate_output)

        if if_transpose == True:
            intermediate_output = np.transpose(intermediate_output, [2, 0, 1])

        print (intermediate_output)
        print (intermediate_output.shape)
        print ("%.30f" % np.sum(intermediate_output))


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
