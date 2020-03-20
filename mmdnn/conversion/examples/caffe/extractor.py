#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import absolute_import
import os
from mmdnn.conversion.examples.imagenet_test import TestKit
from mmdnn.conversion.examples.extractor import base_extractor
from mmdnn.conversion.common.utils import download_file


class caffe_extractor(base_extractor):

    BASE_MODEL_URL = 'http://data.mxnet.io/models/imagenet/test/caffe/'
    MMDNN_BASE_URL = 'http://mmdnn.eastasia.cloudapp.azure.com:89/models/'

    architecture_map = {
## Image Classification
        'alexnet'       : {'prototxt'   : 'https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt',
                        'caffemodel' : 'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel'},
        'inception_v1'  : {'prototxt'   : 'https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt',
                        'caffemodel' : 'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel'},
        'vgg16'         : {'prototxt'   : 'https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/c3ba00e272d9f48594acef1f67e5fd12aff7a806/VGG_ILSVRC_16_layers_deploy.prototxt',
                        'caffemodel' : 'http://data.mxnet.io/models/imagenet/test/caffe/VGG_ILSVRC_16_layers.caffemodel'},
        'vgg19'         : {'prototxt'   : 'https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt',
                        'caffemodel' : 'http://data.mxnet.io/models/imagenet/test/caffe/VGG_ILSVRC_19_layers.caffemodel'},
        'resnet50'      : {'prototxt'   : BASE_MODEL_URL + 'ResNet-50-deploy.prototxt',
                        'caffemodel' : BASE_MODEL_URL + 'ResNet-50-model.caffemodel'},
        'resnet101'     : {'prototxt'   : BASE_MODEL_URL + 'ResNet-101-deploy.prototxt',
                        'caffemodel' : BASE_MODEL_URL + 'ResNet-101-model.caffemodel'},
        'resnet152'     : {'prototxt'   : BASE_MODEL_URL + 'ResNet-152-deploy.prototxt',
                        'caffemodel' : BASE_MODEL_URL + 'ResNet-152-model.caffemodel'},
        'squeezenet'    : {'prototxt' : "https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.1/deploy.prototxt",
                           'caffemodel' : "https://github.com/DeepScale/SqueezeNet/raw/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel"},
        'xception'      : {'prototxt' : MMDNN_BASE_URL + "caffe/xception_deploy.prototxt",
                           'caffemodel' : MMDNN_BASE_URL + "caffe/xception.caffemodel"},
        'inception_v4'  : {'prototxt' : MMDNN_BASE_URL + 'caffe/inception-v4_deploy.prototxt',
                           'caffemodel' : MMDNN_BASE_URL + 'caffe/inception-v4.caffemodel'},
## Semantic Segmentation
        'voc-fcn8s'     : {'prototxt' : 'https://raw.githubusercontent.com/shelhamer/fcn.berkeleyvision.org/master/voc-fcn8s/deploy.prototxt',
                           'caffemodel' : 'http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel'},
        'voc-fcn16s'    : {'prototxt' : 'https://raw.githubusercontent.com/linmajia/mmdnn-models/master/caffe/voc-fcn16s-deploy.prototxt',
                           'caffemodel' : 'http://dl.caffe.berkeleyvision.org/fcn16s-heavy-pascal.caffemodel'},
        'voc-fcn32s'    : {'prototxt' : 'https://raw.githubusercontent.com/linmajia/mmdnn-models/master/caffe/voc-fcn32s-deploy.prototxt',
                           'caffemodel' : 'http://dl.caffe.berkeleyvision.org/fcn32s-heavy-pascal.caffemodel'},
        'trailnet_sresnet': {'prototxt': 'https://raw.githubusercontent.com/NVIDIA-AI-IOT/redtail/master/models/pretrained/TrailNet_SResNet-18.prototxt',
                            'caffemodel': 'https://raw.githubusercontent.com/NVIDIA-AI-IOT/redtail/master/models/pretrained/TrailNet_SResNet-18.caffemodel'}
    }


    @classmethod
    def download(cls, architecture, path="./"):
        if cls.sanity_check(architecture):
            prototxt_name = architecture + "-deploy.prototxt"
            architecture_file = download_file(cls.architecture_map[architecture]['prototxt'], directory=path, local_fname=prototxt_name)
            if not architecture_file:
                return None

            weight_name = architecture + ".caffemodel"
            weight_file = download_file(cls.architecture_map[architecture]['caffemodel'], directory=path, local_fname=weight_name)
            if not weight_file:
                return None


            print("Caffe Model {} saved as [{}] and [{}].".format(architecture, architecture_file, weight_file))
            return (architecture_file, weight_file)

        else:
            return None


    @classmethod
    def inference(cls, architecture_name, architecture, path, image_path):
        if cls.sanity_check(architecture_name):
            import caffe
            import numpy as np
            net = caffe.Net(architecture[0], architecture[1], caffe.TEST)

            func = TestKit.preprocess_func['caffe'][architecture_name]
            img = func(image_path)
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, 0)
            net.blobs['data'].data[...] = img
            predict = np.squeeze(net.forward()[net._output_list[-1]][0])
            predict = np.squeeze(predict)
            return predict

        else:
            return None
