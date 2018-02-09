#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import argparse
import os
from six import text_type as _text_type
from mmdnn.conversion.common.utils import download_file

BASE_MODEL_URL = 'http://data.mxnet.io/models/imagenet/test/caffe/'
# pylint: disable=line-too-long
DEFAULT_MODEL_INFO = {
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
                       'caffemodel' : "https://github.com/DeepScale/SqueezeNet/raw/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel"}
}
# pylint: enable=line-too-long


def _main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--network', type=_text_type, help='Model Type', required=True,
                        choices=DEFAULT_MODEL_INFO.keys())

    parser.add_argument('-i', '--image', default=None,
                        type=_text_type, help='Test Image Path')

    parser.add_argument('-o', '--output_dir', default='./',
                        type=_text_type, help='Caffe Checkpoint file name')

    args = parser.parse_args()

    arch_fn = download_file(DEFAULT_MODEL_INFO[args.network]['prototxt'], directory=args.output_dir)
    if not arch_fn:
        return -1

    weight_fn = download_file(DEFAULT_MODEL_INFO[args.network]['caffemodel'], directory=args.output_dir)
    if not weight_fn:
        return -1

    print("Model {} saved.".format(args.network))

    if args.image:
        import caffe
        import numpy as np
        from mmdnn.conversion.examples.imagenet_test import TestKit

        net = caffe.Net(arch_fn.encode("utf-8"), weight_fn.encode("utf-8"), caffe.TEST)
        # net = caffe.Net(arch_fn, weight_fn, caffe.TEST)
        func = TestKit.preprocess_func['caffe'][args.network]
        img = func(args.image)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        net.blobs['data'].data[...] = img
        predict = np.squeeze(net.forward()['prob'][0])
        predict = np.squeeze(predict)
        top_indices = predict.argsort()[-5:][::-1]
        result = [(i, predict[i]) for i in top_indices]
        print(result)
        print(np.sum(result))

    return 0


if __name__ == '__main__':
    _main()
