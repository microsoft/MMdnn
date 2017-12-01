#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import argparse
import os
from six import text_type as _text_type
import tensorflow as tf
from mmdnn.conversion.examples.imagenet_test import TestKit
from mmdnn.conversion.common.utils import download_file

_base_model_url = 'http://data.mxnet.io/models/imagenet/test/caffe/'
_default_model_info = {
    'bvlc_alexnet': {'prototxt':'https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt',
                             'caffemodel':'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel'},
    'bvlc_googlenet': {'prototxt':'https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt',
                             'caffemodel':'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel'},
    'vgg-16': {'prototxt':'https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/c3ba00e272d9f48594acef1f67e5fd12aff7a806/VGG_ILSVRC_16_layers_deploy.prototxt',
                             'caffemodel':'http://data.mxnet.io/models/imagenet/test/caffe/VGG_ILSVRC_16_layers.caffemodel'},
    'vgg-19': {'prototxt':'https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt',
                             'caffemodel':'http://data.mxnet.io/models/imagenet/test/caffe/VGG_ILSVRC_19_layers.caffemodel'},
    'resnet-50': {'prototxt':_base_model_url+'ResNet-50-deploy.prototxt',
                                 'caffemodel':_base_model_url+'ResNet-50-model.caffemodel'},
    'resnet-101': {'prototxt':_base_model_url+'ResNet-101-deploy.prototxt',
                             'caffemodel':_base_model_url+'ResNet-101-model.caffemodel'},
    'resnet-152': {'prototxt':_base_model_url+'ResNet-152-deploy.prototxt',
                             'caffemodel':_base_model_url+'ResNet-152-model.caffemodel'},
}


def _main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--network', type=_text_type, help='Model Type', required=True,
                        choices = _default_model_info.keys())

    parser.add_argument('-i', '--image', default=None,
                        type=_text_type, help='Test Image Path')

    parser.add_argument('-o', '--output_dir', default='./',
                        type=_text_type, help='Caffe Checkpoint file name')

    args = parser.parse_args()

    if not download_file(_default_model_info[args.network]['prototxt'], directory=args.output_dir):
        return -1

    if not download_file(_default_model_info[args.network]['caffemodel'], directory=args.output_dir):
        return -1

    print("Model {} saved.".format(args.network))

    if args.image:
        # Yuhao TODO: inference code
        pass

    return 0


if __name__ == '__main__':
    _main()
