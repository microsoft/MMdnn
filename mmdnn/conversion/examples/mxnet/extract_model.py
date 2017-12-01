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

_base_model_url = 'http://data.mxnet.io/models/'
_default_model_info = {
    'imagenet1k-inception-bn': {'symbol':_base_model_url+'imagenet/inception-bn/Inception-BN-symbol.json',
                             'params':_base_model_url+'imagenet/inception-bn/Inception-BN-0126.params'},
    'imagenet1k-resnet-18': {'symbol':_base_model_url+'imagenet/resnet/18-layers/resnet-18-symbol.json',
                             'params':_base_model_url+'imagenet/resnet/18-layers/resnet-18-0000.params'},
    'imagenet1k-resnet-34': {'symbol':_base_model_url+'imagenet/resnet/34-layers/resnet-34-symbol.json',
                             'params':_base_model_url+'imagenet/resnet/34-layers/resnet-34-0000.params'},
    'imagenet1k-resnet-50': {'symbol':_base_model_url+'imagenet/resnet/50-layers/resnet-50-symbol.json',
                             'params':_base_model_url+'imagenet/resnet/50-layers/resnet-50-0000.params'},
    'imagenet1k-resnet-101': {'symbol':_base_model_url+'imagenet/resnet/101-layers/resnet-101-symbol.json',
                             'params':_base_model_url+'imagenet/resnet/101-layers/resnet-101-0000.params'},
    'imagenet1k-resnet-152': {'symbol':_base_model_url+'imagenet/resnet/152-layers/resnet-152-symbol.json',
                             'params':_base_model_url+'imagenet/resnet/152-layers/resnet-152-0000.params'},
    'imagenet1k-resnext-50': {'symbol':_base_model_url+'imagenet/resnext/50-layers/resnext-50-symbol.json',
                             'params':_base_model_url+'imagenet/resnext/50-layers/resnext-50-0000.params'},
    'imagenet1k-resnext-101': {'symbol':_base_model_url+'imagenet/resnext/101-layers/resnext-101-symbol.json',
                             'params':_base_model_url+'imagenet/resnext/101-layers/resnext-101-0000.params'},
    'imagenet1k-resnext-101-64x4d': {'symbol':_base_model_url+'imagenet/resnext/101-layers/resnext-101-64x4d-symbol.json',
                                     'params':_base_model_url+'imagenet/resnext/101-layers/resnext-101-64x4d-0000.params'},
    'imagenet11k-resnet-152': {'symbol':_base_model_url+'imagenet-11k/resnet-152/resnet-152-symbol.json',
                             'params':_base_model_url+'imagenet-11k/resnet-152/resnet-152-0000.params'},
    'imagenet11k-place365ch-resnet-152': {'symbol':_base_model_url+'imagenet-11k-place365-ch/resnet-152-symbol.json',
                                          'params':_base_model_url+'imagenet-11k-place365-ch/resnet-152-0000.params'},
    'imagenet11k-place365ch-resnet-50': {'symbol':_base_model_url+'imagenet-11k-place365-ch/resnet-50-symbol.json',
                                         'params':_base_model_url+'imagenet-11k-place365-ch/resnet-50-0000.params'},
}


def _main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--network', type=_text_type, help='Model Type', required=True,
                        choices = _default_model_info.keys())

    parser.add_argument('-i', '--image',
                        type=_text_type, help='Test Image Path')

    parser.add_argument('-o', '--output_dir', default='./',
                        type=_text_type, help='Tensorflow Checkpoint file name')

    args = parser.parse_args()

    if not download_file(_default_model_info[args.network]['symbol'], directory=args.output_dir):
        return -1

    if not download_file(_default_model_info[args.network]['params'], directory=args.output_dir):
        return -1

    print("Model {} saved.".format(args.network))

    return 0


if __name__ == '__main__':
    _main()
