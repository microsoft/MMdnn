#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import argparse
from six import text_type as _text_type
import mxnet as mx
from mmdnn.conversion.examples.imagenet_test import TestKit
from mmdnn.conversion.common.utils import download_file
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

network_name_key = ['resnet', 'vgg19', 'squeezenet', 'inception-bn', 'resnext']

_base_model_url = 'http://data.mxnet.io/models/'
_default_model_info = {
    'imagenet1k-inception-bn'           : {'symbol'     : _base_model_url+'imagenet/inception-bn/Inception-BN-symbol.json',
                                           'params'     : _base_model_url+'imagenet/inception-bn/Inception-BN-0126.params',
                                           'image_size' : 224},
    'imagenet1k-resnet-18'              : {'symbol'     : _base_model_url+'imagenet/resnet/18-layers/resnet-18-symbol.json',
                                           'params'     : _base_model_url+'imagenet/resnet/18-layers/resnet-18-0000.params',
                                           'image_size' : 224},
    'imagenet1k-resnet-34'              : {'symbol'     : _base_model_url+'imagenet/resnet/34-layers/resnet-34-symbol.json',
                                           'params'     : _base_model_url+'imagenet/resnet/34-layers/resnet-34-0000.params',
                                           'image_size' : 224},
    'imagenet1k-resnet-50'              : {'symbol'     : _base_model_url+'imagenet/resnet/50-layers/resnet-50-symbol.json',
                                           'params'     : _base_model_url+'imagenet/resnet/50-layers/resnet-50-0000.params',
                                           'image_size' : 224},
    'imagenet1k-resnet-101'             : {'symbol'     : _base_model_url+'imagenet/resnet/101-layers/resnet-101-symbol.json',
                                           'params'     : _base_model_url+'imagenet/resnet/101-layers/resnet-101-0000.params',
                                           'image_size' : 224},
    'imagenet1k-resnet-152'             : {'symbol'     : _base_model_url+'imagenet/resnet/152-layers/resnet-152-symbol.json',
                                           'params'     : _base_model_url+'imagenet/resnet/152-layers/resnet-152-0000.params',
                                           'image_size' : 224},
    'imagenet1k-resnext-50'             : {'symbol'     : _base_model_url+'imagenet/resnext/50-layers/resnext-50-symbol.json',
                                           'params'     : _base_model_url+'imagenet/resnext/50-layers/resnext-50-0000.params',
                                           'image_size' : 224},
    'imagenet1k-resnext-101'            : {'symbol'     : _base_model_url+'imagenet/resnext/101-layers/resnext-101-symbol.json',
                                           'params'     : _base_model_url+'imagenet/resnext/101-layers/resnext-101-0000.params',
                                           'image_size' : 224},
    'imagenet1k-resnext-101-64x4d'      : {'symbol'     : _base_model_url+'imagenet/resnext/101-layers/resnext-101-64x4d-symbol.json',
                                           'params'     : _base_model_url+'imagenet/resnext/101-layers/resnext-101-64x4d-0000.params',
                                           'image_size' : 224},
    'imagenet11k-resnet-152'            : {'symbol'     : _base_model_url+'imagenet-11k/resnet-152/resnet-152-symbol.json',
                                           'params'     : _base_model_url+'imagenet-11k/resnet-152/resnet-152-0000.params',
                                           'image_size' : 224},
    'imagenet11k-place365ch-resnet-152' : {'symbol'     : _base_model_url+'imagenet-11k-place365-ch/resnet-152-symbol.json',
                                           'params'     : _base_model_url+'imagenet-11k-place365-ch/resnet-152-0000.params',
                                           'image_size' : 224},
    'imagenet11k-place365ch-resnet-50'  : {'symbol'     : _base_model_url+'imagenet-11k-place365-ch/resnet-50-symbol.json',
                                           'params'     : _base_model_url+'imagenet-11k-place365-ch/resnet-50-0000.params',
                                           'image_size' : 224},
    'vgg19'                             : {'symbol'     : _base_model_url+'imagenet/vgg/vgg19-symbol.json',
                                           'params'     : _base_model_url+'imagenet/vgg/vgg19-0000.params',
                                           'image_size' : 224},
    'vgg16'                             : {'symbol'     : _base_model_url+'imagenet/vgg/vgg16-symbol.json',
                                           'params'     : _base_model_url+'imagenet/vgg/vgg16-0000.params',
                                           'image_size' : 224},
    'squeezenet_v1.0'                   : {'symbol'     : _base_model_url+'imagenet/squeezenet/squeezenet_v1.0-symbol.json',
                                           'params'     : _base_model_url+'imagenet/squeezenet/squeezenet_v1.0-0000.params',
                                           'image_size' : 224},
    'squeezenet_v1.1'                   : {'symbol'     : _base_model_url+'imagenet/squeezenet/squeezenet_v1.1-symbol.json',
                                           'params'     : _base_model_url+'imagenet/squeezenet/squeezenet_v1.1-0000.params',
                                           'image_size' : 224}
}


def _search_preprocess_key(original_network_name):
    import re
    for key in network_name_key:
        if re.search(key, original_network_name):
            return key
    raise ValueError('preprocess module cannot support [{}]'.format(original_network_name))


def _main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--network', type=_text_type, help='Model Type', required=True,
                        choices=_default_model_info.keys())

    parser.add_argument('-i', '--image', default=None,
                        type=_text_type, help='Test Image Path')

    parser.add_argument('-o', '--output_dir', default='./',
                        type=_text_type, help='Tensorflow Checkpoint file name')

    args = parser.parse_args()

    if not download_file(_default_model_info[args.network]['symbol'], directory=args.output_dir):
        return -1

    if not download_file(_default_model_info[args.network]['params'], directory=args.output_dir):
        return -1

    print("Model {} saved.".format(args.network))

    file_name = _default_model_info[args.network]['params'].split('/')[-1]
    prefix, epoch_num = file_name[:-7].rsplit('-', 1)

    sym, arg_params, aux_params = mx.model.load_checkpoint(args.output_dir + prefix, int(epoch_num))
    model = mx.mod.Module(symbol=sym)
    model.bind(for_training=False,
               data_shapes=[('data', (1, 3, _default_model_info[args.network]['image_size'],
                                      _default_model_info[args.network]['image_size']))])
    model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

    if args.image:
        import numpy as np

        # need to be updated
        network = _search_preprocess_key(args.network)

        func = TestKit.preprocess_func['mxnet'][network]
        img = func(args.image)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = np.expand_dims(img, axis=0)

        model.forward(Batch([mx.nd.array(img)]))
        predict = model.get_outputs()[0].asnumpy()
        predict = np.squeeze(predict)
        top_indices = predict.argsort()[-5:][::-1]
        result = [(i, predict[i]) for i in top_indices]
        print(result)

    return 0


if __name__ == '__main__':
    _main()
