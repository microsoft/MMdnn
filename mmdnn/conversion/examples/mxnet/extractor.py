#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import absolute_import
from mmdnn.conversion.examples.imagenet_test import TestKit
from mmdnn.conversion.examples.extractor import base_extractor
from mmdnn.conversion.common.utils import download_file


class mxnet_extractor(base_extractor):

    _base_model_url = 'http://data.mxnet.io/models/'

    _image_size     = 224

    from collections import namedtuple
    Batch = namedtuple('Batch', ['data'])

    architecture_map = {
        'imagenet1k-inception-bn' : {'symbol' : _base_model_url+'imagenet/inception-bn/Inception-BN-symbol.json',
                                     'params' : _base_model_url+'imagenet/inception-bn/Inception-BN-0126.params'},
        'imagenet1k-resnet-18' : {'symbol' : _base_model_url+'imagenet/resnet/18-layers/resnet-18-symbol.json',
                                  'params' : _base_model_url+'imagenet/resnet/18-layers/resnet-18-0000.params'},
        'imagenet1k-resnet-34' : {'symbol' : _base_model_url+'imagenet/resnet/34-layers/resnet-34-symbol.json',
                                  'params' : _base_model_url+'imagenet/resnet/34-layers/resnet-34-0000.params'},
        'imagenet1k-resnet-50' : {'symbol' : _base_model_url+'imagenet/resnet/50-layers/resnet-50-symbol.json',
                                  'params' : _base_model_url+'imagenet/resnet/50-layers/resnet-50-0000.params'},
        'imagenet1k-resnet-101' : {'symbol' : _base_model_url+'imagenet/resnet/101-layers/resnet-101-symbol.json',
                                   'params' : _base_model_url+'imagenet/resnet/101-layers/resnet-101-0000.params'},
        'imagenet1k-resnet-152' : {'symbol' : _base_model_url+'imagenet/resnet/152-layers/resnet-152-symbol.json',
                                   'params' : _base_model_url+'imagenet/resnet/152-layers/resnet-152-0000.params'},
        'imagenet1k-resnext-50' : {'symbol' : _base_model_url+'imagenet/resnext/50-layers/resnext-50-symbol.json',
                                   'params' : _base_model_url+'imagenet/resnext/50-layers/resnext-50-0000.params'},
        'imagenet1k-resnext-101' : {'symbol' : _base_model_url+'imagenet/resnext/101-layers/resnext-101-symbol.json',
                                    'params' : _base_model_url+'imagenet/resnext/101-layers/resnext-101-0000.params'},
        'imagenet1k-resnext-101-64x4d' : {'symbol' : _base_model_url+'imagenet/resnext/101-layers/resnext-101-64x4d-symbol.json',
                                          'params' : _base_model_url+'imagenet/resnext/101-layers/resnext-101-64x4d-0000.params'},
        'imagenet11k-resnet-152' : {'symbol' : _base_model_url+'imagenet-11k/resnet-152/resnet-152-symbol.json',
                                    'params' : _base_model_url+'imagenet-11k/resnet-152/resnet-152-0000.params'},
        'imagenet11k-place365ch-resnet-152' : {'symbol' : _base_model_url+'imagenet-11k-place365-ch/resnet-152-symbol.json',
                                               'params' : _base_model_url+'imagenet-11k-place365-ch/resnet-152-0000.params'},
        'imagenet11k-place365ch-resnet-50' : {'symbol' : _base_model_url+'imagenet-11k-place365-ch/resnet-50-symbol.json',
                                              'params' : _base_model_url+'imagenet-11k-place365-ch/resnet-50-0000.params'},
        'vgg19' : {'symbol' : _base_model_url+'imagenet/vgg/vgg19-symbol.json',
                   'params' : _base_model_url+'imagenet/vgg/vgg19-0000.params'},
        'vgg16' : {'symbol' : _base_model_url+'imagenet/vgg/vgg16-symbol.json',
                   'params' : _base_model_url+'imagenet/vgg/vgg16-0000.params'},
        'squeezenet_v1.0' : {'symbol' : _base_model_url+'imagenet/squeezenet/squeezenet_v1.0-symbol.json',
                             'params' : _base_model_url+'imagenet/squeezenet/squeezenet_v1.0-0000.params'},
        'squeezenet_v1.1' : {'symbol' : _base_model_url+'imagenet/squeezenet/squeezenet_v1.1-symbol.json',
                             'params' : _base_model_url+'imagenet/squeezenet/squeezenet_v1.1-0000.params'}
    }


    @classmethod
    def download(cls, architecture, path="./"):
        if cls.sanity_check(architecture):
            architecture_file = download_file(cls.architecture_map[architecture]['symbol'], directory=path)
            if not architecture_file:
                return None

            weight_file = download_file(cls.architecture_map[architecture]['params'], directory=path)
            if not weight_file:
                return None

            print("MXNet Model {} saved as [{}] and [{}].".format(architecture, architecture_file, weight_file))
            return (architecture_file, weight_file)

        else:
            return None


    @classmethod
    def inference(cls, architecture, files, path, image_path):
        import mxnet as mx
        import numpy as np
        if cls.sanity_check(architecture):
            file_name = cls.architecture_map[architecture]['params'].split('/')[-1]
            prefix, epoch_num = file_name[:-7].rsplit('-', 1)

            sym, arg_params, aux_params = mx.model.load_checkpoint(path + prefix, int(epoch_num))
            model = mx.mod.Module(symbol=sym)
            model.bind(for_training=False,
                       data_shapes=[('data', (1, 3, cls._image_size, cls._image_size))])
            model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

            func = TestKit.preprocess_func['mxnet'][architecture]
            img = func(image_path)
            img = np.transpose(img, [2, 0, 1])
            img = np.expand_dims(img, axis=0)

            model.forward(cls.Batch([mx.nd.array(img)]))
            predict = model.get_outputs()[0].asnumpy()
            predict = np.squeeze(predict)

            del model
            return predict

        else:
            return None
