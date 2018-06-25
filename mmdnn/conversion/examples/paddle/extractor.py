#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import absolute_import
from mmdnn.conversion.examples.imagenet_test import TestKit
from mmdnn.conversion.examples.extractor import base_extractor
from mmdnn.conversion.common.utils import download_file


class paddle_extractor(base_extractor):

    _base_model_url = 'http://cloud.dlnel.org/filepub/?uuid='

    _image_size     = 224


    architecture_map = {
            'resnet50'             : {'params' : BASE_MODEL_URL + 'f63f237a-698e-4a22-9782-baf5bb183019',}
            'resnet101'            : {'params' : BASE_MODEL_URL + '3d5fb996-83d0-4745-8adc-13ee960fc55c',}
            'vgg16'                : {'params': BASE_MODEL_URL + 'aa0e397e-474a-4cc1-bd8f-65a214039c2e',}

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
        import paddle.v2 as paddle
        import numpy as np
        if cls.sanity_check(architecture):
            file_name = cls.architecture_map[architecture]['params'].split('/')[-1]
            prefix, epoch_num = file_name[:-7].rsplit('-', 1)

            sym, arg_params, aux_params = mx.model.load_checkpoint(path + prefix, int(epoch_num))
            model = mx.mod.Module(symbol=sym)
            model.bind(for_training=False,
                       data_shapes=[('data', (1, 3, cls._image_size, cls._image_size))])
            model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

            func = TestKit.preprocess_func['paddle'][architecture]
            img = func(image_path)
            img = np.transpose(img, [2, 0, 1])
            test_data = [(img.flatten(),)]
            predict = paddle.infer(output_layer = out, parameters=parameters, input=test_data)
            predict = np.squeeze(predict)

            del model
            return predict

        else:
            return None
