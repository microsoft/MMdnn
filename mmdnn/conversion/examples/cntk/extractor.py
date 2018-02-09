#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import absolute_import
import cntk as C
from mmdnn.conversion.examples.imagenet_test import TestKit
from mmdnn.conversion.examples.extractor import base_extractor
from mmdnn.conversion.common.utils import download_file


class cntk_extractor(base_extractor):

    BASE_MODEL_URL = 'https://www.cntk.ai/Models/CNTK_Pretrained/'

    architecture_map = {
        'alexnet'              : BASE_MODEL_URL + 'AlexNet_ImageNet_CNTK.model',
        'inception_v3'         : BASE_MODEL_URL + 'InceptionV3_ImageNet_CNTK.model',
        'resnet18'             : BASE_MODEL_URL + 'ResNet18_ImageNet_CNTK.model',
        'resnet50'             : BASE_MODEL_URL + 'ResNet50_ImageNet_CNTK.model',
        'resnet101'            : BASE_MODEL_URL + 'ResNet101_ImageNet_CNTK.model',
        'resnet152'            : BASE_MODEL_URL + 'ResNet152_ImageNet_CNTK.model',
        'Fast-RCNN_grocery100' : 'https://www.cntk.ai/Models/FRCN_Grocery/Fast-RCNN_grocery100.model',
        'Fast-RCNN_Pascal'     : 'https://www.cntk.ai/Models/FRCN_Pascal/Fast-RCNN.model'
    }


    @classmethod
    def download(cls, architecture, path="./"):
        if cls.sanity_check(architecture):
            architecture_file = download_file(cls.architecture_map[architecture], directory=path)
            model = C.Function.load(architecture_file)
            if len(model.outputs) > 1:
                for idx, output in enumerate(model.outputs):
                    if len(output.shape) > 0:
                        eval_node = idx
                        break

                model = C.as_composite(model[eval_node].owner)
                model.save(architecture_file)
                print("Cntk Model {} saved as [{}].".format(architecture, architecture_file))
            return architecture_file

        else:
            return None


    @classmethod
    def inference(cls, architecture_name, architecture_path, image_path):
        if cls.sanity_check(architecture_name):
            import numpy as np
            func = TestKit.preprocess_func['cntk'][architecture_name]
            img = func(image_path)
            img = np.transpose(img, (2, 0, 1))
            model = C.Function.load(architecture_path)
            predict = model.eval({model.arguments[0]:[img]})
            predict = np.squeeze(predict)

            top_indices = predict.argsort()[-5:][::-1]
            return predict

        else:
            return None
