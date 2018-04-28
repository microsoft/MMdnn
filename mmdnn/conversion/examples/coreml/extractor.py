#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import absolute_import
import os
import coremltools
from coremltools.models import MLModel
from mmdnn.conversion.examples.imagenet_test import TestKit
from mmdnn.conversion.examples.extractor import base_extractor
from mmdnn.conversion.common.utils import download_file


class coreml_extractor(base_extractor):

    _base_model_url = "https://docs-assets.developer.apple.com/coreml/models/"

    # from collections import namedtuple
    # Batch = namedtuple('Batch', ['data'])

    # TODO
    # Apple has published some of their own models. They can be downloaded from https://developer.apple.com/machine-learning/.
    # Those published models are: SqueezeNet, Places205-GoogLeNet, ResNet50, Inception v3, VGG16
    architecture_map = {
        'inception_v3'      : "https://docs-assets.developer.apple.com/coreml/models/Inceptionv3.mlmodel",
        'vgg16'             : "https://docs-assets.developer.apple.com/coreml/models/VGG16.mlmodel",
        'vgg19'             : None,
        'resnet50'            : "https://docs-assets.developer.apple.com/coreml/models/Resnet50.mlmodel",  # resnet50
        'mobilenet'         : "https://docs-assets.developer.apple.com/coreml/models/MobileNet.mlmodel",
        'xception'          : None,
        'inception_resnet'  : None,
        'densenet'          : None,
        'nasnet'            : None,
        'tinyyolo'          : "https://s3-us-west-2.amazonaws.com/coreml-models/TinyYOLO.mlmodel"

    }



    @classmethod
    def download(cls, architecture, path = './'):
        if cls.sanity_check(architecture):
            architecture_file = download_file(cls.architecture_map[architecture], directory = path)
            if not architecture_file:
                return None


            print('Coreml model {} is saved in [{}]'.format(architecture, path))
            return architecture_file
        else:
            return None


    @classmethod
    def inference(cls, architecture, model_path, image_path):
        # TODO
        from PIL import Image
        import numpy as np
        from coremltools.models._infer_shapes_nn_mlmodel import infer_shapes
        if cls.sanity_check(architecture):
            func = TestKit.preprocess_func['coreml'][architecture]


            import inspect
            funcstr = inspect.getsource(func)

            if len(funcstr.split(',')) == 3:
                size = int(funcstr.split('path,')[1].split(')')[0])
            else:
                size = int(funcstr.split('path,')[1].split(',')[0])



            img = Image.open(image_path)
            img = img.resize((size, size))

            # load model
            model = MLModel(model_path)
            spec = model.get_spec()

            # TODO: Multiple inputs
            input_name = spec.description.input[0].name

            # TODO: Multiple outputs
            output_name = spec.description.output[0].name

            # inference
            input_data = img
            coreml_input = {input_name: img}
            coreml_output = model.predict(coreml_input)


            prob = coreml_output[output_name]
            if isinstance(prob, dict):
                prob = coreml_output[output_name].values()
            prob = np.array(prob).squeeze()

            return prob

        else:
            return None



