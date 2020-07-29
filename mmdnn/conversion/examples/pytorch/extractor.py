#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import absolute_import

import os
from mmdnn.conversion.examples.imagenet_test import TestKit
from mmdnn.conversion.examples.extractor import base_extractor
from mmdnn.conversion.common.utils import download_file
import torch
import torchvision.models as models

class pytorch_extractor(base_extractor):

    architecture_map = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))


    @classmethod
    def help(cls):
        print('Supported models: {}'.format(cls.architecture_map))


    @classmethod
    def download(cls, architecture, path="./"):
        if cls.sanity_check(architecture):
            architecture_file = path + "imagenet_{}.pth".format(architecture)
            if not os.path.exists(architecture_file):
                kwargs = {}
                if architecture == 'inception_v3':
                    kwargs['transform_input'] = False
                model = models.__dict__[architecture](pretrained=True, **kwargs)
                torch.save(model, architecture_file)
                print("PyTorch pretrained model is saved as [{}].".format(architecture_file))
            else:
                print("File [{}] existed!".format(architecture_file))

            return architecture_file

        else:
            return None


    @classmethod
    def inference(cls, architecture, path, image_path):
        model = torch.load(path)

        model.eval()

        import numpy as np
        func = TestKit.preprocess_func['pytorch'][architecture]
        img = func(image_path)
        img = np.transpose(img, (2, 0, 1))

        img = np.expand_dims(img, 0).copy()

        data = torch.from_numpy(img)
        data = torch.autograd.Variable(data, requires_grad=False)

        predict = model(data).data.numpy()
        predict = np.squeeze(predict)

        return predict
