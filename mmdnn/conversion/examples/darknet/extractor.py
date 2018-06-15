#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
import os
from mmdnn.conversion.examples.darknet import darknet as cdarknet
from mmdnn.conversion.examples.imagenet_test import TestKit
from mmdnn.conversion.examples.extractor import base_extractor
from mmdnn.conversion.common.utils import download_file


class darknet_extractor(base_extractor):

    _base_model_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/"

    architecture_map = {
        'yolov3'          : {
            'config'           : _base_model_url + "cfg/yolov3.cfg",
            'weights'          : "https://pjreddie.com/media/files/yolov3.weights"
        },

        'yolov2'          :{
            'config'           : _base_model_url + "cfg/yolov2.cfg",
            'weights'          : "https://pjreddie.com/media/files/yolov2.weights"
        }

    }


    @classmethod
    def download(cls, architecture, path = './'):

        if cls.sanity_check(architecture):
            cfg_name = architecture + ".cfg"
            architecture_file = download_file(cls.architecture_map[architecture]['config'], directory=path, local_fname=cfg_name)
            if not architecture_file:
                return None

            weight_name = architecture + ".weights"
            weight_file = download_file(cls.architecture_map[architecture]['weights'], directory=path, local_fname=weight_name)
            if not weight_file:
                return None

            print("Darknet Model {} saved as [{}] and [{}].".format(architecture, architecture_file, weight_file))
            return (architecture_file, weight_file)

        else:
            return None


    @classmethod
    def inference(cls, architecture, files, model_path, image_path):
        import numpy as np

        if cls.sanity_check(architecture):
            download_file(cls._base_model_url + "cfg/coco.data", directory='./')
            download_file(cls._base_model_url + "data/coco.names", directory='./data/')

            print(files)
            net = cdarknet.load_net(files[0].encode(), files[1].encode(), 0)
            meta = cdarknet.load_meta("coco.data".encode())


            r = cdarknet.detect(net, meta, image_path.encode())
            # print(r)
            return r

        else:
            return None



# d = darknet_extractor()
# model_filename = d.download('yolov3')
# print(model_filename)

# image_path = "./mmdnn/conversion/examples/data/dog.jpg"
# model_path = "./"
# d = darknet_extractor()
# result = d.inference('yolov3', model_filename, model_path, image_path = image_path)
# print(result)
