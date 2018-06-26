#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import absolute_import
from mmdnn.conversion.examples.imagenet_test import TestKit
from mmdnn.conversion.examples.extractor import base_extractor
from mmdnn.conversion.common.utils import download_file
import paddle.v2 as paddle
import gzip
from paddle.trainer_config_helpers.config_parser_utils import \
    reset_parser


class paddle_extractor(base_extractor):

    _base_model_url = 'http://cloud.dlnel.org/filepub/?uuid='

    _image_size     = 224


    architecture_map = {
            'resnet50'             : {'params' : _base_model_url + 'f63f237a-698e-4a22-9782-baf5bb183019',},
            'resnet101'            : {'params' : _base_model_url + '3d5fb996-83d0-4745-8adc-13ee960fc55c',},
            'vgg16'                : {'params': _base_model_url + 'aa0e397e-474a-4cc1-bd8f-65a214039c2e',},

    }

    class_dim_map = {
            'resnet50'             : 1000,
            'resnet101'            : 1000,
            'vgg16'                : 1001, # work at 1001, but fail at 1000
            'alexnet'              : 1001,
    }




    @classmethod
    def dump_v2_config(cls, topology, save_path, binary=False):
        import collections

        from paddle.trainer_config_helpers.layers import LayerOutput
        from paddle.v2.layer import parse_network
        from paddle.proto import TrainerConfig_pb2
        """ Dump the network topology to a specified file.
        This function is only used to dump network defined by using PaddlePaddle V2
        API.
        :param topology: The output layers in the entire network.
        :type topology: LayerOutput|List|Tuple
        :param save_path: The path to save the dump network topology.
        :type save_path: str
        :param binary: Whether to dump the serialized network topology. The default
                    value is false.
        :type binary: bool.
        """

        if isinstance(topology, LayerOutput):
            topology = [topology]
        elif isinstance(topology, collections.Sequence):
            for out_layer in topology:
                assert isinstance(out_layer, LayerOutput), (
                    "The type of each element in the parameter topology "
                    "should be LayerOutput.")
        else:
            raise RuntimeError("Error input type for parameter topology.")

        model_str = parse_network(topology)
        with open(save_path, "w") as fout:
            if binary:
                fout.write(model_str.SerializeToString())
            else:
                fout.write(str(model_str))


    @classmethod
    def download(cls, architecture, path="./"):
        if cls.sanity_check(architecture):
            reset_parser()


            DATA_DIM = 3 * paddle_extractor._image_size * paddle_extractor._image_size  # Use 3 * 331 * 331 or 3 * 299 * 299 for Inception-ResNet-v2.
            CLASS_DIM = paddle_extractor.class_dim_map[architecture]

            image = paddle.layer.data(
                name="image", type=paddle.data_type.dense_vector(DATA_DIM))
            if 'resnet' in architecture:
                from mmdnn.conversion.examples.paddle.models import resnet
                depth = int(architecture.strip('resnet'))
                out = resnet.resnet_imagenet(image, class_dim=CLASS_DIM, depth=depth)
            elif architecture == 'vgg16':
                from mmdnn.conversion.examples.paddle.models import vgg
                out = vgg.vgg16(image, class_dim=CLASS_DIM)
            else:
                print("Not support for {} yet.", architecture)
                return None
            architecture_file = path + architecture + '.bin'
            paddle_extractor.dump_v2_config(out, architecture_file, True)

            weight_file = download_file(cls.architecture_map[architecture]['params'], directory=path, local_fname= architecture +'.tar.gz')
            if not weight_file:
                return None

            print("MXNet Model {} saved as [{}] and [{}].".format(architecture, architecture_file, weight_file))
            return (architecture_file, weight_file)


        else:
            return None


    @classmethod
    def inference(cls, architecture, files, path, image_path):

        import numpy as np
        if cls.sanity_check(architecture):
            # refer to https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/tests/test_rnn_layer.py#L35
            reset_parser()

            # refer to https://github.com/PaddlePaddle/Paddle/issues/7403
            paddle.init(use_gpu=False, trainer_count=1)

            DATA_DIM = 3 * paddle_extractor._image_size * paddle_extractor._image_size  # Use 3 * 331 * 331 or 3 * 299 * 299 for Inception-ResNet-v2.
            CLASS_DIM = paddle_extractor.class_dim_map[architecture]
            image = paddle.layer.data(
                name="image", type=paddle.data_type.dense_vector(DATA_DIM))

            if 'resnet' in architecture:
                from mmdnn.conversion.examples.paddle.models import resnet
                depth = int(architecture.strip('resnet'))
                out = resnet.resnet_imagenet(image, class_dim=CLASS_DIM, depth=depth)
            elif architecture == 'vgg16':
                from mmdnn.conversion.examples.paddle.models import vgg
                out = vgg.vgg16(image, class_dim=CLASS_DIM)
            else:
                print("Not support for {} yet.", architecture)
                return None

            _, parameters_file = files


            with gzip.open(parameters_file, 'r') as f:
                parameters = paddle.parameters.Parameters.from_tar(f)


            func = TestKit.preprocess_func['paddle'][architecture]
            img = func(image_path)
            img = np.transpose(img, [2, 0, 1])
            test_data = [(img.flatten(),)]

            predict = paddle.infer(output_layer = out, parameters=parameters, input=test_data)
            predict = np.squeeze(predict)

            return predict


        else:
            return None
