#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import argparse
import os
from six import text_type as _text_type
from mmdnn.conversion.common.utils import download_file
import paddle.v2 as paddle
import gzip
from paddle.trainer_config_helpers.config_parser_utils import \
    reset_parser

BASE_MODEL_URL = 'http://cloud.dlnel.org/filepub/?uuid='
# pylint: disable=line-too-long
MODEL_URL = {
    'resnet50'             : BASE_MODEL_URL + 'f63f237a-698e-4a22-9782-baf5bb183019',
    'resnet101'            : BASE_MODEL_URL + '3d5fb996-83d0-4745-8adc-13ee960fc55c',
    'vgg16'                : BASE_MODEL_URL + 'aa0e397e-474a-4cc1-bd8f-65a214039c2e',
}
# pylint: enable=line-too-long
IMG_SIZE = 224
CLASS_DIMS = {
        'resnet50'             : 1000,
        'resnet101'            : 1000,
        'vgg16'                : 1001, # work at 1001, but fail at 1000
        'alexnet'              : 1001,
}

def dump_v2_config(topology, save_path, binary=False):
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

def _main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--network', type=_text_type, help='Model Type', required=True,
                        choices=MODEL_URL.keys())

    parser.add_argument('-i', '--image', default=None,
                        type=_text_type, help='Test Image Path')

    parser.add_argument('-o', '--output_dir', default='./',
                        type=_text_type, help='Paddlepaddle parameters file name')

    args = parser.parse_args()

    fn = download_file(MODEL_URL[args.network], local_fname = architecture + '.tar.gz', directory=args.output_dir)
    if not fn:
        return -1


    DATA_DIM = 3 * IMG_SIZE * IMG_SIZE  # Use 3 * 331 * 331 or 3 * 299 * 299 for Inception-ResNet-v2.
    CLASS_DIM = CLASS_DIMS[args.network]

    # refer to https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/tests/test_rnn_layer.py#L35
    reset_parser()

    # refer to https://github.com/PaddlePaddle/Paddle/issues/7403
    paddle.init(use_gpu=False, trainer_count=1)

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

    dump_v2_config(out, args.output_dir + architecture + '.bin')


    print("Model {} is saved as {} and {}.".format(args.network,  args.output_dir + architecture + '.bin', fn))

    if args.image:

        import numpy as np
        from mmdnn.conversion.examples.imagenet_test import TestKit
        func = TestKit.preprocess_func['paddle'][args.network]
        img = func(args.image)
        img = np.transpose(img, (2, 0, 1))
        test_data = [(img.flatten(),)]

        with gzip.open(parameters_file, 'r') as f:
            parameters = paddle.parameters.Parameters.from_tar(f)

        predict = paddle.infer(output_layer = out, parameters=parameters, input=test_data)
        predict = np.squeeze(predict)
        top_indices = predict.argsort()[-5:][::-1]
        result = [(i, predict[i]) for i in top_indices]
        print(result)
        print(np.sum(result))

    return 0


if __name__ == '__main__':
    _main()
