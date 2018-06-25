#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import argparse
import os
from six import text_type as _text_type
import paddle.v2 as paddle
from mmdnn.conversion.common.utils import download_file

BASE_MODEL_URL = 'http://cloud.dlnel.org/filepub/?uuid='
# pylint: disable=line-too-long
MODEL_URL = {
    'resnet50'             : BASE_MODEL_URL + 'f63f237a-698e-4a22-9782-baf5bb183019',
    'resnet101'            : BASE_MODEL_URL + '3d5fb996-83d0-4745-8adc-13ee960fc55c',
    'vgg16'                : BASE_MODEL_URL + 'aa0e397e-474a-4cc1-bd8f-65a214039c2e',
}
# pylint: enable=line-too-long

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

    model = C.Function.load(fn)

    DATA_DIM = 3 * 224 * 224  # Use 3 * 331 * 331 or 3 * 299 * 299 for Inception-ResNet-v2.
    CLASS_DIM = 1001

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

    dump_v2_config(out, path + architecture + '.bin')


    if len(model.outputs) > 1:
        for idx, output in enumerate(model.outputs):
            if len(output.shape) > 0:
                eval_node = idx
                break

        model = C.as_composite(model[eval_node].owner)
        model.save(fn)

        print("Model {} is saved as {}.".format(args.network, fn))

    if args.image:
        import numpy as np
        from mmdnn.conversion.examples.imagenet_test import TestKit
        func = TestKit.preprocess_func['paddle'][args.network]
        img = func(args.image)
        img = np.transpose(img, (2, 0, 1))
        test_data = [(img.flatten(),)]
        predict = paddle.infer(output_layer = out, parameters=parameters, input=test_data)
        predict = np.squeeze(predict)
        top_indices = predict.argsort()[-5:][::-1]
        result = [(i, predict[i]) for i in top_indices]
        print(result)
        print(np.sum(result))

    return 0


if __name__ == '__main__':
    _main()
