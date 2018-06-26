#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import argparse
import numpy as np
import sys
import os
from mmdnn.conversion.examples.imagenet_test import TestKit
import paddle.v2 as paddle
import gzip
from paddle.trainer_config_helpers.config_parser_utils import \
    reset_parser


class TestPaddle(TestKit):

    def __init__(self):
        from six import text_type as _text_type
        parser = argparse.ArgumentParser()

        parser.add_argument('-p', '--preprocess', type=_text_type, help='Model Preprocess Type')

        parser.add_argument('--model', '-n', '-w', type=_text_type,
                            required=True, help='Paddle Model path.')

        parser.add_argument('-s', type=_text_type, help='Source Framework Type',
                            choices=self.truth.keys())

        parser.add_argument('--image', '-i',
                            type=_text_type, help='Test image path.',
                            default="mmdnn/conversion/examples/data/seagull.jpg")

        parser.add_argument('-input', type=_text_type,
                            required=True, help='Paddle Input Node')

        parser.add_argument('-output', type=_text_type,
                            required=True, help='Paddle Output Node')

        parser.add_argument('-size', type=int,
            default=224, help='Paddle Input Image Size')




        self.args = parser.parse_args()

        print("Loading model [{}].".format(self.args.model))

        # import self.model
        # self.model

        # how the model can not load from `***.bin`

        print("Model loading success.")

    def preprocess(self, image_path):
        from PIL import Image as pil_image
        img = pil_image.open(image_path)
        img = img.resize((self.args.size, self.args.size))
        self.data = img

    def print_result(self):
        reset_parser()
        img = np.transpose(self.data, (2, 0, 1))
        test_data = [(img.flatten(),)]

        parameters_file = self.args.w
        with gzip.open(parameters_file, 'r') as f:
            parameters = paddle.parameters.Parameters.from_tar(f)


        predict = paddle.infer(output_layer = self.model, parameters=parameters, input=test_data)
        predict = np.squeeze(predict)

        super(TestPaddle, self).print_result(predict)


    def print_intermediate_result(self, layer_name, if_transpose = False):
        super(TestPaddle, self).print_intermediate_result(self.model.name, if_transpose)


    def inference(self, image_path):
        self.preprocess(image_path)
        self.print_result()


if __name__=='__main__':
    tester = TestPaddle()
    tester.inference(tester.args.image)
