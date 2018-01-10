#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import argparse
import numpy as np
import sys
import os
from mmdnn.conversion.examples.imagenet_test import TestKit
import coremltools

class TestCoreML(TestKit):

    def __init__(self):
        from six import text_type as _text_type
        parser = argparse.ArgumentParser()

        parser.add_argument('-p', '--preprocess', type=_text_type, help='Model Preprocess Type')

        parser.add_argument('--model', '-n', '-w', type=_text_type,
                            required=True, help='CoreML Model path.')

        parser.add_argument('-s', type=_text_type, help='Source Framework Type',
                            choices=self.truth.keys())

        parser.add_argument('--image', '-i',
                            type=_text_type, help='Test image path.',
                            default="mmdnn/conversion/examples/data/seagull.jpg")

        parser.add_argument('-input', type=_text_type,
                            required=True, help='CoreML Input Node')

        parser.add_argument('-output', type=_text_type,
                            required=True, help='CoreML Output Node')

        parser.add_argument('-size', type=int,
            default=224, help='CoreML Input Image Size')


        self.args = parser.parse_args()

        print("Loading model [{}].".format(self.args.model))

        self.model = coremltools.models.MLModel(self.args.model.encode())

        print("Model loading success.")

    def preprocess(self, image_path):
        from PIL import Image as pil_image
        img = pil_image.open(image_path)
        img = img.resize((self.args.size, self.args.size))
        self.data = img

    def print_result(self):
        coreml_inputs = {self.args.input: self.data}
        self.coreml_output = self.model.predict(coreml_inputs, useCPUOnly=False)
        predict = self.coreml_output[self.args.output]
        super(TestCoreML, self).print_result(predict)


    def print_intermediate_result(self, layer_name, if_transpose = False):
        super(TestCoreML, self).print_intermediate_result(self.coreml_output[layer_name], if_transpose)


    def inference(self, image_path):
        self.preprocess(image_path)

        self.print_result()

        # self.print_intermediate_result('conv1_7x7_s2_1', True)

        # self.test_truth()

if __name__=='__main__':
    tester = TestCoreML()
    tester.inference(tester.args.image)
