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

        self.args = parser.parse_args()

        print("Loading model [{}].".format(self.args.model))

        self.model = coremltools.models.MLModel(self.args.model.encode())


    def preprocess(self, image_path):
        x = super(TestCoreML, self).preprocess(image_path)
        # self.data = np.expand_dims(x, 0)
        self.data = x

    def print_result(self):
        coreml_inputs = {'data': self.data}
        coreml_output = self.model.predict(coreml_inputs, useCPUOnly=True)
        predict = coreml_output['prob']
        super(TestCoreML, self).print_result(predict)


    def print_intermediate_result(self, layer_name, if_transpose = False):
        # testop = tf.get_default_graph().get_operation_by_name(layer_name)
        testop = self.testop
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            intermediate_output = sess.run(testop, feed_dict = {self.input : self.data})

        super(TestTF, self).print_intermediate_result(intermediate_output, if_transpose)


    def inference(self, image_path):
        self.preprocess(image_path)

        # self.print_intermediate_result('conv1_7x7_s2_1', True)

        self.print_result()

        self.test_truth()

if __name__=='__main__':
    tester = TestCoreML()
    tester.inference(tester.args.image)
