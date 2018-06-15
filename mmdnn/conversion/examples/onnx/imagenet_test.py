#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import argparse
import numpy as np
import sys
import os
import tensorflow as tf
from onnx_tf.backend import prepare
from mmdnn.conversion.examples.imagenet_test import TestKit

class TestONNX(TestKit):

    def __init__(self):
        super(TestONNX, self).__init__()
        self.model = prepare(self.MainModel.KitModel(self.args.w))
        # self.input, self.model, self.testop = self.MainModel.KitModel(self.args.w)


    def preprocess(self, image_path):
        x = super(TestONNX, self).preprocess(image_path)
        self.data = np.expand_dims(x, 0)


    def print_result(self):
        predict = self.model.run(self.data)[0]
        super(TestONNX, self).print_result(predict)


    def print_intermediate_result(self, layer_name, if_transpose = False):
        # testop = tf.get_default_graph().get_operation_by_name(layer_name)
        testop = self.testop
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            intermediate_output = sess.run(testop, feed_dict = {self.input : self.data})

        super(TestONNX, self).print_intermediate_result(intermediate_output, if_transpose)


    def inference(self, image_path):
        self.preprocess(image_path)

        # self.print_intermediate_result('conv1_7x7_s2_1', True)

        self.print_result()

        self.test_truth()

if __name__=='__main__':
    tester = TestONNX()
    if tester.args.dump:
        tester.dump()
    else:
        tester.inference(tester.args.image)
