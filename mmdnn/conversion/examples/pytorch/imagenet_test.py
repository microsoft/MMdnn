#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import argparse
import numpy as np
import sys
import os
from mmdnn.conversion.examples.imagenet_test import TestKit
import torch


class TestTorch(TestKit):

    def __init__(self):
        super(TestTorch, self).__init__()
        self.model = self.MainModel.KitModel(self.args.w)
        self.model.eval()

    def preprocess(self, image_path):
        x = super(TestTorch, self).preprocess(image_path)
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0).copy()
        self.data = torch.from_numpy(x)
        self.data = torch.autograd.Variable(self.data, requires_grad = False)


    def print_result(self):
        predict = self.model(self.data)
        predict = predict.data.numpy()        
        super(TestTorch, self).print_result(predict)


    def print_intermediate_result(self, layer_name, if_transpose = False):
        testop = self.testop
        intermediate_output = testop(self.data).data.numpy()
        super(TestTorch, self).predict(intermediate_output, if_transpose)


    def inference(self, image_path):
        self.preprocess(image_path)

        # self.print_intermediate_result('conv1_7x7_s2_1', False)

        self.print_result()

        self.test_truth()



if __name__=='__main__':
    tester = TestTorch()
    tester.inference(tester.args.image)