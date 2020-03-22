from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import six
from conversion_imagenet import TestModels

def get_test_table():
    return { 'caffe' :
        {
            # Cannot run on Travis since it seems to consume too much memory.
            'voc-fcn8s'     : [
                TestModels.cntk_emit,
                TestModels.coreml_emit,
                TestModels.tensorflow_emit
                ],
            'voc-fcn16s'     : [
                TestModels.cntk_emit,
                TestModels.coreml_emit,
                TestModels.tensorflow_emit
                ],
            'voc-fcn32s'     : [
                TestModels.cntk_emit,
                TestModels.coreml_emit,
                TestModels.tensorflow_emit
                ],
            #Temporarily disable 'xception'      : [TestModels.mxnet_emit, TestModels.pytorch_emit],
            #Temporarily disable 'inception_v4'  : [TestModels.cntk_emit, TestModels.coreml_emit, TestModels.keras_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
        }
    }

def test_caffe():
    test_table = get_test_table()
    tester = TestModels(test_table)
    tester._test_function('caffe', tester.caffe_parse)


if __name__ == '__main__':
    test_caffe()