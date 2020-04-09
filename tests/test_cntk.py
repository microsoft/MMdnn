from __future__ import absolute_import
from __future__ import print_function

import os
import sys
from conversion_imagenet import TestModels

def get_test_table():
    return { 'cntk' :
        {
            'inception_v3'     : [
                TestModels.onnx_emit,
                #TestModels.caffe_emit,
                TestModels.cntk_emit,
                #TestModels.coreml_emit,
                #TestModels.keras_emit,
                #TestModels.mxnet_emit,
                TestModels.pytorch_emit,
                TestModels.tensorflow_emit
                ],
            'resnet18'    : [
                TestModels.onnx_emit,
                TestModels.caffe_emit,
                TestModels.cntk_emit,
                TestModels.coreml_emit,
                TestModels.keras_emit,
                TestModels.mxnet_emit,
                TestModels.pytorch_emit,
                TestModels.tensorflow_emit
                ]
        }
    }

def test_cntk():
    test_table = get_test_table()
    tester = TestModels(test_table)
    tester._test_function('cntk', tester.cntk_parse)


if __name__ == '__main__':
    test_cntk()
