from __future__ import absolute_import
from __future__ import print_function

import os
import sys
from conversion_imagenet import TestModels

def get_test_table():
    return { 'pytorch' :
        {
            'vgg19'    : [
                #TestModels.onnx_emit,
                TestModels.caffe_emit,
                #TestModels.cntk_emit,
                TestModels.coreml_emit,
                TestModels.keras_emit,
                TestModels.mxnet_emit,
                TestModels.pytorch_emit,
                TestModels.tensorflow_emit
                ],
            'vgg19_bn'    : [
                #TestModels.onnx_emit,
                TestModels.caffe_emit,
                #TestModels.cntk_emit,
                TestModels.coreml_emit,
                TestModels.keras_emit,
                TestModels.mxnet_emit,
                TestModels.pytorch_emit,
                TestModels.tensorflow_emit
                ]
        }
    }

def test_pytorch():
    test_table = get_test_table()
    tester = TestModels(test_table)
    tester._test_function('pytorch', tester.pytorch_parse)


if __name__ == '__main__':
    test_pytorch()
