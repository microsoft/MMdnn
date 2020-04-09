from __future__ import absolute_import
from __future__ import print_function

import os
import sys
from conversion_imagenet import TestModels

def get_test_table():
    return { 'tensorflow_frozen' :
        {
            'inception_v1'    : [
                TestModels.onnx_emit,
                #TestModels.caffe_emit,
                #TestModels.cntk_emit,
                TestModels.coreml_emit,
                TestModels.keras_emit,
                TestModels.mxnet_emit,
                #TestModels.pytorch_emit,
                TestModels.tensorflow_emit
                ],
            'inception_v3'    : [
                TestModels.onnx_emit,
                #TestModels.caffe_emit,
                #TestModels.cntk_emit,
                TestModels.coreml_emit,
                TestModels.keras_emit,
                TestModels.mxnet_emit,
                #TestModels.pytorch_emit,
                TestModels.tensorflow_emit
                ],
            'mobilenet_v1_1.0'    : [
                TestModels.onnx_emit,
                #TestModels.caffe_emit,
                #TestModels.cntk_emit,
                TestModels.coreml_emit,
                TestModels.keras_emit,
                TestModels.mxnet_emit,
                #TestModels.pytorch_emit,
                TestModels.tensorflow_emit
                ]
        }
    }


def test_tensorflow_frozen():
    tester = TestModels()
    tester._test_function('tensorflow_frozen', tester.tensorflow_frozen_parse)


if __name__ == '__main__':
    test_tensorflow_frozen()
