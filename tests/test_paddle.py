from __future__ import absolute_import
from __future__ import print_function

import os
import sys
from conversion_imagenet import TestModels
from conversion_imagenet import is_paddle_supported

def get_test_table():
    return { 'paddle' : {
        'resnet50'      : [
                TestModels.onnx_emit,
                #TestModels.caffe_emit,
                #TestModels.cntk_emit,
                TestModels.coreml_emit,
                TestModels.keras_emit,
                TestModels.mxnet_emit,
                TestModels.pytorch_emit,
                TestModels.tensorflow_emit
                ],
        'resnet101'              : [
                #TestModels.onnx_emit,
                #TestModels.caffe_emit,
                #TestModels.cntk_emit,
                TestModels.coreml_emit,
                TestModels.keras_emit,
                TestModels.mxnet_emit,
                TestModels.pytorch_emit,
                TestModels.tensorflow_emit
                ],
        'vgg16'        : [
                TestModels.onnx_emit,
                #TestModels.caffe_emit,
                #TestModels.cntk_emit,
                #TestModels.coreml_emit,
                #TestModels.keras_emit,
                #TestModels.mxnet_emit,
                #TestModels.pytorch_emit,
                #TestModels.tensorflow_emit
                ],
    }}

def test_paddle():
    if not is_paddle_supported():
        return
    # omit tensorflow lead to crash
    import tensorflow as tf
    test_table = get_test_table()
    tester = TestModels(test_table)
    tester._test_function('paddle', tester.paddle_parse)


if __name__ == '__main__':
    test_paddle()
