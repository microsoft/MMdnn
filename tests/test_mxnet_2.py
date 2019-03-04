from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import six
from conversion_imagenet import TestModels

def get_test_table():
    TRAVIS_CI = os.environ.get('TRAVIS')
    if not TRAVIS_CI or TRAVIS_CI.lower() != 'true':
        return None

    if six.PY2: return None

    ONNX = os.environ.get('TEST_ONNX')
    if ONNX and ONNX.lower() == 'true':
        return { 'mxnet' : {
            'imagenet1k-resnet-18'         : [TestModels.onnx_emit],
            'imagenet1k-resnet-152'        : [TestModels.onnx_emit],
        }}
    else:
        return { 'mxnet' : {
            'imagenet1k-resnet-18'         : [TestModels.caffe_emit, TestModels.cntk_emit, TestModels.coreml_emit, TestModels.keras_emit, TestModels.mxnet_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
            'imagenet1k-resnet-152'        : [TestModels.caffe_emit, TestModels.cntk_emit, TestModels.coreml_emit, TestModels.keras_emit, TestModels.mxnet_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
        }}


def test_mxnet():
    test_table = get_test_table()
    tester = TestModels(test_table)
    tester._test_function('mxnet', tester.mxnet_parse)


if __name__ == '__main__':
    test_mxnet()

