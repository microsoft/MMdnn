from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import six
from conversion_imagenet import TestModels

def get_test_table():
    if six.PY3:
        return None

    ONNX = os.environ.get('TEST_ONNX')
    if ONNX and ONNX.lower() == 'true':
        return {
            'keras' : {
                'densenet'     : [TestModels.onnx_emit],
                'mobilenet'    : [TestModels.onnx_emit],
                # 'xception'     : [TestModels.onnx_emit],
                # 'nasnet'       : [TestModels.onnx_emit],
            },
        }

    else:
        return {
            'keras' : {
                'densenet'     : [TestModels.caffe_emit, TestModels.cntk_emit, TestModels.coreml_emit, TestModels.keras_emit, TestModels.mxnet_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
                'mobilenet'    : [TestModels.coreml_emit, TestModels.keras_emit, TestModels.tensorflow_emit],
        }}


def test_keras():
    test_table = get_test_table()
    tester = TestModels(test_table)
    tester._test_function('keras', tester.keras_parse)


if __name__ == '__main__':
    test_keras()
