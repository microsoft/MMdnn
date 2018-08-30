from __future__ import absolute_import
from __future__ import print_function

import os
import six
from test_conversion_imagenet import TestModels

def get_test_table():
    if six.PY3:
        return None

    ONNX = os.environ.get('TEST_ONNX')
    if ONNX and ONNX.lower() == 'true':
        return {
            'keras' : {
                'vgg16'        : [TestModels.OnnxEmit],
                'vgg19'        : [TestModels.OnnxEmit],
                'inception_v3' : [TestModels.OnnxEmit],
                'resnet50'     : [TestModels.OnnxEmit],
                'densenet'     : [TestModels.OnnxEmit],
                # 'xception'     : [TestModels.OnnxEmit],
                'mobilenet'    : [TestModels.OnnxEmit],
                # 'nasnet'       : [TestModels.OnnxEmit],
            },
        }

    else:
        return {
            'keras' : {
                'vgg19'        : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
                'inception_v3' : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
                'resnet50'     : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
                'densenet'     : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
                'xception'     : [TestModels.TensorflowEmit, TestModels.KerasEmit, TestModels.CoreMLEmit],
                'mobilenet'    : [TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.TensorflowEmit],
        }}


def test_keras():
    test_table = get_test_table()
    tester = TestModels(test_table)
    tester._test_function('keras', tester.KerasParse)