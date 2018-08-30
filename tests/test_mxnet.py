from __future__ import absolute_import
from __future__ import print_function

import os
import six
from test_conversion_imagenet import TestModels

def get_test_table():
    TRAVIS_CI = os.environ.get('TRAVIS')
    if not TRAVIS_CI or TRAVIS_CI.lower() != 'true':
        return None

    if six.PY2: return None

    ONNX = os.environ.get('TEST_ONNX')
    if ONNX and ONNX.lower() == 'true':
        return { 'mxnet' : {
            'imagenet1k-inception-bn'      : [TestModels.OnnxEmit],
            'imagenet1k-resnet-18'         : [TestModels.OnnxEmit],
            'imagenet1k-resnet-152'        : [TestModels.OnnxEmit],
            'squeezenet_v1.1'              : [TestModels.OnnxEmit],
            'imagenet1k-resnext-50'        : [TestModels.OnnxEmit],
        }}
    else:
        return { 'mxnet' : {
            'imagenet1k-inception-bn'      : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
            'imagenet1k-resnet-18'         : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
            'imagenet1k-resnet-152'        : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
            'squeezenet_v1.1'              : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
            'imagenet1k-resnext-50'        : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
        }}


def test_mxnet():
    test_table = get_test_table()
    tester = TestModels(test_table)
    tester._test_function('mxnet', tester.MXNetParse)