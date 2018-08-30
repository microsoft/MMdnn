from __future__ import absolute_import
from __future__ import print_function

import os
import six
from test_conversion_imagenet import TestModels

def get_test_table():
    TRAVIS_CI = os.environ.get('TRAVIS')
    if not TRAVIS_CI or TRAVIS_CI.lower() != 'true':
        return None

    ONNX = os.environ.get('TEST_ONNX')
    if ONNX and ONNX.lower() == 'true': return None

    if six.PY2: return None

    return { 'mxnet' : {
        'imagenet1k-inception-bn'      : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
        'imagenet1k-resnet-18'         : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
        'imagenet1k-resnet-152'        : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
        'squeezenet_v1.1'              : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
        'imagenet1k-resnext-101-64x4d' : [CaffeEmit, CntkEmit, CoreMLEmit, MXNetEmit, PytorchEmit, TensorflowEmit], # Keras is ok but too slow
        'imagenet1k-resnext-50'        : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
    }}


def test_mxnet():
    tester = TestModels(get_test_table())
    tester._test_function('mxnet', tester.MXNetParse)