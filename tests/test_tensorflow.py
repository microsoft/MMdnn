from __future__ import absolute_import
from __future__ import print_function

import os
from test_conversion_imagenet import TestModels

def get_test_table():
    TRAVIS_CI = os.environ.get('TRAVIS')
    if not TRAVIS_CI or TRAVIS_CI.lower() != 'true':
        return None

    ONNX = os.environ.get('TEST_ONNX')
    if ONNX and ONNX.lower() == 'true':
        return None

    return { 'tensorflow' :
    {
        'vgg19'                : [TestModels.CaffeEmit, TestModels.CoreMLEmit, TestModels.CntkEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit],
        'inception_v1'         : [TestModels.CaffeEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
        'inception_v3'         : [TestModels.CaffeEmit, TestModels.CoreMLEmit, TestModels.CntkEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
        'resnet_v1_152'        : [TestModels.CaffeEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
        'resnet_v2_152'        : [TestModels.CaffeEmit, TestModels.CoreMLEmit, TestModels.CntkEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
        'mobilenet_v1_1.0'     : [TestModels.CoreMLEmit, TestModels.CntkEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
        'mobilenet_v2_1.0_224' : [TestModels.CoreMLEmit, TestModels.CntkEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
        #'nasnet-a_large'       : [TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
        #'inception_resnet_v2'  : [TestModels.CaffeEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
    }}


def test_tensorflow():
    test_table = get_test_table()
    tester = TestModels(test_table)
    tester._test_function('tensorflow', tester.TensorFlowParse)
