from __future__ import absolute_import
from __future__ import print_function

import os
import six
from test_conversion_imagenet import TestModels

def get_test_table():
    if six.PY2:
        return {
            'keras' : {
                'vgg19'        : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
                'inception_v3' : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
                'resnet50'     : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
                'densenet'     : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
                'xception'     : [TestModels.TensorflowEmit, TestModels.KerasEmit, TestModels.CoreMLEmit],
                'mobilenet'    : [TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.TensorflowEmit], # TODO: MXNetEmit
                # 'nasnet'       : [TensorflowEmit, KerasEmit, CoreMLEmit],
            }},
    else:
        return None


def test_keras():
    tester = TestModels(get_test_table())
    tester._test_function('keras', tester.KerasParse)