from __future__ import absolute_import
from __future__ import print_function

import os
import six
from conversion_imagenet import TestModels

def get_test_table():
    TRAVIS_CI = os.environ.get('TRAVIS')
    if not TRAVIS_CI or TRAVIS_CI.lower() != 'true':
        return None

    ONNX = os.environ.get('TEST_ONNX')
    if ONNX and ONNX.lower() == 'true':
        return { 'caffe' :
            {
                'alexnet'       : [TestModels.onnx_emit],
                'inception_v1'  : [TestModels.onnx_emit],
                'inception_v4'  : [TestModels.onnx_emit],
                'resnet152'     : [TestModels.onnx_emit],
                'squeezenet'    : [TestModels.onnx_emit],
                # 'vgg19'         : [TestModels.onnx_emit],
                'xception'      : [TestModels.onnx_emit],
            }
        }
    elif six.PY2: return { 'caffe' :
            {
                'alexnet'       : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
                'inception_v1'  : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
                'inception_v4'  : [TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
                'resnet152'     : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
                'squeezenet'    : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
                # 'voc-fcn8s'    : [TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.TensorflowEmit],
                'xception'      : [TestModels.CoreMLEmit, TestModels.CntkEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
            }
        }
    else:
        return { 'caffe' :
            {
                # 'alexnet'       : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
                'inception_v1'  : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
                'inception_v4'  : [TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
                'resnet152'     : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
                'squeezenet'    : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
                # 'vgg19'         : [TestModels.CaffeEmit, TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
                # 'voc-fcn8s'     : [TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.TensorflowEmit],
                # 'voc-fcn16s'    : [TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.TensorflowEmit],
                # 'voc-fcn32s'    : [TestModels.CntkEmit, TestModels.CoreMLEmit, TestModels.TensorflowEmit],
                'xception'      : [TestModels.CoreMLEmit, TestModels.CntkEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
            }
        }

def test_caffe():
    test_table = get_test_table()
    tester = TestModels(test_table)
    tester._test_function('caffe', tester.caffe_parse)