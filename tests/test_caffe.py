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
                'inception_v1'  : [TestModels.caffe_emit, TestModels.cntk_emit, TestModels.coreml_emit, TestModels.keras_emit, TestModels.mxnet_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
                'resnet152'     : [TestModels.caffe_emit, TestModels.cntk_emit, TestModels.coreml_emit, TestModels.keras_emit, TestModels.mxnet_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
                'xception'      : [TestModels.coreml_emit, TestModels.cntk_emit, TestModels.tensorflow_emit],

            }
        }
    else:
        return { 'caffe' :
            {
                # 'alexnet'       : [TestModels.caffe_emit, TestModels.cntk_emit, TestModels.coreml_emit, TestModels.mxnet_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
                'inception_v1'  : [TestModels.caffe_emit, TestModels.cntk_emit, TestModels.coreml_emit, TestModels.keras_emit, TestModels.mxnet_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
                'resnet152'     : [TestModels.caffe_emit, TestModels.cntk_emit, TestModels.coreml_emit, TestModels.keras_emit, TestModels.mxnet_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
                'xception'      : [TestModels.coreml_emit, TestModels.cntk_emit, TestModels.tensorflow_emit],
            }
        }

def test_caffe():
    test_table = get_test_table()
    tester = TestModels(test_table)
    tester._test_function('caffe', tester.caffe_parse)


if __name__ == '__main__':
    test_caffe()