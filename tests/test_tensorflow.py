from __future__ import absolute_import
from __future__ import print_function

import os
from conversion_imagenet import TestModels

def get_test_table():
    TRAVIS_CI = os.environ.get('TRAVIS')
    if not TRAVIS_CI or TRAVIS_CI.lower() != 'true':
        return None

    ONNX = os.environ.get('TEST_ONNX')
    if ONNX and ONNX.lower() == 'true':
        return None

    return { 'tensorflow' :
    {
        'vgg19'                : [TestModels.caffe_emit, TestModels.coreml_emit, TestModels.cntk_emit, TestModels.keras_emit, TestModels.mxnet_emit, TestModels.pytorch_emit],
        'inception_v1'         : [TestModels.caffe_emit, TestModels.coreml_emit, TestModels.keras_emit, TestModels.mxnet_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
        'inception_v3'         : [TestModels.caffe_emit, TestModels.coreml_emit, TestModels.cntk_emit, TestModels.keras_emit, TestModels.mxnet_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
        'resnet_v1_152'        : [TestModels.caffe_emit, TestModels.coreml_emit, TestModels.keras_emit, TestModels.mxnet_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
        'resnet_v2_152'        : [TestModels.caffe_emit, TestModels.coreml_emit, TestModels.cntk_emit, TestModels.keras_emit, TestModels.mxnet_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
        'mobilenet_v1_1.0'     : [TestModels.coreml_emit, TestModels.cntk_emit, TestModels.keras_emit, TestModels.mxnet_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
        'mobilenet_v2_1.0_224' : [TestModels.coreml_emit, TestModels.cntk_emit, TestModels.keras_emit, TestModels.mxnet_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
        #'nasnet-a_large'       : [TestModels.mxnet_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
        #'inception_resnet_v2'  : [TestModels.caffe_emit, TestModels.keras_emit, TestModels.mxnet_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
    }}


def test_tensorflow():
    test_table = get_test_table()
    tester = TestModels(test_table)

    tester._test_function('tensorflow', tester.TensorFlowParse)

if __name__ == "__main__":
    test_tensorflow()


