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

    return { 'pytorch' : {
        'alexnet'     : [TestModels.caffe_emit, TestModels.coreml_emit, TestModels.keras_emit, TestModels.mxnet_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
        'densenet201' : [TestModels.caffe_emit, TestModels.coreml_emit, TestModels.keras_emit, TestModels.mxnet_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
        'inception_v3': [TestModels.caffe_emit, TestModels.coreml_emit, TestModels.keras_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
        'resnet152'   : [TestModels.caffe_emit, TestModels.coreml_emit, TestModels.keras_emit, TestModels.mxnet_emit, TestModels.pytorch_emit, TestModels.tensorflow_emit],
    }}

def test_pytorch():
    test_table = get_test_table()
    tester = TestModels(test_table)
    tester._test_function('pytorch', tester.pytorch_parse)


# def main():
#     test_pytorch()

# if __name__ == '__main__':
#     main()