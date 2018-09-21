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

    return { 'pytorch' : {
        'alexnet'     : [TestModels.CaffeEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
        'densenet201' : [TestModels.CaffeEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
        'inception_v3': [TestModels.CaffeEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
        'resnet152'   : [TestModels.CaffeEmit, TestModels.CoreMLEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.PytorchEmit, TestModels.TensorflowEmit],
    }}

def test_pytorch():
    test_table = get_test_table()
    tester = TestModels(test_table)
    tester._test_function('pytorch', tester.PytorchParse)


# def main():
#     test_pytorch()

# if __name__ == '__main__':
#     main()