from __future__ import absolute_import
from __future__ import print_function

from test_conversion_imagenet import TestModels
import os

def get_test_table():
    
    TRAVIS_CI = os.environ.get('TRAVIS')
    if not TRAVIS_CI or TRAVIS_CI.lower() != 'true':
        return None

    ONNX = os.environ.get('TEST_ONNX')
    if ONNX and ONNX.lower() == 'true':
        return None

    return { 'tensorflow_frozen' : {
                'inception_v1'      : [TestModels.TensorflowEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.CoreMLEmit], # TODO: CntkEmit
                'inception_v3'      : [TestModels.TensorflowEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.CoreMLEmit], # TODO: CntkEmit
                'mobilenet_v1_1.0'  : [TestModels.TensorflowEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.CoreMLEmit],
                'facenet'           : [TestModels.TensorflowEmit, TestModels.KerasEmit, TestModels.MXNetEmit, TestModels.CoreMLEmit]
        }}

def test_tensorflow_frozen():
    # test_table = get_test_table()
    tester = TestModels()
    tester._test_function('tensorflow_frozen', tester.TensorFlowFrozenParse)

def main():
    test_tensorflow_frozen()

if __name__ == '__main__':
    main()