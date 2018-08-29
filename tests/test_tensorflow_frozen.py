from __future__ import absolute_import
from __future__ import print_function

from test_conversion_imagenet import TestModels

def test_tensorflow_frozen():
    tester = TestModels()
    tester._test_function('tensorflow_frozen', tester.TensorFlowFrozenParse)