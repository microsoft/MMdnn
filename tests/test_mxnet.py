from __future__ import absolute_import
from __future__ import print_function


from test_conversion_imagenet import TestModels

def test_mxnet():
    tester = TestModels()
    tester._test_function('mxnet', tester.MXNetParse)