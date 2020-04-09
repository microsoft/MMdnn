from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import six
from conversion_imagenet import TestModels

def get_test_table():
    return {
        'mxnet' : {
            # Run too slow on Travis.
            'imagenet1k-resnext-101-64x4d'      : [
                TestModels.onnx_emit,
                TestModels.caffe_emit,
                # cntk_emit OOM on Travis
                TestModels.cntk_emit,
                TestModels.coreml_emit,
                TestModels.keras_emit,
                TestModels.mxnet_emit,
                TestModels.pytorch_emit,
                TestModels.tensorflow_emit
                ]
    }}


def test_mxnet():
    test_table = get_test_table()
    tester = TestModels(test_table)
    tester._test_function('mxnet', tester.mxnet_parse)


if __name__ == '__main__':
    test_mxnet()

