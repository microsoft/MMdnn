from __future__ import absolute_import
from __future__ import print_function

import sys

from test_conversion_imagenet import TestModels

def test_paddle():
    # omit tensorflow lead to crash
    import tensorflow as tf
    try:
        tester = TestModels()
        tester._test_function('paddle', tester.PaddleParse)
    except ImportError as error:
        print('Please install Paddlepaddle! Or Paddlepaddle is not supported in your platform.\nError:[{}]'.format(error), file=sys.stderr)


