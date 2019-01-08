from __future__ import absolute_import
from __future__ import print_function

import os
import sys
from conversion_imagenet import TestModels
from conversion_imagenet import is_paddle_supported

def test_paddle():
    if not is_paddle_supported():
        return
    # omit tensorflow lead to crash
    import tensorflow as tf
    tester = TestModels()
    tester._test_function('paddle', tester.paddle_parse)


if __name__ == '__main__':
    test_paddle()
