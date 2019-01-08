from __future__ import absolute_import
from __future__ import print_function

import os
import sys
from conversion_imagenet import TestModels

def test_paddle():
    if (sys.version_info > (2, 7)):
        print('PaddlePaddle does not support Python {0}'.format(sys.version), file=sys.stderr)
        return
    # omit tensorflow lead to crash
    import tensorflow as tf
    tester = TestModels()
    tester._test_function('paddle', tester.paddle_parse)


if __name__ == '__main__':
    test_paddle()
