from __future__ import absolute_import
from __future__ import print_function

import sys

from test_conversion_imagenet import TestModels

def test_darknet():
    try:
        tester = TestModels()
        tester._test_function('darknet', tester.DarknetParse)
    except ImportError:
        print('Please install Darknet! Or Darknet is not supported in your platform.', file=sys.stderr)
