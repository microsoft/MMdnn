from __future__ import absolute_import
from __future__ import print_function

import sys

from test_conversion_imagenet import TestModels

def test_cntk():
    try:
        tester = TestModels()
        tester._test_function('cntk', tester.CntkParse)
    except ImportError:
        print('Please install cntk! Or cntk is not supported in your platform.', file=sys.stderr)