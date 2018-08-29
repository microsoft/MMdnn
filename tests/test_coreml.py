from __future__ import absolute_import
from __future__ import print_function

import sys
from test_conversion_imagenet import TestModels

def test_coreml():
    from coremltools.models.utils import macos_version
    if macos_version() < (10, 13):
        print('Coreml is not supported in your platform.', file=sys.stderr)
    else:
        tester = TestModels()
        tester._test_function('coreml', tester.CoreMLParse)

