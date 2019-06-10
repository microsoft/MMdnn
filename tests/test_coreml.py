from __future__ import absolute_import
from __future__ import print_function

import os
import sys
from conversion_imagenet import TestModels
from conversion_imagenet import is_coreml_supported

def test_coreml():
    if is_coreml_supported():
        tester = TestModels()
        tester._test_function('coreml', tester.coreml_parse)


if __name__ == '__main__':
    test_coreml()
