from __future__ import absolute_import
from __future__ import print_function

import os
import sys
from conversion_imagenet import TestModels

def test_cntk():
    tester = TestModels()
    tester._test_function('cntk', tester.cntk_parse)


if __name__ == '__main__':
    test_cntk()
