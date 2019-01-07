from __future__ import absolute_import
from __future__ import print_function

from conversion_imagenet import TestModels

def test_tensorflow_frozen():
    tester = TestModels()
    tester._test_function('tensorflow_frozen', tester.tensorflow_frozen_parse)

def main():
    test_tensorflow_frozen()

if __name__ == '__main__':
    main()