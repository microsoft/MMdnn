from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import io
import os
import argparse
import yaml

model_template_str = '''
models:
  - model:
      name: 'vgg19'
      source: 'tensorflow'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'inception_v1'
      source: 'tensorflow'
      targets: ['onnx', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'inception_v3'
      source: 'tensorflow'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'resnet_v1_152'
      source: 'tensorflow'
      targets: ['tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'resnet_v2_152'
      source: 'tensorflow'
      targets: ['cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'mobilenet_v1_1.0'
      source: 'tensorflow'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'mobilenet_v2_1.0_224'
      source: 'tensorflow'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'nasnet-a_large'
      source: 'tensorflow'
      targets: ['tensorflow', 'mxnet', 'pytorch']
  - model:
      name: 'inception_resnet_v2'
      source: 'tensorflow'
      targets: ['onnx', 'tensorflow', 'mxnet', 'pytorch', 'keras', 'caffe']
  - model:
      name: 'facenet'
      source: 'tensorflow'
      targets: ['onnx', 'tensorflow', 'mxnet', 'pytorch', 'keras', 'caffe']
  - model:
      name: 'rnn_embedding'
      source: 'tensorflow'
      targets: ['cntk', 'tensorflow', 'mxnet', 'pytorch', 'keras']

  - model:
      name: 'inception_v1'
      source: 'tensorflow_frozen'
      targets: ['onnx', 'tensorflow', 'mxnet', 'coreml', 'keras']
  - model:
      name: 'inception_v3'
      source: 'tensorflow_frozen'
      targets: ['onnx', 'tensorflow', 'mxnet', 'coreml', 'keras']
  - model:
      name: 'mobilenet_v1_1.0'
      source: 'tensorflow_frozen'
      targets: ['onnx', 'tensorflow', 'mxnet', 'coreml', 'keras']
  - model:
      name: 'facenet'
      source: 'tensorflow_frozen'
      targets: ['onnx', 'tensorflow', 'mxnet', 'keras']

  - model:
      name: 'inception_v3'
      source: 'cntk'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet']
  - model:
      name: 'resnet18'
      source: 'cntk'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'resnet152'
      source: 'cntk'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']

  - model:
      name: 'vgg19'
      source: 'mxnet'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'imagenet1k-inception-bn'
      source: 'mxnet'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'imagenet1k-resnet-18'
      source: 'mxnet'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'imagenet1k-resnet-152'
      source: 'mxnet'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'squeezenet_v1.1'
      source: 'mxnet'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'imagenet1k-resnext-101-64x4d'
      source: 'mxnet'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'imagenet1k-resnext-50'
      source: 'mxnet'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']

  - model:
      name: 'alexnet'
      source: 'pytorch'
      targets: ['tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'densenet201'
      source: 'pytorch'
      targets: ['tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'inception_v3'
      source: 'pytorch'
      targets: ['tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'vgg19'
      source: 'pytorch'
      targets: ['tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'vgg19_bn'
      source: 'pytorch'
      targets: ['tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'resnet152'
      source: 'pytorch'
      targets: ['tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']

  - model:
      name: 'inception_v3'
      source: 'coreml'
      targets: ['onnx', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'mobilenet'
      source: 'coreml'
      targets: ['onnx', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'resnet50'
      source: 'coreml'
      targets: ['onnx', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'tinyyolo'
      source: 'coreml'
      targets: ['onnx', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras']
  - model:
      name: 'vgg16'
      source: 'coreml'
      targets: ['onnx', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']

  - model:
      name: 'vgg19'
      source: 'keras'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'inception_v3'
      source: 'keras'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'resnet50'
      source: 'keras'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'densenet'
      source: 'keras'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'xception'
      source: 'keras'
      targets: ['tensorflow', 'coreml', 'keras']
  - model:
      name: 'mobilenet'
      source: 'keras'
      targets: ['onnx', 'tensorflow', 'coreml', 'keras']
  - model:
      name: 'yolo2'
      source: 'keras'
      targets: ['onnx', 'keras']

  - model:
      name: 'alexnet'
      source: 'caffe'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'caffe']
  - model:
      name: 'inception_v1'
      source: 'caffe'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'inception_v4'
      source: 'caffe'
      targets: ['onnx', 'cntk', 'tensorflow', 'pytorch', 'coreml', 'keras']
  - model:
      name: 'resnet152'
      source: 'caffe'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'squeezenet'
      source: 'caffe'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'vgg19'
      source: 'caffe'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml', 'keras', 'caffe']
  - model:
      name: 'voc-fcn8s'
      source: 'caffe'
      targets: ['cntk', 'tensorflow', 'coreml']
  - model:
      name: 'voc-fcn16s'
      source: 'caffe'
      targets: ['cntk', 'tensorflow', 'coreml']
  - model:
      name: 'voc-fcn32s'
      source: 'caffe'
      targets: ['cntk', 'tensorflow', 'coreml']
  - model:
      name: 'xception'
      source: 'caffe'
      targets: ['onnx', 'cntk', 'tensorflow', 'mxnet', 'pytorch', 'coreml']

  - model:
      name: 'resnet50'
      source: 'paddle'
      targets: ['onnx']
  - model:
      name: 'vgg16'
      source: 'paddle'
      targets: ['onnx']

'''

code_template_str = '''
from __future__ import absolute_import
from __future__ import print_function

import os
from conversion_imagenet import TestModels
from conversion_imagenet import check_env

def get_test_table():
    return {{ '{1}' :
    {{
        '{0}'                : [TestModels.{2}_emit]
    }}}}



def test_{1}_{2}_{3}():
    if not check_env('{1}', '{2}', '{0}'):
        return

    test_table = get_test_table()
    tester = TestModels(test_table)
    tester._test_function('{1}', tester.{1}_parse)


if __name__ == '__main__':
    test_{1}_{2}_{3}()

'''

travis_template_str = '''
sudo: required
dist: xenial

os:
  - linux

language: python
python:
  - "2.7"
  - "3.5"

env:
{0}

cache:
  directories:
    - $HOME/.cache/pip

addons:
  apt:
    update: true

before_install:
  - sudo apt-get install -y openmpi-bin
  - sudo apt-get install -y libprotobuf-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
  - sudo apt-get install -y libatlas-base-dev
  - sudo apt-get install -y libgflags-dev libgoogle-glog-dev
  - if [ "$TEST_SOURCE_FRAMEWORK" = "caffe" ] || [ "$TEST_TARGET_FRAMEWORK" = "caffe" ]; then sudo apt-get install -y --no-install-recommends libboost-all-dev; fi

install:
  - pip install -q -r $(python requirements/select_requirements.py)
  - pip install wget

before_script:
  - export LD_LIBRARY_PATH=$(python -c "import os; print(os.path.dirname(os.__file__) + '/site-packages/caffe/libs')"):${{LD_LIBRARY_PATH}}

after_failure: true

after_success: true

after_script: true

script: bash test.sh $TEST_SOURCE_FRAMEWORK $TEST_TARGET_FRAMEWORK $TEST_MODEL

matrix:
  fast_finish: true

  allow_failures:
    - env: TEST_SOURCE_FRAMEWORK=paddle TEST_MODEL=resnet50
    - env: TEST_SOURCE_FRAMEWORK=paddle TEST_MODEL=vgg16

notifications:
  email:
    on_success: never
    on_failure: never

'''


def gen_test(output_dir, model):
    model_name = model['name']
    normalized_model_name = model_name.replace('.', '_')
    normalized_model_name2 = normalized_model_name.replace('-', '_')
    length = len(model['targets'])
    for i in range(length):
        test_file = os.path.join(output_dir, 'test_{0}_{1}_{2}.py'
                    .format(model['source'], model['targets'][i], normalized_model_name))
        with open(test_file, "w+") as f:
            code = code_template_str.format(model_name, model['source'], model['targets'][i], normalized_model_name2)
            f.write(code)


def gen_tests(output_dir):
    y = yaml.load(model_template_str)
    length = len(y['models'])
    for i in range(length):
        gen_test(output_dir, y['models'][i]['model'])


def gen_travis(output_dir):
    y = yaml.load(model_template_str)
    travis_file = os.path.join(output_dir, 'travis.yml')

    env_str = ''
    length = len(y['models'])
    for i in range(length):
        model = y['models'][i]['model']
        model_name = model['name']
        normalized_model_name = model_name.replace('.', '_')
        source_framework = model['source']
        if False:
            env_str += '  - TEST_SOURCE_FRAMEWORK={0} TEST_MODEL={1}\n'.format(source_framework, normalized_model_name)
        else:
            length2 = len(model['targets'])
            for j in range(length2):
                target_framework = model['targets'][j]
                env_str += '  - TEST_SOURCE_FRAMEWORK={0} TEST_TARGET_FRAMEWORK={1} TEST_MODEL={2}\n'.format(source_framework, target_framework, normalized_model_name)

    with open(travis_file, "w+") as f:
        code = travis_template_str.format(env_str)
        f.write(code)

    return


def prepare_env(FLAGS):
    output_dir = FLAGS.output_dir
    if (not os.path.exists(output_dir)):
        os.mkdir(output_dir)
    if ((not os.path.isdir(output_dir)) or (not os.path.exists(output_dir))):
        print('Cannot create target output directory: "{0}"'.format(output_dir))
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', help='The output directory.', required=True)
    FLAGS, unparsed = parser.parse_known_args()
    if (not prepare_env(FLAGS)):
        return

    output_dir = FLAGS.output_dir
    gen_travis(output_dir)
    gen_tests(output_dir)


if __name__ == '__main__':
    main()
