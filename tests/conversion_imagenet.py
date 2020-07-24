from __future__ import absolute_import
from __future__ import print_function

import os
TEST_ONNX = os.environ.get('TEST_ONNX')
import sys
import imp
import numpy as np
from mmdnn.conversion.examples.imagenet_test import TestKit
import utils
from utils import *
from datetime import datetime


def is_paddle_supported():
    if (sys.version_info > (2, 7)):
        print('PaddlePaddle does not support Python {0}'.format(sys.version), file=sys.stderr)
        return False

    return True


def is_coreml_supported():
    import sys
    if sys.platform == 'darwin':
        import platform
        ver_str = platform.mac_ver()[0]
        if (tuple([int(v) for v in ver_str.split('.')]) >= (10, 13)):
            return True

    print('CoreML is not supported on your platform.', file=sys.stderr)
    return False


def check_env(source_framework, target_framework, model_name):
    if ((source_framework == 'paddle') or (target_framework == 'paddle')):
        if not is_paddle_supported():
            return False

    if ((source_framework == 'coreml') or (target_framework == 'coreml')):
        if not is_coreml_supported():
            return False

    return True


class TestModels(CorrectnessTest):

    image_path = "mmdnn/conversion/examples/data/seagull.jpg"
    cachedir = "tests/cache/"
    tmpdir = "tests/tmp/"
    sentence_path = "mmdnn/conversion/examples/data/one_imdb.npy"
    vocab_size = 30000

    def __init__(self, test_table=None, methodName='test_nothing'):
        super(TestModels, self).__init__(methodName)
        if test_table:
            print ("Reset the test_table!", file=sys.stderr)
            self.test_table = test_table


    @staticmethod
    def tensorflow_parse(architecture_name, test_input_path):
        from mmdnn.conversion.examples.tensorflow.extractor import tensorflow_extractor
        from mmdnn.conversion.tensorflow.tensorflow_parser import TensorflowParser

        # get original model prediction result
        original_predict = tensorflow_extractor.inference(architecture_name, None, TestModels.cachedir, test_input_path)
        del tensorflow_extractor

        # original to IR
        IR_file = TestModels.tmpdir + 'tensorflow_' + architecture_name + "_converted"
        parser = TensorflowParser(
            TestModels.cachedir + "imagenet_" + architecture_name + ".ckpt.meta",
            TestModels.cachedir + "imagenet_" + architecture_name + ".ckpt",
            ["MMdnn_Output"])
        parser.run(IR_file)
        del parser
        del TensorflowParser

        return original_predict


    @staticmethod
    def tensorflow_frozen_parse(architecture_name, test_input_path):
        from mmdnn.conversion.examples.tensorflow.extractor import tensorflow_extractor
        from mmdnn.conversion.tensorflow.tensorflow_frozenparser import TensorflowParser2

        # get original model prediction result
        original_predict = tensorflow_extractor.inference(architecture_name, None, TestModels.cachedir, test_input_path, is_frozen = True)
        para = tensorflow_extractor.get_frozen_para(architecture_name)
        del tensorflow_extractor

        # original to IR
        IR_file = TestModels.tmpdir + 'tensorflow_frozen_' + architecture_name + "_converted"
        parser = TensorflowParser2(
            TestModels.cachedir + para[0], para[1], para[2], para[3])
        parser.run(IR_file)
        del parser
        del TensorflowParser2
        return original_predict


    @staticmethod
    def keras_parse(architecture_name, test_input_path):
        from mmdnn.conversion.examples.keras.extractor import keras_extractor
        from mmdnn.conversion.keras.keras2_parser import Keras2Parser

        # download model
        model_filename = keras_extractor.download(architecture_name, TestModels.cachedir)

        # get original model prediction result
        original_predict = keras_extractor.inference(architecture_name, model_filename, TestModels.cachedir, test_input_path)
        # print(original_predict)
        del keras_extractor

        # original to IR
        IR_file = TestModels.tmpdir + 'keras_' + architecture_name + "_converted"
        parser = Keras2Parser(model_filename)
        parser.run(IR_file)
        del parser
        del Keras2Parser
        return original_predict


    @staticmethod
    def mxnet_parse(architecture_name, test_input_path):
        from mmdnn.conversion.examples.mxnet.extractor import mxnet_extractor
        from mmdnn.conversion.mxnet.mxnet_parser import MXNetParser

        # download model
        architecture_file, weight_file = mxnet_extractor.download(architecture_name, TestModels.cachedir)

        # get original model prediction result
        original_predict = mxnet_extractor.inference(architecture_name, None, TestModels.cachedir, test_input_path)
        del mxnet_extractor

        # original to IR
        import re
        if re.search('.', weight_file):
            weight_file = weight_file[:-7]
        prefix, epoch = weight_file.rsplit('-', 1)
        model = (architecture_file, prefix, epoch, [3, 224, 224])

        IR_file = TestModels.tmpdir + 'mxnet_' + architecture_name + "_converted"
        parser = MXNetParser(model)
        parser.run(IR_file)
        del parser
        del MXNetParser

        return original_predict


    @staticmethod
    def caffe_parse(architecture_name, test_input_path):
        from mmdnn.conversion.examples.caffe.extractor import caffe_extractor

        # download model
        architecture_file, weight_file = caffe_extractor.download(architecture_name, TestModels.cachedir)

        # get original model prediction result
        original_predict = caffe_extractor.inference(architecture_name, (architecture_file, weight_file), TestModels.cachedir, test_input_path)
        del caffe_extractor

        # original to IR
        from mmdnn.conversion.caffe.transformer import CaffeTransformer
        transformer = CaffeTransformer(architecture_file, weight_file, "tensorflow", None, phase = 'TEST')
        graph = transformer.transform_graph()
        data = transformer.transform_data()
        del CaffeTransformer

        from mmdnn.conversion.caffe.writer import ModelSaver, PyWriter

        prototxt = graph.as_graph_def().SerializeToString()
        IR_file = TestModels.tmpdir + 'caffe_' + architecture_name + "_converted"
        pb_path = IR_file + '.pb'
        with open(pb_path, 'wb') as of:
            of.write(prototxt)
        print ("IR network structure is saved as [{}].".format(pb_path))

        import numpy as np
        npy_path = IR_file + '.npy'
        with open(npy_path, 'wb') as of:
            np.save(of, data)
        print ("IR weights are saved as [{}].".format(npy_path))

        if original_predict.ndim == 3:
            original_predict = np.transpose(original_predict, (1, 2, 0))

        return original_predict


    @staticmethod
    def cntk_parse(architecture_name, test_input_path):
        from mmdnn.conversion.examples.cntk.extractor import cntk_extractor
        from mmdnn.conversion.cntk.cntk_parser import CntkParser
        # download model
        architecture_file = cntk_extractor.download(architecture_name, TestModels.cachedir)

        # get original model prediction result
        original_predict = cntk_extractor.inference(architecture_name, architecture_file, test_input_path)
        del cntk_extractor

        # original to IR
        IR_file = TestModels.tmpdir + 'cntk_' + architecture_name + "_converted"
        parser = CntkParser(architecture_file)
        parser.run(IR_file)
        del parser
        del CntkParser
        return original_predict


    @staticmethod
    def coreml_parse(architecture_name, test_input_path):
        from mmdnn.conversion.examples.coreml.extractor import coreml_extractor
        from mmdnn.conversion.coreml.coreml_parser import CoremlParser

        # download model
        architecture_file = coreml_extractor.download(architecture_name, TestModels.cachedir)

        # get original model prediction result
        original_predict = coreml_extractor.inference(architecture_name, architecture_file, test_input_path)
        del coreml_extractor

         # original to IR
        IR_file = TestModels.tmpdir + 'coreml_' + architecture_name + "_converted"
        parser = CoremlParser(architecture_file)
        parser.run(IR_file)
        del parser
        del CoremlParser
        return original_predict

    @staticmethod
    def paddle_parse(architecture_name, test_input_path):
        from mmdnn.conversion.examples.paddle.extractor import paddle_extractor
        from mmdnn.conversion.paddle.paddle_parser import PaddleParser

        # download model
        model_filename = paddle_extractor.download(architecture_name, TestModels.cachedir)

        # get original model prediction result
        original_predict = paddle_extractor.inference(architecture_name, model_filename, TestModels.cachedir, test_input_path)
        del paddle_extractor

        # original to IR
        IR_file = TestModels.tmpdir + 'paddle_' + architecture_name + "_converted"

        parser = PaddleParser(model_filename)
        parser.run(IR_file)
        del parser
        del PaddleParser
        return original_predict

    @staticmethod
    def pytorch_parse(architecture_name, test_input_path):
        from mmdnn.conversion.examples.pytorch.extractor import pytorch_extractor
        from mmdnn.conversion.pytorch.pytorch_parser import PytorchParser040
        from mmdnn.conversion.pytorch.pytorch_parser import PytorchParser151
        import torch
        # download model
        architecture_file = pytorch_extractor.download(architecture_name, TestModels.cachedir)


        # get original model prediction result
        original_predict = pytorch_extractor.inference(architecture_name, architecture_file, test_input_path)
        del pytorch_extractor

        # get shape
        func = TestKit.preprocess_func['pytorch'][architecture_name]

        import inspect
        funcstr = inspect.getsource(func)

        pytorch_pre = funcstr.split('(')[0].split('.')[-1]

        if len(funcstr.split(',')) == 3:
            size = int(funcstr.split('path,')[1].split(')')[0])

        elif  len(funcstr.split(',')) == 4:
            size = int(funcstr.split('path,')[1].split(',')[0])

        elif len(funcstr.split(',')) == 11:
            size = int(funcstr.split('path,')[1].split(',')[0])


         # original to IR
        IR_file = TestModels.tmpdir + 'pytorch_' + architecture_name + "_converted"
        if torch.__version__ == "0.4.0":
            parser = PytorchParser040(architecture_file, [3, size, size])
        else:
            parser = PytorchParser151(architecture_file, [3, size, size])
        parser.run(IR_file)
        del parser
        del PytorchParser040
        del PytorchParser151
        return original_predict


    @staticmethod
    def darknet_parse(architecture_name, test_input_path):
        ensure_dir("./data/")
        from mmdnn.conversion.examples.darknet.extractor import darknet_extractor
        from mmdnn.conversion.darknet.darknet_parser import DarknetParser
        # download model
        architecture_file = darknet_extractor.download(architecture_name, TestModels.cachedir)

        # get original model prediction result
        original_predict = darknet_extractor.inference(architecture_name, architecture_file, TestModels.cachedir, test_input_path)
        del darknet_extractor

        # original to IR
        IR_file = TestModels.tmpdir + 'darknet_' + architecture_name + "_converted"

        if architecture_name == "yolov3":
            start = "1"
        else:
            start = "0"

        parser = DarknetParser(architecture_file[0], architecture_file[1], start)
        parser.run(IR_file)
        del parser
        del DarknetParser
        return original_predict


    @staticmethod
    def cntk_emit(original_framework, architecture_name, architecture_path, weight_path, test_input_path):
        from mmdnn.conversion.cntk.cntk_emitter import CntkEmitter

        # IR to code
        converted_file = TestModels.tmpdir + original_framework + '_cntk_' + architecture_name + "_converted"
        converted_file = converted_file.replace('.', '_')
        emitter = CntkEmitter((architecture_path, weight_path))
        emitter.run(converted_file + '.py', None, 'test')
        del emitter
        del CntkEmitter

        model_converted = imp.load_source('CntkModel', converted_file + '.py').KitModel(weight_path)

        if 'rnn' not in architecture_name:
            func = TestKit.preprocess_func[original_framework][architecture_name]
            img = func(test_input_path)
            input_data = img
        else:
            sentence = np.load(test_input_path)
            from keras.utils import to_categorical
            input_data = to_categorical(sentence, 30000)[0]


        predict = model_converted.eval({model_converted.arguments[0]:[input_data]})
        converted_predict = np.squeeze(predict)
        del model_converted
        del sys.modules['CntkModel']
        os.remove(converted_file + '.py')

        return converted_predict


    @staticmethod
    def tensorflow_emit(original_framework, architecture_name, architecture_path, weight_path, test_input_path):
        import tensorflow as tf
        from mmdnn.conversion.tensorflow.tensorflow_emitter import TensorflowEmitter
        # IR to code
        converted_file = TestModels.tmpdir + original_framework + '_tensorflow_' + architecture_name + "_converted"
        converted_file = converted_file.replace('.', '_')

        emitter = TensorflowEmitter((architecture_path, weight_path))
        emitter.run(converted_file + '.py', None, 'test')
        del emitter
        del TensorflowEmitter

        # import converted model
        model_converted = imp.load_source('TFModel', converted_file + '.py').KitModel(weight_path)

        input_tf, model_tf = model_converted

        original_framework = checkfrozen(original_framework)

        if 'rnn' not in architecture_name:
            func = TestKit.preprocess_func[original_framework][architecture_name]
            img = func(test_input_path)
            input_data = np.expand_dims(img, 0)
        else:
            input_data = np.load(test_input_path)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            predict = sess.run(model_tf, feed_dict = {input_tf : input_data})
        del model_converted
        del sys.modules['TFModel']
        os.remove(converted_file + '.py')
        converted_predict = np.squeeze(predict)

        del tf

        return converted_predict


    @staticmethod
    def pytorch_emit(original_framework, architecture_name, architecture_path, weight_path, test_input_path):
        from mmdnn.conversion.pytorch.pytorch_emitter import PytorchEmitter

        # IR to code
        converted_file = TestModels.tmpdir + original_framework + '_pytorch_' + architecture_name + "_converted"
        converted_file = converted_file.replace('.', '_')
        emitter = PytorchEmitter((architecture_path, weight_path))
        emitter.run(converted_file + '.py', converted_file + '.npy', 'test')
        del emitter
        del PytorchEmitter

        # import converted model
        import torch
        model_converted = imp.load_source('PytorchModel', converted_file + '.py').KitModel(converted_file + '.npy')

        model_converted.eval()

        original_framework = checkfrozen(original_framework)
        if 'rnn' not in architecture_name:
            func = TestKit.preprocess_func[original_framework][architecture_name]
            img = func(test_input_path)
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, 0).copy()
            input_data = torch.from_numpy(img)
            input_data = torch.autograd.Variable(input_data, requires_grad = False)
        else:
            sentence = np.load(test_input_path)
            input_data = torch.from_numpy(sentence)
            input_data = torch.autograd.Variable(input_data, requires_grad = False)

        predict = model_converted(input_data)
        predict = predict.data.numpy()
        converted_predict = np.squeeze(predict)

        del model_converted
        del sys.modules['PytorchModel']
        del torch
        os.remove(converted_file + '.py')
        os.remove(converted_file + '.npy')

        return converted_predict


    @staticmethod
    def keras_emit(original_framework, architecture_name, architecture_path, weight_path, test_input_path):
        from mmdnn.conversion.keras.keras2_emitter import Keras2Emitter

        # IR to code
        converted_file = TestModels.tmpdir + original_framework + '_keras_' + architecture_name + "_converted"
        converted_file = converted_file.replace('.', '_')
        emitter = Keras2Emitter((architecture_path, weight_path))
        emitter.run(converted_file + '.py', None, 'test')
        del emitter
        del Keras2Emitter


        # import converted model
        model_converted = imp.load_source('KerasModel', converted_file + '.py').KitModel(weight_path)

        original_framework = checkfrozen(original_framework)
        if 'rnn' not in architecture_name:
            func = TestKit.preprocess_func[original_framework][architecture_name]
            img = func(test_input_path)
            input_data = np.expand_dims(img, 0)
        else:
            input_data = np.load(test_input_path)

        predict = model_converted.predict(input_data)

        if original_framework == "darknet":
            converted_predict = None
        else:
            converted_predict = np.squeeze(predict)

        del model_converted
        del sys.modules['KerasModel']

        import keras.backend as K
        K.clear_session()

        os.remove(converted_file + '.py')

        return converted_predict


    @staticmethod
    def mxnet_emit(original_framework, architecture_name, architecture_path, weight_path, test_input_path):
        from mmdnn.conversion.mxnet.mxnet_emitter import MXNetEmitter
        from collections import namedtuple
        Batch = namedtuple('Batch', ['data'])

        import mxnet

        # IR to code
        converted_file = TestModels.tmpdir + original_framework + '_mxnet_' + architecture_name + "_converted"
        converted_file = converted_file.replace('.', '_')
        output_weights_file = converted_file + "-0000.params"
        emitter = MXNetEmitter((architecture_path, weight_path, output_weights_file))
        emitter.run(converted_file + '.py', None, 'test')
        del emitter
        del MXNetEmitter

        # import converted model
        imported = imp.load_source('MXNetModel', converted_file + '.py')

        model_converted = imported.RefactorModel()
        model_converted = imported.deploy_weight(model_converted, output_weights_file)

        original_framework = checkfrozen(original_framework)
        if 'rnn' not in architecture_name:
            func = TestKit.preprocess_func[original_framework][architecture_name]
            img = func(test_input_path)
            img = np.transpose(img, (2, 0, 1))
            input_data = np.expand_dims(img, 0)
        else:
            input_data = np.load(test_input_path)

        model_converted.forward(Batch([mxnet.nd.array(input_data)]))
        predict = model_converted.get_outputs()[0].asnumpy()
        converted_predict = np.squeeze(predict)

        del model_converted
        del sys.modules['MXNetModel']
        del mxnet

        os.remove(converted_file + '.py')
        os.remove(output_weights_file)

        return converted_predict


    @staticmethod
    def caffe_emit(original_framework, architecture_name, architecture_path, weight_path, test_input_path):
        try:
            import caffe
            from mmdnn.conversion.caffe.caffe_emitter import CaffeEmitter

            # IR to code
            converted_file = TestModels.tmpdir + original_framework + '_caffe_' + architecture_name + "_converted"
            converted_file = converted_file.replace('.', '_')
            emitter = CaffeEmitter((architecture_path, weight_path))
            emitter.run(converted_file + '.py', converted_file + '.npy', 'test')
            del emitter
            del CaffeEmitter

            # import converted model
            imported = imp.load_source('CaffeModel', converted_file + '.py')

            imported.make_net(converted_file + '.prototxt')
            imported.gen_weight(converted_file + '.npy', converted_file + '.caffemodel', converted_file + '.prototxt')
            model_converted = caffe.Net(converted_file + '.prototxt', converted_file + '.caffemodel', caffe.TEST)

            original_framework = checkfrozen(original_framework)
            func = TestKit.preprocess_func[original_framework][architecture_name]
            img = func(test_input_path)
            img = np.transpose(img, [2, 0, 1])
            input_data = np.expand_dims(img, 0)

            model_converted.blobs[model_converted.inputs[0]].data[...] = input_data
            predict = model_converted.forward()[model_converted.outputs[-1]]
            converted_predict = np.squeeze(predict)

            del model_converted
            del sys.modules['CaffeModel']
            del caffe
            os.remove(converted_file + '.py')
            os.remove(converted_file + '.npy')
            os.remove(converted_file + '.prototxt')
            os.remove(converted_file + '.caffemodel')

            return converted_predict

        except ImportError:
            print ("Cannot import Caffe. Caffe Emit is not tested.")
            return None


    @staticmethod
    def coreml_emit(original_framework, architecture_name, architecture_path, weight_path, test_input_path):
        from mmdnn.conversion.coreml.coreml_emitter import CoreMLEmitter
        from coremltools.models import MLModel
        import coremltools
        from PIL import Image

        def prep_for_coreml(prename, BGRTranspose):
            # The list is in RGB oder
            if prename == 'Standard':
                return 0.00784313725490196,-1, -1, -1
            elif prename == 'ZeroCenter' :
                return 1, -123.68, -116.779, -103.939
            elif prename == 'Identity':
                return 1, 1, 1, 1
            else:
                raise ValueError()

        # IR to Model
        # converted_file = original_framework + '_coreml_' + architecture_name + "_converted"
        # converted_file = converted_file.replace('.', '_')

        original_framework = checkfrozen(original_framework)
        func = TestKit.preprocess_func[original_framework][architecture_name]

        import inspect
        funcstr = inspect.getsource(func)

        coreml_pre = funcstr.split('(')[0].split('.')[-1]

        if len(funcstr.split(',')) == 3:
            BGRTranspose = bool(0)
            size = int(funcstr.split('path,')[1].split(')')[0])
            prep_list = prep_for_coreml(coreml_pre, BGRTranspose)
        elif  len(funcstr.split(',')) == 4:
            BGRTranspose = funcstr.split(',')[-2].split(')')[0].strip() == str(True)
            size = int(funcstr.split('path,')[1].split(',')[0])
            prep_list = prep_for_coreml(coreml_pre, BGRTranspose)

        elif len(funcstr.split(',')) == 11:
            BGRTranspose = funcstr.split(',')[-2].split(')')[0].strip() == str(True)

            size = int(funcstr.split('path,')[1].split(',')[0])
            prep_list = (   float(funcstr.split(',')[2]),
                            float(funcstr.split(',')[3].split('[')[-1]),
                            float(funcstr.split(',')[4]),
                            float(funcstr.split(',')[5].split(']')[0])
                        )

        emitter = CoreMLEmitter(architecture_path, weight_path)


        model, input_name, output_name = emitter.gen_model(
                input_names=None,
                output_names=None,
                image_input_names=test_input_path,
                is_bgr=BGRTranspose,
                red_bias=prep_list[1],
                green_bias=prep_list[2],
                blue_bias=prep_list[3],
                gray_bias=0.0,
                image_scale=prep_list[0],
                class_labels=None,
                predicted_feature_name=None,
                predicted_probabilities_output=''
            )

        input_name = str(input_name[0][0])
        output_name = str(output_name[0][0])

        # load model
        model = MLModel(model)


        # save model
        # coremltools.utils.save_spec(model.get_spec(), converted_file)

        if not is_coreml_supported():
            return None
        else:

            from PIL import Image as pil_image
            img = pil_image.open(test_input_path)
            img = img.resize((size, size))

            # inference

            coreml_input = {input_name: img}
            coreml_output = model.predict(coreml_input)
            prob = coreml_output[output_name]
            prob = np.array(prob).squeeze()

            return prob


    @staticmethod
    def onnx_emit(original_framework, architecture_name, architecture_path, weight_path, test_input_path):
        from mmdnn.conversion.onnx.onnx_emitter import OnnxEmitter

        # IR to code
        converted_file = TestModels.tmpdir + original_framework + '_onnx_' + architecture_name + "_converted"
        converted_file = converted_file.replace('.', '_')
        emitter = OnnxEmitter(architecture_path, weight_path)
        emitter.run(converted_file + '.py', converted_file + '.npy', 'test')
        del emitter
        del OnnxEmitter

        # import converted model
        from onnx_tf.backend import prepare
        model_converted = imp.load_source('OnnxModel', converted_file + '.py').KitModel(converted_file + '.npy')

        tf_rep = prepare(model_converted)

        original_framework = checkfrozen(original_framework)
        func = TestKit.preprocess_func[original_framework][architecture_name]
        img = func(test_input_path)
        input_data = np.expand_dims(img, 0)

        predict = tf_rep.run(input_data)[0]

        del prepare
        del model_converted
        del tf_rep
        del sys.modules['OnnxModel']

        os.remove(converted_file + '.py')
        os.remove(converted_file + '.npy')

        return predict


    # In case of odd number add the extra padding at the end for SAME_UPPER(eg. pads:[0, 2, 2, 0, 0, 3, 3, 0]) and at the beginning for SAME_LOWER(eg. pads:[0, 3, 3, 0, 0, 2, 2, 0])

    exception_tabel = {
        'cntk_keras_resnet18',                      # Cntk Padding is SAME_LOWER, but Keras Padding is SAME_UPPER, in first convolution layer.
        'cntk_keras_resnet152',                     # Cntk Padding is SAME_LOWER, but Keras Padding is SAME_UPPER, in first convolution layer.
        'cntk_tensorflow_resnet18',                 # Cntk Padding is SAME_LOWER, but Keras Padding is SAME_UPPER, in first convolution layer.
        'cntk_tensorflow_resnet152',                # Cntk Padding is SAME_LOWER, but Keras Padding is SAME_UPPER, in first convolution layer.
        'tensorflow_cntk_inception_v1',             # Cntk Padding is SAME_LOWER, but Tensorflow Padding is SAME_UPPER, in first convolution layer.
        'tensorflow_cntk_resnet_v1_50',             # Cntk Padding is SAME_LOWER, but Tensorflow Padding is SAME_UPPER, in first convolution layer.
        'tensorflow_cntk_resnet_v2_50',             # Cntk Padding is SAME_LOWER, but Tensorflow Padding is SAME_UPPER, in first convolution layer.
        'tensorflow_cntk_resnet_v1_152',            # Cntk Padding is SAME_LOWER, but Tensorflow Padding is SAME_UPPER, in first convolution layer.
        'tensorflow_cntk_resnet_v2_152',            # Cntk Padding is SAME_LOWER, but Tensorflow Padding is SAME_UPPER, in first convolution layer.
        'tensorflow_cntk_mobilenet_v1_1.0',         # Cntk Padding is SAME_LOWER, but Tensorflow Padding is SAME_UPPER, in first convolution layer.
        'tensorflow_cntk_mobilenet_v2_1.0_224',     # Cntk Padding is SAME_LOWER, but Tensorflow Padding is SAME_UPPER, in first convolution layer.
        'tensorflow_caffe_mobilenet_v1_1.0',        # Caffe No Relu6
        'tensorflow_caffe_mobilenet_v2_1.0_224',    # Caffe No Relu6
        'tensorflow_frozen_mxnet_inception_v1',     # different after AvgPool. AVG POOL padding difference between these two framework. MXNet AVGPooling Padding is SAME_LOWER, Tensorflow AVGPooling Padding is SAME_UPPER
        'tensorflow_mxnet_inception_v3',            # different after "InceptionV3/InceptionV3/Mixed_5b/Branch_3/AvgPool_0a_3x3/AvgPool". AVG POOL padding difference between these two framework.
        'darknet_keras_yolov2',                     # accumulation of small difference
        'darknet_keras_yolov3',                     # accumulation of small difference
    }

    if TEST_ONNX and TEST_ONNX.lower() == 'true':
        test_table = {
            'cntk' : {
                'inception_v3'  : [onnx_emit],
                'resnet18'      : [onnx_emit],
                'resnet152'     : [onnx_emit],
            },

            'keras' : {
                'vgg16'        : [onnx_emit],
                'vgg19'        : [onnx_emit],
                'inception_v3' : [onnx_emit],
                'resnet50'     : [onnx_emit],
                'densenet'     : [onnx_emit],
                # 'xception'     : [onnx_emit],
                'mobilenet'    : [onnx_emit],
                # 'nasnet'       : [onnx_emit],
                #Temporarily disable 'yolo2'        : [onnx_emit],
            },

            'mxnet' : {
                'vgg19'                        : [onnx_emit],
                'imagenet1k-inception-bn'      : [onnx_emit],
                'imagenet1k-resnet-18'         : [onnx_emit],
                'imagenet1k-resnet-152'        : [onnx_emit],
                'squeezenet_v1.1'              : [onnx_emit],
                'imagenet1k-resnext-101-64x4d' : [onnx_emit],
                'imagenet1k-resnext-50'        : [onnx_emit],
            },

            'caffe' : {
                'alexnet'       : [onnx_emit],
                'inception_v1'  : [onnx_emit],
                #Temporarily disable 'inception_v4'  : [onnx_emit],
                'resnet152'     : [onnx_emit],
                'squeezenet'    : [onnx_emit],
                'vgg19'         : [onnx_emit],
                # 'voc-fcn8s'     : [onnx_emit], # TODO: ConvTranspose, Crop
                # 'voc-fcn16s'    : [onnx_emit], # TODO: ConvTranspose, Crop
                # 'voc-fcn32s'    : [onnx_emit], # TODO: ConvTranspose, Crop
                #Temporarily disable 'xception'      : [onnx_emit],
            },

            'tensorflow' : {
                #Temporarily disable 'facenet'               : [onnx_emit],
                'vgg19'                 : [onnx_emit],
                'inception_v1'          : [onnx_emit],
                'inception_v3'          : [onnx_emit],
                # 'resnet_v1_50'          : [onnx_emit], # POOL: strides > window_shape not supported due to inconsistency between CPU and GPU implementations
                # 'resnet_v1_152'         : [onnx_emit], # POOL: strides > window_shape not supported due to inconsistency between CPU and GPU implementations
                # 'resnet_v2_50'          : [onnx_emit], # POOL: strides > window_shape not supported due to inconsistency between CPU and GPU implementations
                # 'resnet_v2_152'         : [onnx_emit], # POOL: strides > window_shape not supported due to inconsistency between CPU and GPU implementations
                'mobilenet_v1_1.0'      : [onnx_emit],
                'mobilenet_v2_1.0_224'  : [onnx_emit],
                # 'nasnet-a_large'        : [onnx_emit], # POOL: strides > window_shape not supported due to inconsistency between CPU and GPU implementations
                'inception_resnet_v2'   : [onnx_emit],
            },

            'tensorflow_frozen' : {
                'inception_v1'      : [onnx_emit],
                'inception_v3'      : [onnx_emit],
                'mobilenet_v1_1.0'  : [onnx_emit],
                #Temporarily disable 'facenet'           : [onnx_emit],
            },

            'coreml' : {
                'inception_v3' : [onnx_emit],
                'mobilenet'    : [onnx_emit],
                'resnet50'     : [onnx_emit],
                'tinyyolo'     : [onnx_emit],
                'vgg16'        : [onnx_emit],
            },

            'darknet' : {
            },

            'paddle'  : {
                'resnet50'     : [onnx_emit],
                'vgg16'        : [onnx_emit],      # First 1000 exactly the same, the last one is different
            },

            'pytorch' : {
                # TODO: coredump
            },


        }

    else:
        test_table = {
            'cntk' : {
                # 'alexnet'       : [cntk_emit, keras_emit, tensorflow_emit],
                'inception_v3'  : [cntk_emit, pytorch_emit, tensorflow_emit], #TODO: Caffe, Keras, and MXNet no constant layer
                'resnet18'      : [caffe_emit, cntk_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'resnet152'     : [caffe_emit, cntk_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
            },

            'keras' : {
                'vgg19'        : [caffe_emit, cntk_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'inception_v3' : [caffe_emit, cntk_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'resnet50'     : [caffe_emit, cntk_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'densenet'     : [caffe_emit, cntk_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'xception'     : [tensorflow_emit, keras_emit, coreml_emit],
                'mobilenet'    : [coreml_emit, keras_emit, tensorflow_emit], # TODO: mxnet_emit
                # 'nasnet'       : [tensorflow_emit, keras_emit, coreml_emit],
                #Temporarily disable 'yolo2'        : [keras_emit],
                # 'facenet'      : [tensorflow_emit, coreml_emit,mxnet_emit,keras_emit]  # TODO
            },

            'mxnet' : {
                'vgg19'                        : [caffe_emit, cntk_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'imagenet1k-inception-bn'      : [caffe_emit, cntk_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'imagenet1k-resnet-18'         : [caffe_emit, cntk_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'imagenet1k-resnet-152'        : [caffe_emit, cntk_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'squeezenet_v1.1'              : [caffe_emit, cntk_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'imagenet1k-resnext-101-64x4d' : [caffe_emit, cntk_emit, coreml_emit, mxnet_emit, pytorch_emit, tensorflow_emit], # Keras is ok but too slow
                'imagenet1k-resnext-50'        : [caffe_emit, cntk_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
            },

            'caffe' : {
                'alexnet'       : [caffe_emit, cntk_emit, coreml_emit, mxnet_emit, pytorch_emit, tensorflow_emit], # TODO: keras_emit('Tensor' object has no attribute '_keras_history')
                'inception_v1'  : [caffe_emit, cntk_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                #Temporarily disable 'inception_v4'  : [cntk_emit, coreml_emit, keras_emit, pytorch_emit, tensorflow_emit], # TODO mxnet_emit(Small error), caffe_emit(Crash for shape)
                'resnet152'     : [caffe_emit, cntk_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'squeezenet'    : [caffe_emit, cntk_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'vgg19'         : [caffe_emit, cntk_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'voc-fcn8s'     : [cntk_emit, coreml_emit, tensorflow_emit],
                'voc-fcn16s'    : [cntk_emit, coreml_emit, tensorflow_emit],
                'voc-fcn32s'    : [cntk_emit, coreml_emit, tensorflow_emit],
                #Temporarily disable 'xception'      : [coreml_emit, cntk_emit, mxnet_emit, pytorch_emit, tensorflow_emit], #  TODO: Caffe(Crash) keras_emit(too slow)
            },

            'tensorflow' : {
                'vgg19'                 : [caffe_emit, coreml_emit, cntk_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'inception_v1'          : [caffe_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit], # TODO: cntk_emit
                'inception_v3'          : [caffe_emit, coreml_emit, cntk_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'resnet_v1_152'         : [caffe_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit], # TODO: cntk_emit
                'resnet_v2_152'         : [caffe_emit, coreml_emit, cntk_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'mobilenet_v1_1.0'      : [caffe_emit, coreml_emit, cntk_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'mobilenet_v2_1.0_224'  : [caffe_emit, coreml_emit, cntk_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'nasnet-a_large'        : [mxnet_emit, pytorch_emit, tensorflow_emit], # TODO: keras_emit(Slice Layer: https://blog.csdn.net/lujiandong1/article/details/54936185)
                'inception_resnet_v2'   : [caffe_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit], #  CoremlEmit worked once, then always
                #Temporarily disable 'facenet'               : [mxnet_emit, tensorflow_emit, keras_emit, pytorch_emit, caffe_emit], # TODO: coreml_emit
                #Temporarily disable 'rnn_lstm_gru_stacked'  : [tensorflow_emit, keras_emit, pytorch_emit, mxnet_emit] #TODO cntk_emit
            },

            'tensorflow_frozen' : {
                'inception_v1'      : [tensorflow_emit, keras_emit, mxnet_emit, coreml_emit], # TODO: cntk_emit
                'inception_v3'      : [tensorflow_emit, keras_emit, mxnet_emit, coreml_emit], # TODO: cntk_emit
                'mobilenet_v1_1.0'  : [tensorflow_emit, keras_emit, mxnet_emit, coreml_emit],
                #Temporarily disable 'facenet'           : [mxnet_emit, tensorflow_emit, keras_emit, caffe_emit] # TODO: coreml_emit
            },

            'coreml' : {
                'inception_v3' : [caffe_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'mobilenet'    : [caffe_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'resnet50'     : [caffe_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                # 'tinyyolo'     : [coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'vgg16'        : [caffe_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
            },

            'darknet' : {
                'yolov2': [keras_emit],
                'yolov3': [keras_emit],
            },

            'paddle' : {
                'resnet50': [coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit], # caffe_emit crash, due to gflags_reporting.cc
                'resnet101': [coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit], # caffe_emit crash
                # 'vgg16': [tensorflow_emit],
                # 'alexnet': [tensorflow_emit]
            },

            'pytorch' : {
                'alexnet'     : [caffe_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'densenet201' : [caffe_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'inception_v3': [caffe_emit, coreml_emit, keras_emit, pytorch_emit, tensorflow_emit],  # Mxnet broken https://github.com/apache/incubator-mxnet/issues/10194
                'vgg19'       : [caffe_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'vgg19_bn'    : [caffe_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
                'resnet152'   : [caffe_emit, coreml_emit, keras_emit, mxnet_emit, pytorch_emit, tensorflow_emit],
            }
        }


    def _get_test_input(self, architecture_name):
        if 'rnn' in architecture_name:
            return self.sentence_path
        else:
            return self.image_path


    @classmethod
    def _need_assert(cls, original_framework, target_framework, network_name, original_prediction, converted_prediction):
        test_name = original_framework + '_' + target_framework + '_' + network_name
        if test_name in cls.exception_tabel:
            return False

        if target_framework == 'coreml':
            if not is_coreml_supported():
                return False

        if target_framework == 'onnx' or target_framework == 'caffe':
            if converted_prediction is None:
                return False

        return True


    def _test_function(self, original_framework, parser):
        print("[{}] Testing {} models starts.".format(datetime.now(), original_framework), file=sys.stderr)
        
        ensure_dir(self.cachedir)
        ensure_dir(self.tmpdir)

        for network_name in self.test_table[original_framework].keys():
            print("[{}] Testing {} {} starts.".format(datetime.now(), original_framework, network_name), file=sys.stderr)

            # get test input path
            test_input = self._get_test_input(network_name)

            # get original model prediction result
            original_predict = parser(network_name, test_input)


            IR_file = TestModels.tmpdir + original_framework + '_' + network_name + "_converted"
            for emit in self.test_table[original_framework][network_name]:
                if isinstance(emit, staticmethod):
                    emit = emit.__func__
                target_framework = emit.__name__[:-5]

                if (target_framework == 'coreml'):
                    if not is_coreml_supported():
                        continue

                print('[{}] Converting {} from {} to {} starts.'.format(datetime.now(), network_name, original_framework, target_framework), file=sys.stderr)
                converted_predict = emit(
                    original_framework,
                    network_name,
                    IR_file + ".pb",
                    IR_file + ".npy",
                    test_input)


                self._compare_outputs(
                    original_framework,
                    target_framework,
                    network_name,
                    original_predict,
                    converted_predict,
                    self._need_assert(original_framework, target_framework, network_name, original_predict, converted_predict)
                )
                print('[{}] Converting {} from {} to {} passed.'.format(datetime.now(), network_name, original_framework, target_framework), file=sys.stderr)

            try:
                os.remove(IR_file + ".json")
            except OSError:
                pass

            os.remove(IR_file + ".pb")
            os.remove(IR_file + ".npy")
            print("[{}] Testing {} {} passed.".format(datetime.now(), original_framework, network_name), file=sys.stderr)

        print("[{}] Testing {} models passed.".format(datetime.now(), original_framework), file=sys.stderr)


    def test_nothing(self):
        pass

    # def test_caffe(self):
    #     try:
    #         import caffe
    #         self._test_function('caffe', self.caffe_parse)
    #     except ImportError:
    #         print('Please install caffe! Or caffe is not supported in your platform.', file=sys.stderr)


    # def test_cntk(self):
    #     try:
    #         import cntk
    #         self._test_function('cntk', self.cntk_parse)
    #     except ImportError:
    #         print('Please install cntk! Or cntk is not supported in your platform.', file=sys.stderr)


    # def test_coreml(self):
    #     from coremltools.models.utils import macos_version
    #     if macos_version() < (10, 13):
    #         print('Coreml is not supported in your platform.', file=sys.stderr)
    #     else:
    #         self._test_function('coreml', self.coreml_parse)


    # def test_keras(self):
    #     self._test_function('keras', self.keras_parse)


    # def test_mxnet(self):
    #     self._test_function('mxnet', self.mxnet_parse)


    # def test_darknet(self):
    #     self._test_function('darknet', self.darknet_parse)


    # def test_paddle(self):
    #     # omit tensorflow lead to crash
    #     import tensorflow as tf
    #     try:
    #         import paddle.v2 as paddle
    #         self._test_function('paddle', self.paddle_parse)
    #     except ImportError:
    #         print('Please install Paddlepaddle! Or Paddlepaddle is not supported in your platform.', file=sys.stderr)


    # def test_pytorch(self):
    #     self._test_function('pytorch', self.pytorch_parse)


    # def test_tensorflow(self):
    #     self._test_function('tensorflow', self.tensorflow_parse)


    # def test_tensorflow_frozen(self):
    #     self._test_function('tensorflow_frozen', self.tensorflow_frozen_parse)
