from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import unittest
import numpy as np

from mmdnn.conversion.examples.imagenet_test import TestKit


def _compute_SNR(x,y):
    noise = x - y
    noise_var = np.sum(noise ** 2) / len(noise) + 1e-7
    signal_energy = np.sum(y ** 2) / len(y)
    max_signal_energy = np.amax(y ** 2)
    SNR = 10 * np.log10(signal_energy / noise_var)
    PSNR = 10 * np.log10(max_signal_energy / noise_var)
    return SNR, PSNR


def _compute_max_relative_error(x, y):
    from six.moves import xrange
    rerror = 0
    index = 0
    for i in xrange(len(x)):
        den = max(1.0, np.abs(x[i]), np.abs(y[i]))
        if np.abs(x[i]/den - y[i] / den) > rerror:
            rerror = np.abs(x[i] / den - y[i] / den)
            index = i
    return rerror, index


def _compute_L1_error(x, y):
    return np.linalg.norm(x - y, ord=1)


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def checkfrozen(f):
    if f == 'tensorflow_frozen':
        return 'tensorflow'
    else:
        return f


class CorrectnessTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        """ Set up the unit test by loading common utilities.
        """
        self.err_thresh = 0.15
        self.snr_thresh = 12
        self.psnr_thresh = 30

    def _compare_outputs(self, original_predict, converted_predict, need_assert=True):
        # Function self.assertEquals has deprecated, change to assertEqual
        if converted_predict is None and not need_assert:
            return


        # self.assertEqual(original_predict.shape, converted_predict.shape)
        original_predict = original_predict.flatten()
        converted_predict = converted_predict.flatten()
        len1 = original_predict.shape[0]
        len2 = converted_predict.shape[0]
        length = min(len1, len2)
        original_predict = np.sort(original_predict)[::-1]
        converted_predict = np.sort(converted_predict)[::-1]
        original_predict = original_predict[0:length]
        converted_predict = converted_predict[0:length]
        error, ind = _compute_max_relative_error(converted_predict, original_predict)
        L1_error = _compute_L1_error(converted_predict, original_predict)
        SNR, PSNR = _compute_SNR(converted_predict, original_predict)
        print("error:", error)
        print("L1 error:", L1_error)
        print("SNR:", SNR)
        print("PSNR:", PSNR)

        if need_assert:
            self.assertGreater(SNR, self.snr_thresh)
            self.assertGreater(PSNR, self.psnr_thresh)
            self.assertLess(error, self.err_thresh)


class TestModels(CorrectnessTest):

    image_path = "mmdnn/conversion/examples/data/seagull.jpg"
    cachedir = "tests/cache/"
    tmpdir = "tests/tmp/"

    @staticmethod
    def TensorFlowParse(architecture_name, image_path):
        from mmdnn.conversion.examples.tensorflow.extractor import tensorflow_extractor
        from mmdnn.conversion.tensorflow.tensorflow_parser import TensorflowParser

        # get original model prediction result
        original_predict = tensorflow_extractor.inference(architecture_name, None, TestModels.cachedir, image_path)
        del tensorflow_extractor

        # original to IR
        IR_file = TestModels.tmpdir + 'tensorflow_' + architecture_name + "_converted"
        parser = TensorflowParser(
            TestModels.cachedir + "imagenet_" + architecture_name + ".ckpt.meta",
            TestModels.cachedir + "imagenet_" + architecture_name + ".ckpt",
            None,
            "MMdnn_Output")
        parser.run(IR_file)
        del parser
        del TensorflowParser

        return original_predict


    @staticmethod
    def TensorFlowFrozenParse(architecture_name, image_path):
        from mmdnn.conversion.examples.tensorflow.extractor import tensorflow_extractor
        from mmdnn.conversion.tensorflow.tensorflow_frozenparser import TensorflowParser2

        # get original model prediction result
        original_predict = tensorflow_extractor.inference(architecture_name, None, TestModels.cachedir, image_path, is_frozen = True)
        para = tensorflow_extractor.get_frozen_para(architecture_name)
        del tensorflow_extractor

        # original to IR
        IR_file = TestModels.tmpdir + 'tensorflow_frozen_' + architecture_name + "_converted"
        parser = TensorflowParser2(
            TestModels.cachedir + para[0], para[1], para[2].split(':')[0], para[3].split(':')[0])
        parser.run(IR_file)
        del parser
        del TensorflowParser2

        return original_predict


    @staticmethod
    def KerasParse(architecture_name, image_path):
        from mmdnn.conversion.examples.keras.extractor import keras_extractor
        from mmdnn.conversion.keras.keras2_parser import Keras2Parser

        # download model
        model_filename = keras_extractor.download(architecture_name, TestModels.cachedir)

        # get original model prediction result
        original_predict = keras_extractor.inference(architecture_name, model_filename, TestModels.cachedir, image_path)
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
    def MXNetParse(architecture_name, image_path):
        from mmdnn.conversion.examples.mxnet.extractor import mxnet_extractor
        from mmdnn.conversion.mxnet.mxnet_parser import MXNetParser

        # download model
        architecture_file, weight_file = mxnet_extractor.download(architecture_name, TestModels.cachedir)

        # get original model prediction result
        original_predict = mxnet_extractor.inference(architecture_name, None, TestModels.cachedir, image_path)
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
    def CaffeParse(architecture_name, image_path):
        from mmdnn.conversion.examples.caffe.extractor import caffe_extractor

        # download model
        architecture_file, weight_file = caffe_extractor.download(architecture_name, TestModels.cachedir)

        # get original model prediction result
        original_predict = caffe_extractor.inference(architecture_name, (architecture_file, weight_file), TestModels.cachedir, image_path)
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
    def CntkParse(architecture_name, image_path):
        from mmdnn.conversion.examples.cntk.extractor import cntk_extractor
        from mmdnn.conversion.cntk.cntk_parser import CntkParser
        # download model
        architecture_file = cntk_extractor.download(architecture_name, TestModels.cachedir)

        # get original model prediction result
        original_predict = cntk_extractor.inference(architecture_name, architecture_file, image_path)
        del cntk_extractor

        # original to IR
        IR_file = TestModels.tmpdir + 'cntk_' + architecture_name + "_converted"
        parser = CntkParser(architecture_file)
        parser.run(IR_file)
        del parser
        del CntkParser
        return original_predict

    @staticmethod
    def CoremlParse(architecture_name, image_path):
        from mmdnn.conversion.examples.coreml.extractor import coreml_extractor
        from mmdnn.conversion.coreml.coreml_parser import CoremlParser

        # download model
        architecture_file = coreml_extractor.download(architecture_name, TestModels.cachedir)

        # get original model prediction result
        original_predict = coreml_extractor.inference(architecture_name, architecture_file, image_path)
        del coreml_extractor

         # original to IR
        IR_file = TestModels.tmpdir + 'coreml_' + architecture_name + "_converted"
        parser = CoremlParser(architecture_file)
        parser.run(IR_file)
        del parser
        del CoremlParser
        return original_predict





    @staticmethod
    def DarknetParse(architecture_name, image_path):
        ensure_dir("./data/")
        from mmdnn.conversion.examples.darknet.extractor import darknet_extractor
        from mmdnn.conversion.darknet.darknet_parser import DarknetParser
        # download model
        architecture_file = darknet_extractor.download(architecture_name, TestModels.cachedir)

        # get original model prediction result
        original_predict = darknet_extractor.inference(architecture_name, architecture_file, TestModels.cachedir, image_path)
        del darknet_extractor

        # original to IR
        IR_file = TestModels.tmpdir + 'darknet_' + architecture_name + "_converted"

        parser = DarknetParser(architecture_file[0], architecture_file[1], architecture_name)
        parser.run(IR_file)
        del parser
        del DarknetParser
        return original_predict


    @staticmethod
    def CntkEmit(original_framework, architecture_name, architecture_path, weight_path, image_path):
        from mmdnn.conversion.cntk.cntk_emitter import CntkEmitter

        # IR to code
        converted_file = original_framework + '_cntk_' + architecture_name + "_converted"
        converted_file = converted_file.replace('.', '_')
        emitter = CntkEmitter((architecture_path, weight_path))
        emitter.run(converted_file + '.py', None, 'test')
        del emitter
        del CntkEmitter

        model_converted = __import__(converted_file).KitModel(weight_path)

        func = TestKit.preprocess_func[original_framework][architecture_name]
        img = func(image_path)
        predict = model_converted.eval({model_converted.arguments[0]:[img]})
        converted_predict = np.squeeze(predict)
        del model_converted
        del sys.modules[converted_file]
        os.remove(converted_file + '.py')

        return converted_predict


    @staticmethod
    def TensorflowEmit(original_framework, architecture_name, architecture_path, weight_path, image_path):
        import tensorflow as tf
        from mmdnn.conversion.tensorflow.tensorflow_emitter import TensorflowEmitter

        original_framework = checkfrozen(original_framework)

        # IR to code
        converted_file = original_framework + '_tensorflow_' + architecture_name + "_converted"
        converted_file = converted_file.replace('.', '_')
        print(architecture_path)
        print(weight_path)
        emitter = TensorflowEmitter((architecture_path, weight_path))
        emitter.run(converted_file + '.py', None, 'test')
        del emitter
        del TensorflowEmitter

        # import converted model
        model_converted = __import__(converted_file).KitModel(weight_path)
        input_tf, model_tf = model_converted

        func = TestKit.preprocess_func[original_framework][architecture_name]
        img = func(image_path)
        input_data = np.expand_dims(img, 0)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            predict = sess.run(model_tf, feed_dict = {input_tf : input_data})
        del model_converted
        del sys.modules[converted_file]
        os.remove(converted_file + '.py')
        converted_predict = np.squeeze(predict)

        del tf

        return converted_predict


    @staticmethod
    def PytorchEmit(original_framework, architecture_name, architecture_path, weight_path, image_path):
        import torch
        from mmdnn.conversion.pytorch.pytorch_emitter import PytorchEmitter

        # IR to code
        converted_file = original_framework + '_pytorch_' + architecture_name + "_converted"
        converted_file = converted_file.replace('.', '_')
        emitter = PytorchEmitter((architecture_path, weight_path))
        emitter.run(converted_file + '.py', converted_file + '.npy', 'test')
        del emitter
        del PytorchEmitter

        # import converted model
        model_converted = __import__(converted_file).KitModel(converted_file + '.npy')
        model_converted.eval()

        func = TestKit.preprocess_func[original_framework][architecture_name]
        img = func(image_path)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0).copy()
        input_data = torch.from_numpy(img)
        input_data = torch.autograd.Variable(input_data, requires_grad = False)

        predict = model_converted(input_data)
        predict = predict.data.numpy()
        converted_predict = np.squeeze(predict)

        del model_converted
        del sys.modules[converted_file]
        del torch
        os.remove(converted_file + '.py')
        os.remove(converted_file + '.npy')

        return converted_predict


    @staticmethod
    def KerasEmit(original_framework, architecture_name, architecture_path, weight_path, image_path):
        from mmdnn.conversion.keras.keras2_emitter import Keras2Emitter

        original_framework = checkfrozen(original_framework)

        # IR to code
        converted_file = original_framework + '_keras_' + architecture_name + "_converted"
        converted_file = converted_file.replace('.', '_')
        emitter = Keras2Emitter((architecture_path, weight_path))
        emitter.run(converted_file + '.py', None, 'test')
        del emitter
        del Keras2Emitter


        # import converted model
        model_converted = __import__(converted_file).KitModel(weight_path)


        func = TestKit.preprocess_func[original_framework][architecture_name]


        img = func(image_path)
        input_data = np.expand_dims(img, 0)

        predict = model_converted.predict(input_data)

        if original_framework == "darknet":
            converted_predict = None
        else:
            converted_predict = np.squeeze(predict)

        del model_converted
        del sys.modules[converted_file]

        import keras.backend as K
        K.clear_session()

        os.remove(converted_file + '.py')

        return converted_predict


    @staticmethod
    def MXNetEmit(original_framework, architecture_name, architecture_path, weight_path, image_path):
        from mmdnn.conversion.mxnet.mxnet_emitter import MXNetEmitter
        from collections import namedtuple
        Batch = namedtuple('Batch', ['data'])

        original_framework = checkfrozen(original_framework)

        import mxnet as mx

        # IR to code
        converted_file = original_framework + '_mxnet_' + architecture_name + "_converted"
        converted_file = converted_file.replace('.', '_')
        output_weights_file = converted_file + "-0000.params"
        emitter = MXNetEmitter((architecture_path, weight_path, output_weights_file))
        emitter.run(converted_file + '.py', None, 'test')
        del emitter
        del MXNetEmitter

        # import converted model
        imported = __import__(converted_file)
        model_converted = imported.RefactorModel()
        model_converted = imported.deploy_weight(model_converted, output_weights_file)

        func = TestKit.preprocess_func[original_framework][architecture_name]
        img = func(image_path)
        img = np.transpose(img, (2, 0, 1))
        input_data = np.expand_dims(img, 0)

        model_converted.forward(Batch([mx.nd.array(input_data)]))
        predict = model_converted.get_outputs()[0].asnumpy()
        converted_predict = np.squeeze(predict)

        del model_converted
        del sys.modules[converted_file]
        del mx

        os.remove(converted_file + '.py')
        os.remove(output_weights_file)

        return converted_predict


    @staticmethod
    def CaffeEmit(original_framework, architecture_name, architecture_path, weight_path, image_path):
        import caffe
        from mmdnn.conversion.caffe.caffe_emitter import CaffeEmitter

        original_framework = checkfrozen(original_framework)

        # IR to code
        converted_file = original_framework + '_caffe_' + architecture_name + "_converted"
        converted_file = converted_file.replace('.', '_')
        emitter = CaffeEmitter((architecture_path, weight_path))
        emitter.run(converted_file + '.py', converted_file + '.npy', 'test')
        del emitter
        del CaffeEmitter

        # import converted model
        imported = __import__(converted_file)
        imported.make_net(converted_file + '.prototxt')
        imported.gen_weight(converted_file + '.npy', converted_file + '.caffemodel', converted_file + '.prototxt')
        model_converted = caffe.Net(converted_file + '.prototxt', converted_file + '.caffemodel', caffe.TEST)

        func = TestKit.preprocess_func[original_framework][architecture_name]
        img = func(image_path)
        img = np.transpose(img, [2, 0, 1])
        input_data = np.expand_dims(img, 0)

        model_converted.blobs[model_converted._layer_names[0]].data[...] = input_data
        predict = model_converted.forward()[model_converted._layer_names[-1]][0]
        converted_predict = np.squeeze(predict)

        del model_converted
        del sys.modules[converted_file]
        os.remove(converted_file + '.py')
        os.remove(converted_file + '.npy')
        os.remove(converted_file + '.prototxt')
        os.remove(converted_file + '.caffemodel')

        return converted_predict


    @staticmethod
    def CoreMLEmit(original_framework, architecture_name, architecture_path, weight_path, image_path):


        from mmdnn.conversion.coreml.coreml_emitter import CoreMLEmitter
        from coremltools.models import MLModel
        import coremltools
        from PIL import Image


        original_framework = checkfrozen(original_framework)


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
                image_input_names=image_path,
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

        from coremltools.models.utils import macos_version

        if macos_version() < (10, 13):
            return None
        else:

            from PIL import Image as pil_image
            img = pil_image.open(image_path)
            img = img.resize((size, size))

            # inference

            coreml_input = {input_name: img}
            coreml_output = model.predict(coreml_input)
            prob = coreml_output[output_name]
            prob = np.array(prob).squeeze()

            return prob

    @staticmethod
    def OnnxEmit(original_framework, architecture_name, architecture_path, weight_path, image_path):
        from mmdnn.conversion.onnx.onnx_emitter import OnnxEmitter

        original_framework = checkfrozen(original_framework)

        # IR to code
        converted_file = original_framework + '_onnx_' + architecture_name + "_converted"
        converted_file = converted_file.replace('.', '_')
        emitter = OnnxEmitter(architecture_path, weight_path)
        emitter.run(converted_file + '.py', weight_path, 'test')
        del emitter
        del OnnxEmitter

        # import converted model
        from onnx_tf.backend import prepare
        model_converted = __import__(converted_file).KitModel(weight_path)
        tf_rep = prepare(model_converted)

        func = TestKit.preprocess_func[original_framework][architecture_name]
        img = func(image_path)
        input_data = np.expand_dims(img, 0)

        predict = tf_rep.run(input_data)[0]

        del prepare
        del model_converted
        del tf_rep
        os.remove(converted_file + '.py')

        return predict

    exception_tabel = {
        'cntk_Keras_resnet18',                      # Cntk Padding is SAME_UPPER, but Keras Padding is SAME_LOWER, in first convolution layer.
        'cntk_Keras_resnet152',                     # Cntk Padding is SAME_UPPER, but Keras Padding is SAME_LOWER, in first convolution layer.
        'cntk_Tensorflow_resnet18',                 # Cntk Padding is SAME_UPPER, but Keras Padding is SAME_LOWER, in first convolution layer.
        'cntk_Tensorflow_resnet152',                # Cntk Padding is SAME_UPPER, but Keras Padding is SAME_LOWER, in first convolution layer.
        'tensorflow_Cntk_inception_v1',             # TODO
        'tensorflow_Cntk_resnet_v1_50',             # TODO
        'tensorflow_Cntk_resnet_v2_50',             # TODO
        'tensorflow_Cntk_resnet_v1_152',            # TODO
        'tensorflow_Cntk_resnet_v2_152',            # TODO
        'tensorflow_Cntk_mobilenet_v1_1.0',         # TODO
        'tensorflow_frozen_MXNet_inception_v1',     # TODO
        'tensorflow_MXNet_inception_v3',            # different after "InceptionV3/InceptionV3/Mixed_5b/Branch_3/AvgPool_0a_3x3/AvgPool". AVG POOL padding difference between these two framework.
        'caffe_Pytorch_inception_v1',               # TODO
        'caffe_Pytorch_alexnet',                    # TODO
        'caffe_Pytorch_inception_v4',               # TODO, same with caffe_Cntk_inception_v4
        'darknet_Keras_yolov2',                     # TODO,
        'darknet_Keras_yolov3',                     # TODO,
    }


    test_table = {
        'cntk' : {
            # 'alexnet'       : [CntkEmit, KerasEmit, TensorflowEmit],
            'inception_v3'  : [CntkEmit, PytorchEmit, TensorflowEmit, OnnxEmit], #TODO: Caffe, Keras, and MXNet no constant layer
            'resnet18'      : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit, OnnxEmit],
            'resnet152'     : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit, OnnxEmit],
        },

        'keras' : {
            'vgg16'        : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit, OnnxEmit],
            'vgg19'        : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit, OnnxEmit],
            'inception_v3' : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit, OnnxEmit],
            'resnet50'     : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit, OnnxEmit],
            'densenet'     : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit, OnnxEmit],
            'xception'     : [TensorflowEmit, KerasEmit, CoreMLEmit],
            'mobilenet'    : [TensorflowEmit, KerasEmit, CoreMLEmit], # TODO: MXNetEmit
            'nasnet'       : [TensorflowEmit, KerasEmit, CoreMLEmit],
            'yolo2'        : [KerasEmit],
        },

        'mxnet' : {
            'vgg19'                        : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit, OnnxEmit],
            'imagenet1k-inception-bn'      : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit, OnnxEmit],
            'imagenet1k-resnet-18'         : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit, OnnxEmit],
            'imagenet1k-resnet-152'        : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit, OnnxEmit],
            'squeezenet_v1.1'              : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit, OnnxEmit],
            'imagenet1k-resnext-101-64x4d' : [CaffeEmit, CntkEmit, CoreMLEmit, MXNetEmit, PytorchEmit, TensorflowEmit, OnnxEmit], # Keras is ok but too slow
            'imagenet1k-resnext-50'        : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit, OnnxEmit],
        },

        'caffe' : {
            'alexnet'       : [CaffeEmit, CntkEmit, CoreMLEmit, MXNetEmit, PytorchEmit, TensorflowEmit], # TODO: KerasEmit('Tensor' object has no attribute '_keras_history')
            'inception_v1'  : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
            'inception_v4'  : [CoreMLEmit, CntkEmit, KerasEmit, PytorchEmit, TensorflowEmit], # TODO MXNetEmit(Small error), CaffeEmit(Crash for shape)
            'resnet152'     : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
            'squeezenet'    : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
            'vgg19'         : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
            'voc-fcn8s'     : [CntkEmit, CoreMLEmit, TensorflowEmit],
            'voc-fcn16s'    : [CntkEmit, CoreMLEmit, TensorflowEmit],
            'voc-fcn32s'    : [CntkEmit, CoreMLEmit, TensorflowEmit],
            'xception'      : [CoreMLEmit, CntkEmit, MXNetEmit, PytorchEmit, TensorflowEmit], #  TODO: Caffe(Crash) KerasEmit(too slow)
        },

        'tensorflow' : {
            'vgg16'             : [CaffeEmit, CoreMLEmit, CntkEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
            'vgg19'             : [CaffeEmit, CoreMLEmit, CntkEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
            'inception_v1'      : [CaffeEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit], # TODO: CntkEmit
            'inception_v3'      : [CaffeEmit, CoreMLEmit, CntkEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
            'resnet_v1_50'      : [CaffeEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit], # TODO: CntkEmit
            'resnet_v1_152'     : [CaffeEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit], # TODO: CntkEmit
            'resnet_v2_50'      : [CaffeEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit], # TODO: CntkEmit
            'resnet_v2_152'     : [CaffeEmit, CoreMLEmit, CntkEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
            'mobilenet_v1_1.0'  : [CoreMLEmit, CntkEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit], # TODO: CaffeEmit(Crash)
            'mobilenet_v2_1.0_224'  : [CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit], # TODO: CaffeEmit(Crash) CntkEmit
            'nasnet-a_large'    : [MXNetEmit, PytorchEmit, TensorflowEmit], # TODO: KerasEmit(Slice Layer: https://blog.csdn.net/lujiandong1/article/details/54936185)
            # 'inception_resnet_v2' : [TensorflowEmit], # TODO PytorchEmit

        },

        'tensorflow_frozen' : {
            'inception_v1'      : [TensorflowEmit, KerasEmit, MXNetEmit, CoreMLEmit], # TODO: CntkEmit
            'inception_v3'      : [TensorflowEmit, KerasEmit, MXNetEmit, CoreMLEmit], # TODO: CntkEmit
            'mobilenet_v1_1.0'  : [TensorflowEmit, KerasEmit, MXNetEmit, CoreMLEmit]
        },

        'coreml' : {
            'inception_v3' : [CaffeEmit, CoreMLEmit, KerasEmit, MXNetEmit, TensorflowEmit],
            'mobilenet'    : [CaffeEmit, CoreMLEmit, KerasEmit, MXNetEmit, TensorflowEmit],
            'resnet50'     : [CaffeEmit, CoreMLEmit, KerasEmit, MXNetEmit, TensorflowEmit],
            'tinyyolo'     : [CoreMLEmit, KerasEmit, MXNetEmit, TensorflowEmit],
            'vgg16'        : [CaffeEmit, CoreMLEmit, KerasEmit, MXNetEmit, TensorflowEmit],
        },

        'darknet' : {
            'yolov2': [KerasEmit],
            'yolov3': [KerasEmit],
        }
    }


    @classmethod
    def _need_assert(cls, original_framework, target_framework, network_name):
        test_name = original_framework + '_' + target_framework + '_' + network_name
        if test_name in cls.exception_tabel:
            return False

        if target_framework == 'CoreML':
            from coremltools.models.utils import macos_version
            if macos_version() < (10, 13):
                return False

        return True


    def _test_function(self, original_framework, parser):
        ensure_dir(self.cachedir)
        ensure_dir(self.tmpdir)

        for network_name in self.test_table[original_framework].keys():
            print("Test {} from {} start.".format(network_name, original_framework), file=sys.stderr)

            # get original model prediction result
            original_predict = parser(network_name, self.image_path)

            IR_file = TestModels.tmpdir + original_framework + '_' + network_name + "_converted"
            for emit in self.test_table[original_framework][network_name]:
                target_framework = emit.__func__.__name__[:-4]
                print('Testing {} from {} to {}.'.format(network_name, original_framework, target_framework), file=sys.stderr)
                converted_predict = emit.__func__(
                    original_framework,
                    network_name,
                    IR_file + ".pb",
                    IR_file + ".npy",
                    self.image_path)
                self._compare_outputs(original_predict, converted_predict, self._need_assert(original_framework, target_framework, network_name))
                print('Conversion {} from {} to {} passed.'.format(network_name, original_framework, target_framework), file=sys.stderr)

            try:
                os.remove(IR_file + ".json")
            except OSError:
                pass

            os.remove(IR_file + ".pb")
            os.remove(IR_file + ".npy")
            print("Testing {} model {} passed.".format(original_framework, network_name), file=sys.stderr)

        print("Testing {} model all passed.".format(original_framework), file=sys.stderr)



    def test_caffe(self):
        try:
            import caffe
            self._test_function('caffe', self.CaffeParse)
        except ImportError:
            print('Please install caffe! Or caffe is not supported in your platform.', file=sys.stderr)


    def test_cntk(self):
        try:
            import cntk
            self._test_function('cntk', self.CntkParse)
        except ImportError:
            print('Please install cntk! Or cntk is not supported in your platform.', file=sys.stderr)


    def test_coreml(self):
        from coremltools.models.utils import macos_version
        if macos_version() < (10, 13):
            print('Coreml is not supported in your platform.', file=sys.stderr)
        else:
            self._test_function('coreml', self.CoremlParse)


    def test_keras(self):
        self._test_function('keras', self.KerasParse)


    def test_mxnet(self):
        self._test_function('mxnet', self.MXNetParse)


    def test_darknet(self):
        self._test_function('darknet', self.DarknetParse)


    def test_tensorflow(self):
        self._test_function('tensorflow', self.TensorFlowParse)


    def test_tensorflow_frozen(self):
        self._test_function('tensorflow_frozen', self.TensorFlowFrozenParse)
