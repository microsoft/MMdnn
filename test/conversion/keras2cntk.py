import os
import keras
import tensorflow as tf
import unittest
import numpy as np
from imp import reload
from mmdnn.conversion.examples.imagenet_test import TestKit
from mmdnn.conversion.examples.keras.extractor import keras_extractor
from mmdnn.conversion.keras.keras2_parser import Keras2Parser
from mmdnn.conversion.cntk.cntk_emitter import CntkEmitter
from mmdnn.conversion.tensorflow.tensorflow_emitter import TensorflowEmitter

_Emitter = {
    'caffe'       : "CaffeEmitter",
    'cntk'        : "CntkEmitter",
    'coreml'      : "",
    'keras'       : "Keras2Emitter",
    'mxnet'       : "MXNetEmitter",
    'pytorch'     : "PytorchEmitter",
    'tensorflow'  : "TensorflowEmitter"
}
image_name = "mmdnn/conversion/examples/data/seagull.jpg"

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def _compute_SNR(x,y):
    noise = x - y
    noise_var = np.sum(noise ** 2)/len(noise) + 1e-7
    signal_energy = np.sum(y ** 2)/len(y)
    max_signal_energy = np.amax(y ** 2)
    SNR = 10 * np.log10(signal_energy/noise_var)
    PSNR = 10 * np.log10(max_signal_energy/noise_var)   
    return SNR, PSNR  

def _compute_max_relative_error(x,y):
    rerror = 0
    index = 0
    for i in range(len(x)):
        den = max(1.0, np.abs(x[i]), np.abs(y[i]))
        if np.abs(x[i]/den - y[i]/den) > rerror:
            rerror = np.abs(x[i]/den - y[i]/den)
            index = i
    return rerror, index  

class CorrectnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """ Set up the unit test by loading common utilities.
        """
        self.err_thresh = 0.15
        self.snr_thresh = 12
        self.psnr_thresh = 30
        # self.red_bias = -1
        # self.blue_bias = -1
        # self.green_bias = -1
        # self.image_scale = 2.0/255

    def _compare_outputs(self, keras_out, cntk_out):
        self.assertEquals(len(keras_out), len(cntk_out))
        error, ind = _compute_max_relative_error(cntk_out, keras_out)    
        SNR, PSNR = _compute_SNR(cntk_out, keras_out)
        print("error:", error)
        print("SNR:", SNR)
        print("PSNR:", PSNR)
        self.assertGreater(SNR, self.snr_thresh)
        self.assertGreater(PSNR, self.psnr_thresh)
        self.assertLess(error, self.err_thresh)

class TestModels(CorrectnessTest):         
    def test_keras(self):
        filename = "test/model/"
        ensure_dir(filename)
        # keras original      
        framework_list = ['tensorflow']  
        # network_name_list = ['resnet50','vgg19', 'vgg16','inception_v3'] 
        network_name_list = [ 'xception', 'mobilenet']
        for framework in framework_list:
            for network_name in network_name_list:
                original_predict = keras_extractor.inference(network_name, image_name)

                # target framework
                keras_extractor.download(network_name)
                model2parser = "test/model/imagenet_{}.h5".format(network_name)

                # to IR
                parser = Keras2Parser(model2parser)
                parser.gen_IR()        
                parser.save_to_proto("test/model/" + network_name + "_converted.pb")
                parser.save_weights("test/model/" + network_name + "_converted.npy")

                # to code
                Emitter_cls_name = _Emitter[framework]
                parameter_emitter = '(("test/model/" + network_name + "_converted.pb", "test/model/" + network_name + "_converted.npy"))'
                emitter = eval(Emitter_cls_name + parameter_emitter)
                # emitter = CntkEmitter("test/model/" + network_name + "_converted.pb", "test/model/" + network_name + "_converted.npy")
                emitter.run("converted_model.py", None, 'test')


                # import converted model
                import converted_model
                reload (converted_model)
                model_converted = converted_model.KitModel("test/model/" + network_name + "_converted.npy")

                func = TestKit.preprocess_func['keras'][network_name]
                img = func(image_name)

                if framework == 'cntk':
                    predict = model_converted.eval({model_converted.arguments[0]:[img]})
                    
                elif framework == 'tensorflow':
                    input_tf, model_tf = model_converted
                    input_data = np.expand_dims(img, 0)
                    with tf.Session() as sess:
                        init = tf.global_variables_initializer()
                        sess.run(init)
                        predict = sess.run(model_tf, feed_dict = {input_tf : input_data})

                # print(predict)
                converted_predict = np.squeeze(predict)                    
                self._compare_outputs(original_predict, converted_predict)
                os.remove("test/model/" + network_name + "_converted.pb")
                os.remove("test/model/" + network_name + "_converted.npy")
                os.remove("converted_model.py")

