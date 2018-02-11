#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import os
from six.moves import xrange

from mmdnn.conversion.common.IR.IR_graph import IRGraph, IRGraphNode
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.DataStructure.emitter import Emitter
from mmdnn.conversion.common.utils import *


class Keras2Emitter(Emitter):

    dtype_map = {
        graph_pb2.DT_FLOAT16 : "float16",
        graph_pb2.DT_FLOAT32 : "float32",
        graph_pb2.DT_FLOAT64 : "float64",
        graph_pb2.DT_INT16 : "int16",
        graph_pb2.DT_INT32 : "int32",
        graph_pb2.DT_INT64 : "int64",
        graph_pb2.DT_UINT8 : "uint8",
        graph_pb2.DT_UINT16 : "uint16"
    }


    def __init__(self, model):
        super(Keras2Emitter, self).__init__()
        from six import string_types as _string_types
        if isinstance(model, _string_types):
            network_path = model
        else:
            network_path = model[0]
            weight_path = model[1]

        self.IR_graph = IRGraph(network_path)
        self.IR_graph.build()


    @property
    def header_code(self):
        return """import keras
from keras.models import Model
from keras import layers
import keras.backend as K
import numpy as np


def load_weights_from_file(weight_file):
    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


def set_layer_weights(model, weights_dict):
    for layer in model.layers:
        if layer.name in weights_dict:
            cur_dict = weights_dict[layer.name]
            current_layer_parameters = list()
            if layer.__class__.__name__ == "BatchNormalization":
                if 'scale' in cur_dict:
                    current_layer_parameters.append(cur_dict['scale'])
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
                current_layer_parameters.extend([cur_dict['mean'], cur_dict['var']])
            elif layer.__class__.__name__ == "SeparableConv2D":
                current_layer_parameters = [cur_dict['depthwise_filter'], cur_dict['pointwise_filter']]
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
            else:
                # rot weights
                current_layer_parameters = [cur_dict['weights']]
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
            model.get_layer(layer.name).set_weights(current_layer_parameters)

    return model


def KitModel(weight_file = None):
    weights_dict = load_weights_from_file(weight_file) if not weight_file == None else None
        """


    def gen_code(self, phase):
        self.add_body(0, self.header_code)
        for layer in self.IR_graph.topological_sort:
            current_node = self.IR_graph.get_node(layer)
            node_type = current_node.type

            if hasattr(self, "emit_" + node_type):
                func = getattr(self, "emit_" + node_type)
                func(current_node)
            else:
                print("KerasEmitter has not supported operator [%s]." % (node_type))
                self.emit_UNKNOWN(current_node)

        self.add_body(1, "{:<15} = Model(inputs = [{}], outputs = [{}])".format(
            "model",
            ', '.join([self.IR_graph.get_node(i).real_variable_name for i in self.IR_graph.input_layers]),
            ', '.join([self.IR_graph.get_node(i).real_variable_name for i in self.IR_graph.output_layers])))
        self.add_body(1, ["set_layer_weights(model, weights_dict)", "return model"])

        for i in self.used_layers:
            func = getattr(self, "_layer_" + i)
            func()

        return self.body_code


    @staticmethod
    def shapeToStr(shapes):
        return ', '.join('%s' % i for i in filter(lambda x:x > 0, shapes))


    def _emit_activation(self, IR_node, op):
        self.add_body(1, "{:<15} = layers.Activation(name = '{}', activation = '{}')({})".format(
            IR_node.variable_name,
            IR_node.name,
            op,
            self.parent_variable_name(IR_node)))


    def _emit_merge(self, IR_node, func):
        inputs = ', '.join('%s' % self.IR_graph.get_node(i).real_variable_name for i in IR_node.in_edges)
        axis = ' axis = {},'.format(IR_node.get_attr('axis')) if 'axis' in IR_node.layer.attr else ""
        self.add_body(1, "{:<15} = layers.{}(name = '{}',{} inputs = [{}])".format(
            IR_node.variable_name,
            func,
            IR_node.name,
            axis,
            inputs))


    @staticmethod
    def _convert_padding(padding):
        padding = convert_onnx_pad_to_tf(padding)[1:-1]
        for idx, pad in enumerate(padding):
            padding[idx] = tuple(pad)
        padding = tuple(padding)
        return padding


    def _defuse_padding(self, IR_node):
        auto_pad = IR_node.get_attr('auto_pad')
        if auto_pad:
            input_node = self.parent_variable_name(IR_node)
            if auto_pad == 'VALID':
                padding = 'valid'
            elif auto_pad.startswith("SAME"):
                padding = 'same'
            else:
                assert False
            return input_node, padding

        else:
            padding = IR_node.get_attr("pads")
            padding = self._convert_padding(padding)
            if is_valid_padding(padding) == False:
                input_node = IR_node.variable_name + '_input'
                self.add_body(1, "{:<15} = layers.ZeroPadding{}D(padding = {})({})".format(
                    input_node,
                    len(padding),
                    padding,
                    self.parent_variable_name(IR_node)))
            else:
                input_node = self.parent_variable_name(IR_node)

            return input_node, 'valid'


    def _emit_convolution(self, IR_node, conv_type):
        self.used_layers.add('Conv')
        # assert IR_node.get_attr('group', 1) == 1
        group = IR_node.get_attr("group", 1)

        if conv_type.endswith('Transpose'):
            filters = IR_node.get_attr('kernel_shape')[-2]
        else:
            filters = IR_node.get_attr('kernel_shape')[-1]

        filters_str = 'filters={}'.format(filters) if conv_type.startswith('layer') else 'depth_multiplier={}'.format(filters)

        input_node, padding = self._defuse_padding(IR_node)

        dilations = IR_node.get_attr('dilations')
        if not dilations:
            dilations = [1] * len(IR_node.get_attr('kernel_shape'))

        self.add_body(1, "{:<15} = convolution(weights_dict, name='{}', input={}, group={}, conv_type='{}', {}, kernel_size={}, strides={}, dilation_rate={}, padding='{}', use_bias={})".format(
            IR_node.variable_name,
            IR_node.name,
            input_node,
            group,
            conv_type,
            filters_str,
            tuple(IR_node.get_attr('kernel_shape')[:-2]),
            tuple(IR_node.get_attr('strides')[1:-1]),
            tuple(dilations[1:-1]),
            padding,
            IR_node.get_attr('use_bias')))


    def emit_ConvTranspose(self, IR_node):
        dim = len(IR_node.get_attr('kernel_shape')) - 2
        self._emit_convolution(IR_node, 'layers.Conv{}DTranspose'.format(dim))


    def emit_Conv(self, IR_node):
        dim = len(IR_node.get_attr('kernel_shape')) - 2
        self._emit_convolution(IR_node, 'layers.Conv{}D'.format(dim))


    #############
    # Operators #
    #############

    def emit_UNKNOWN(self, IR_node):
        print (IR_node.name)


    def emit_Add(self, IR_node):
        self._emit_merge(IR_node, "add")


    def emit_DataInput(self, IR_node):
        shape_str = IRGraph.shapeToStr(IR_node.IR_layer.attr["shape"].shape)
        dtype_str = ", dtype = '{}'".format(self.dtype_map[IR_node.layer.attr['dtype'].type]) if 'dtype' in IR_node.layer.attr else ""
        self.add_body(1, "{:<15} = layers.Input(name = '{}', shape = ({},) {})".format(
            IR_node.variable_name,
            IR_node.name,
            shape_str,
            dtype_str))


    def emit_Dropout(self, IR_node):
        seed = 'None'
        if 'seed' in IR_node.IR_layer.attr:
            seed = IR_node.IR_layer.attr['seed'].i

        self.add_body(1, "{:<15} = layers.Dropout(name = '{}', rate = {}, seed = {})({})".format(
            IR_node.variable_name,
            IR_node.name,
            IR_node.IR_layer.attr["keep_prob"].f,
            seed,
            self.parent_variable_name(IR_node)))


    def emit_FullyConnected(self, IR_node):
        self.add_body(1, "{:<15} = layers.Dense(name = '{}', units = {}, use_bias = {})({})".format(
            IR_node.variable_name,
            IR_node.name,
            IR_node.get_attr('units'),
            IR_node.get_attr('use_bias'),
            self.parent_variable_name(IR_node)))


    def emit_Flatten(self, IR_node):
        self.used_layers.add('Flatten')
        self.add_body(1, "{:<15} = __flatten(name = '{}', input = {})".format(
            IR_node.variable_name,
            IR_node.name,
            self.parent_variable_name(IR_node)))


    def emit_Pool(self, IR_node):
        dim = len(IR_node.get_attr("strides")) - 2

        pooling_type = IR_node.get_attr('pooling_type')
        if  pooling_type == "MAX":
            pool_name = "MaxPooling{}D".format(dim)
        elif pooling_type == "AVG":
            pool_name = "AveragePooling{}D".format(dim)
        else:
            assert False

        if IR_node.layer.attr['global_pooling'].b:
            self.add_body(1, "{:<15} = layers.Global{}(name = '{}')({})".format(
                IR_node.variable_name,
                pool_name,
                IR_node.name,
                self.parent_variable_name(IR_node)))

        else:
            dilations = IR_node.get_attr('dilations')
            if dilations:
                for e in IR_node.get_attr('dilations'):
                    assert e == 1

            pool_size = IR_node.get_attr('kernel_shape')[1:-1]
            pool_size = ', '.join('%s' % i for i in pool_size)
            strides = IR_node.get_attr('strides')[1:-1]
            strides = ', '.join('%s' % i for i in strides)

            input_node, padding = self._defuse_padding(IR_node)

            self.add_body(1, "{:<15} = layers.{}(name = '{}', pool_size = ({}), strides = ({}), padding = '{}')({})".format(
                IR_node.variable_name,
                pool_name,
                IR_node.name,
                pool_size,
                strides,
                padding,
                input_node))


    def emit_Reshape(self, IR_node):
        shape_str = self.shapeToStr(IR_node.IR_layer.attr["shape"].list.i)
        self.add_body(1, "{:<15} = layers.Reshape(name = '{}', target_shape = ({},))({})".format(
            IR_node.variable_name,
            IR_node.name,
            shape_str,
            self.parent_variable_name(IR_node)))


    def emit_Tanh(self, IR_node):
        self._emit_activation(IR_node, 'tanh')


    def emit_Relu(self, IR_node):
        self._emit_activation(IR_node, 'relu')


    def emit_Softmax(self, IR_node):
        self._emit_activation(IR_node, 'softmax')


    def emit_Sigmoid(self, IR_node):
        self._emit_activation(IR_node, 'sigmoid')


    def emit_Embedding(self, IR_node):
        self.add_body(1, "{:<15} = layers.Embedding(input_dim = {}, output_dim = {}, mask_zero = {})({})".format(
            IR_node.variable_name,
            IR_node.get_attr('input_dim'),
            IR_node.get_attr('output_dim'),
            IR_node.get_attr('mask_zero'),
            IR_node.in_edges[0]))


    def emit_RNNs(self, IR_node, func):
        # for Keras
        if "dropout" in IR_node.IR_layer.attr:
            dropout_str = ",dropout = {}, recurrent_dropout = {}".format(
                    IR_node.IR_layer.attr['dropout'].f,
                    IR_node.IR_layer.attr['recurrent_dropout'].f)
        else:
            dropout_str = ""

        code = "{:<15} = layers.{}(units = {}, use_bias = {} {})({})".format(
                IR_node.name,
                func,
                IR_node.IR_layer.attr['units'].i,
                IR_node.IR_layer.attr['use_bias'].b,
                dropout_str,
                IR_node.in_edges[0])

        return code


    def emit_LSTM(self, IR_node):
        return self.emit_RNNs(IR_node, "LSTM")


    def emit_GRU(self, IR_node):
        return self.emit_RNNs(IR_node, "GRU")


    def emit_Concat(self, IR_node):
        self._emit_merge(IR_node, "concatenate")


    def emit_BatchNorm(self, IR_node):
        axis = IR_node.layer.attr['axis'].i if 'axis' in IR_node.layer.attr else -1
        self.add_body(1, "{:<15} = layers.BatchNormalization(name = '{}', axis = {}, epsilon = {}, center = {}, scale = {})({})".format(
            IR_node.variable_name,
            IR_node.name,
            axis,
            IR_node.layer.attr['epsilon'].f,
            IR_node.layer.attr['bias'].b,
            IR_node.layer.attr['scale'].b,
            self.parent_variable_name(IR_node)))


    def emit_Pad(self, IR_node):
        mode = IR_node.get_attr('mode', 'constant')
        if mode == "constant":
            func = "ZeroPadding"
        else:
            print (mode)
            raise NotImplementedError()

        dim = len(IR_node.get_attr('pads')) // 2 - 2

        padding = self._convert_padding(IR_node.get_attr('pads'))
        self.add_body(1, "{:<15} = layers.{}{}D(name='{}', padding={})({})".format(
            IR_node.variable_name,
            func,
            dim,
            IR_node.name,
            padding,
            self.parent_variable_name(IR_node)))


    def emit_Squeeze(self, IR_node):
        self.emit_Flatten(IR_node)


    def emit_ReduceMean(self, IR_node):
        axes = ', '.join('%s' % i for i in IR_node.get_attr('axes'))
        self.add_body(1,"{:<15} = layers.Lambda(lambda x: K.mean(x, axis=[{}], keepdims={}))({})".format(
            IR_node.variable_name,
            axes,
            IR_node.get_attr('keepdims'),
            self.parent_variable_name(IR_node)))


    def emit_LRN(self, IR_node):
        self.used_layers.add(IR_node.type)
        self.add_body(1, "{:<15} = LRN(size = {}, alpha = {}, beta = {}, k = {}, name = '{}')({})".format(
            IR_node.variable_name,
            IR_node.get_attr('size'),
            IR_node.get_attr('alpha'),
            IR_node.get_attr('beta'),
            IR_node.get_attr('k'),
            IR_node.name,
            self.parent_variable_name(IR_node)))


    def emit_SeparableConv(self, IR_node):
        assert len(IR_node.get_attr("strides")) == 4
        return self._emit_convolution(IR_node, "layers.SeparableConv2D")


    def emit_Relu6(self, IR_node):
        self.add_body(1, "{:<15} = layers.Activation(keras.applications.mobilenet.relu6, name = '{}')({})".format(
            IR_node.variable_name,
            IR_node.name,
            self.IR_graph.get_node(IR_node.in_edges[0]).real_variable_name))


    def emit_DepthwiseConv(self, IR_node):
        self._emit_convolution(IR_node, 'keras.applications.mobilenet.DepthwiseConv2D')

    def emit_Crop(self, IR_node):
        border = IR_node.get_attr('border')
        rank = len(border) // 2
        cropping = []
        for idx in xrange(rank):
            cropping.append(tuple([border[idx * 2], border[idx * 2 + 1]]))

        self.add_body(1, "{:<15} = layers.Cropping{}D(cropping={}, name='{}')({})".format(
            IR_node.variable_name,
            rank,
            tuple(cropping),
            IR_node.name,
            self.parent_variable_name(IR_node)))


    def _layer_Flatten(self):
        self.add_body(0, '''
def __flatten(name, input):
    if input.shape.ndims > 2: return layers.Flatten(name = name)(input)
    else: return input
''')

    def _layer_LRN(self):
        self.add_body(0, '''
from keras.layers.core import Layer
class LRN(Layer):

    def __init__(self, size=5, alpha=0.0005, beta=0.75, k=2, **kwargs):
        self.n = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        super(LRN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        super(LRN, self).build(input_shape)

    def call(self, x, mask=None):
        half_n = self.n - 1
        squared = K.square(x)
        scale = self.k
        norm_alpha = self.alpha / (2 * half_n + 1)
        if K.image_dim_ordering() == "th":
            b, f, r, c = self.shape
            squared = K.expand_dims(squared, 0)
            squared = K.spatial_3d_padding(squared, padding=((half_n, half_n), (0, 0), (0,0)))
            squared = K.squeeze(squared, 0)
            for i in range(half_n*2+1):
                scale += norm_alpha * squared[:, i:i+f, :, :]
        else:
            b, r, c, f = self.shape
            squared = K.expand_dims(squared, -1)
            squared = K.spatial_3d_padding(squared, padding=((0, 0), (0,0), (half_n, half_n)))
            squared = K.squeeze(squared, -1)
            for i in range(half_n*2+1):
                scale += norm_alpha * squared[:, :, :, i:i+f]

        scale = K.pow(scale, self.beta)
        return x / scale

    def compute_output_shape(self, input_shape):
        return input_shape''')


    def _layer_Conv(self):
        self.add_body(0, """
def convolution(weights_dict, name, input, group, conv_type, filters=None, **kwargs):
    if not conv_type.startswith('layer'):
        layer = keras.applications.mobilenet.DepthwiseConv2D(name=name, **kwargs)(input)
        return layer

    grouped_channels = int(filters / group)
    group_list = []

    if group == 1:
        func = getattr(layers, conv_type.split('.')[-1])
        layer = func(name = name, filters = filters, **kwargs)(input)
        return layer

    weight_groups = list()
    if not weights_dict == None:
        w = np.array(weights_dict[name]['weights'])
        weight_groups = np.split(w, indices_or_sections=group, axis=-1)

    for c in range(group):
        x = layers.Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(input)
        x = layers.Conv2D(name=name + "_" + str(c), filters=grouped_channels, **kwargs)(x)
        weights_dict[name + "_" + str(c)] = dict()
        weights_dict[name + "_" + str(c)]['weights'] = weight_groups[c]

        group_list.append(x)

    layer = layers.concatenate(group_list, axis = -1)

    if 'bias' in weights_dict[name]:
        b = K.variable(weights_dict[name]['bias'], name = name + "_bias")
        layer = layer + b
    return layer""")