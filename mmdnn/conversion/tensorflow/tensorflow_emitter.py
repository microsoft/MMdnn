#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import os

from mmdnn.conversion.common.IR.IR_graph import IRGraph, IRGraphNode
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.DataStructure.emitter import Emitter
from mmdnn.conversion.common.utils import *


class TensorflowEmitter(Emitter):

    dtype_map = {
        graph_pb2.DT_FLOAT16 : "tf.float16",
        graph_pb2.DT_FLOAT32 : "tf.float32",
        graph_pb2.DT_FLOAT64 : "tf.float64",
        graph_pb2.DT_INT16 : "tf.int16",
        graph_pb2.DT_INT32 : "tf.int32",
        graph_pb2.DT_INT64 : "tf.int64",
        graph_pb2.DT_UINT8 : "tf.uint8",
        graph_pb2.DT_UINT16 : "tf.uint16"
    }


    @property
    def header_code(self):
        return """import tensorflow as tf

__weights_dict = dict()

is_train = {}

def load_weights(weight_file):
    import numpy as np

    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file = None):
    global __weights_dict
    __weights_dict = load_weights(weight_file)
""".format(self.trainable)


    def __init__(self, model):
        super(TensorflowEmitter, self).__init__()

        from six import string_types as _string_types
        if isinstance(model, _string_types):
            network_path = model
        else:
            network_path = model[0]
            self._load_weights(model[1])

        self.IR_graph = IRGraph(network_path)
        super(TensorflowEmitter, self)._build()


    def gen_code(self, phase):
        self.trainable = (phase == 'train')
        self.add_body(0, self.header_code)

        for layer in self.IR_graph.topological_sort:
            current_node = self.IR_graph.get_node(layer)
            node_type = current_node.type

            if hasattr(self, "emit_" + node_type):
                func = getattr(self, "emit_" + node_type)
                func(current_node)
            else:
                print("TensorflowEmitter has not supported operator [%s]." % (node_type))
                self.emit_UNKNOWN(current_node)

        self.add_body(1, "return {}, {}".format(
            ', '.join([self.IR_graph.get_node(name).real_variable_name for name in self.IR_graph.input_layers]),
            ', '.join([self.IR_graph.get_node(name).real_variable_name for name in self.IR_graph.output_layers])))

        self.add_body(0, "")
        for i in self.used_layers:
            func = getattr(self, "_layer_" + i)
            func()

        return self.body_code


    @staticmethod
    def _shapeToStr(shapes):
        ret = [dim.size if dim.size != -1 else 'None' for dim in shapes.dim]
        return ', '.join('%s' % i for i in ret)


    def emit_Conv(self, IR_node):
        self.used_layers.add(IR_node.type)
        strides_str = ', '.join('%s' % i for i in IR_node.get_attr('strides')[1:-1])
        input_node, padding = self._defuse_padding(IR_node)
        self.add_body(1, "{:<15} = convolution({}, group={}, strides=[{}], padding='{}', name='{}')".format(
            IR_node.variable_name,
            input_node,
            IR_node.get_attr('group', 1),
            strides_str,
            padding,
            IR_node.name))


    def _defuse_padding(self, IR_node, extra_str=""):
        auto_pad = IR_node.get_attr('auto_pad')
        if auto_pad:
            input_node = self.parent_variable_name(IR_node)
            if auto_pad == 'VALID':
                padding = 'VALID'
            elif auto_pad.startswith("SAME"):
                padding = 'SAME'
            else:
                raise ValueError("Unknown padding type [{}].".format(auto_pad))

            return input_node, padding

        else:
            padding = IR_node.get_attr("pads")
            padding = convert_onnx_pad_to_tf(padding)
            if not is_valid_padding(padding):
                input_node = IR_node.variable_name + '_pad'
                self.add_body(1, "{:<15} = tf.pad({}, paddings = {}{})".format(
                    input_node,
                    self.parent_variable_name(IR_node),
                    padding,
                    extra_str
                    ))
            else:
                input_node = self.parent_variable_name(IR_node)

            return input_node, 'VALID'


    def emit_Constant(self, IR_node):
        self.add_body(1, "{:<15} = tf.constant(__weights_dict['{}']['value'], name='{}')".format(
            IR_node.variable_name,
            IR_node.name,
            IR_node.name
        ))


    def emit_Pool(self, IR_node):
        pooling_type = IR_node.get_attr('pooling_type')
        if pooling_type == 'MAX':
            op = 'max_pool'
            padding_const = ", constant_values=float('-Inf')"
        elif pooling_type == 'AVG':
            op = 'avg_pool'
            padding_const = ""
        else:
            raise ValueError("unknown pooling type [{}].".format(pooling_type))

        arrlen = len(IR_node.get_attr('strides'))
        dim_str = '3d' if arrlen == 5 else ""

        if IR_node.layer.attr['global_pooling'].b:
            self.add_body(1, "{:<15} = tf.nn.{}{}({}, [1] + {}.get_shape().as_list()[1:-1] + [1], strides = [1] * {}, padding = 'VALID', name = '{}')".format(
                IR_node.variable_name,
                op,
                dim_str,
                self.parent_variable_name(IR_node),
                self.parent_variable_name(IR_node),
                arrlen,
                IR_node.name))

        else:
            dim = len(IR_node.get_attr("strides")) - 2
            dilations = IR_node.get_attr('dilations')
            if dilations:
                for e in IR_node.get_attr('dilations'):
                    assert e == 1

            pool_size = IR_node.get_attr('kernel_shape')[1:-1]

            strides = IR_node.get_attr('strides')[1:-1]
            padding = IR_node.get_attr('pads')[1:dim]

            if pooling_type == "AVG" and pool_size.count(pool_size[0]) == len(pool_size) and strides[0] == 1 and strides.count(strides[0]) == len(strides) and padding.count(padding[0]) == len(padding) and pool_size[0] == padding[0]*2 + 1:
                kernel_shape_str = ', '.join('%s' % i for i in IR_node.get_attr('kernel_shape'))
                strides_str = ', '.join('%s' % i for i in IR_node.get_attr('strides'))


                self.add_body(1, "{:<15} = tf.nn.{}{}({}, [{}], [{}], padding='{}', name='{}')".format(
                    IR_node.variable_name,
                    op,
                    dim_str,
                    self.parent_variable_name(IR_node),
                    kernel_shape_str,
                    strides_str,
                    'SAME',
                    IR_node.name))

            else:

                kernel_shape_str = ', '.join('%s' % i for i in IR_node.get_attr('kernel_shape'))
                strides_str = ', '.join('%s' % i for i in IR_node.get_attr('strides'))

                input_node, padding = self._defuse_padding(IR_node, padding_const)

                self.add_body(1, "{:<15} = tf.nn.{}{}({}, [{}], [{}], padding='{}', name='{}')".format(
                    IR_node.variable_name,
                    op,
                    dim_str,
                    input_node,
                    kernel_shape_str,
                    strides_str,
                    padding,
                    IR_node.name))


    def emit_UNKNOWN(self, IR_node):
        print(IR_node.name)


    def emit_Add(self, IR_node):
        self.add_body(1, "{:<15} = {}".format(
            IR_node.variable_name,
            ' + '.join('%s' % self.IR_graph.get_node(s).real_variable_name for s in IR_node.in_edges)))


    def emit_DataInput(self, IR_node):
        assert not IR_node.in_edges
        shape_str = self._shapeToStr(IR_node.layer.attr["shape"].shape)

        if 'dtype' in IR_node.layer.attr:
            dtype_str = "{}, ".format(self.dtype_map[IR_node.layer.attr['dtype'].type])
        else:
            dtype_str = "tf.float32,"

        code = "{:<15} = tf.placeholder({} shape = ({}), name = '{}')".format(
            IR_node.variable_name, dtype_str, shape_str, IR_node.name
        )

        self.add_body(1, code)


    def emit_Dropout(self, IR_node):
        parent = self.IR_graph.get_parent(IR_node.name, [0])
        if self.trainable:
            self.add_body(1, "{:<15} = Dropout(name = '{}', dropout_rate = {})({})".format(
                IR_node.variable_name,
                IR_node.name,
                1 - IR_node.IR_layer.attr["keep_prob"].f,
                parent.real_variable_name))
        else:
            IR_node.real_name = parent.real_name


    def emit_FullyConnected(self, IR_node):
        if IR_node.name in self.weights_dict and 'weights' in self.weights_dict[IR_node.name]:
            kernel_str = "kernel_initializer = tf.constant_initializer(__weights_dict['{}']['weights']), ".format(IR_node.name)
        else: kernel_str = ""

        if IR_node.name in self.weights_dict and 'bias' in self.weights_dict[IR_node.name]:
            bias_str = "bias_initializer = tf.constant_initializer(__weights_dict['{}']['bias']), ".format(IR_node.name)
        else: bias_str = ""

        # check whether flatten operator should be added
        parent = self.IR_graph.get_parent(IR_node.name, [0])
        parent_shape = shape_to_list(parent.get_attr('_output_shapes')[0])
        if len(parent_shape) > 2:
            # flatten is needed
            self.add_body(1, "{:<15} = tf.contrib.layers.flatten({})".format(
                IR_node.variable_name + '_flatten',
                self.parent_variable_name(IR_node)))

            code = "{:<15} = tf.layers.dense({}, {}, {}{}use_bias = {})".format(
                IR_node.variable_name,
                IR_node.variable_name + '_flatten',
                IR_node.layer.attr['units'].i,
                kernel_str,
                bias_str,
                IR_node.layer.attr['use_bias'].b)
            self.add_body(1, code)

        else:
            code = "{:<15} = tf.layers.dense({}, {}, {}{}use_bias = {})".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                IR_node.layer.attr['units'].i,
                kernel_str,
                bias_str,
                IR_node.layer.attr['use_bias'].b)
            self.add_body(1, code)


    def emit_Flatten(self, IR_node):
        #self._emit_unary_operation(IR_node, "contrib.layers.flatten")
        self.add_body(1, "{:<15} = tf.contrib.layers.flatten({})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node)))


    def emit_Mul(self, IR_node):
        self.add_body(1, "{:<15} = {}".format(
            IR_node.variable_name,
            ' * '.join('%s' % self.IR_graph.get_node(s).real_variable_name for s in IR_node.in_edges)))


    def emit_Reshape(self, IR_node):
        self.add_body(1, "{:<15} = tf.reshape({}, [{}], '{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            ', '.join('%s' % i for i in IR_node.get_attr('shape')),
            IR_node.name))


    def emit_Sub(self, IR_node):
        self.add_body(1, "{:<15} = {}".format(
            IR_node.variable_name,
            ' - '.join('%s' % self.IR_graph.get_node(s).real_variable_name for s in IR_node.in_edges)))


    def _emit_unary_operation(self, IR_node, op_name):
        self.add_body(1, "{:<15} = tf.{}({}, name = '{}')".format(
            IR_node.variable_name,
            op_name,
            self.parent_variable_name(IR_node),
            IR_node.name))


    def emit_Tanh(self, IR_node):
        self._emit_unary_operation(IR_node, 'tanh')


    def emit_Elu(self, IR_node):
        self._emit_unary_operation(IR_node, 'nn.elu')


    def emit_Relu(self, IR_node):
        self._emit_unary_operation(IR_node, 'nn.relu')


    def emit_Relu6(self, IR_node):
        self._emit_unary_operation(IR_node, 'nn.relu6')


    def emit_CRelu(self, IR_node):
        self._emit_unary_operation(IR_node, 'nn.crelu')


    def emit_PRelu(self, IR_node):
        self.used_layers.add(IR_node.type)
        self.add_body(1, "{:<15} = prelu({}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.name))

    def emit_LeakyRelu(self, IR_node):
        self.add_body(1, "{:<15} = tf.nn.leaky_relu({}, alpha={}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('alpha'),
            IR_node.name
        ))


    def emit_Softmax(self, IR_node):
        self._emit_unary_operation(IR_node, 'nn.softmax')


    def emit_Sigmoid(self, IR_node):
        self._emit_unary_operation(IR_node, 'sigmoid')


    def emit_Embedding(self, IR_node):
        raise NotImplementedError()
        ret = "{:<15} = Embedding(input_dim = {}, output_dim = {}, mask_zero = {})({})".format(
                IR_node.name,
                IR_node.IR_layer.attr['input_dim'].i,
                IR_node.IR_layer.attr['output_dim'].i,
                IR_node.IR_layer.attr['mask_zero'].b,
                IR_node.in_edges[0])

        return ret


        assert False


    def emit_LSTM(self, IR_node):
        return self.emit_RNNs(IR_node, "LSTM")


    def emit_GRU(self, IR_node):
        return self.emit_RNNs(IR_node, "GRU")


    def emit_Concat(self, IR_node):
        self.add_body(1, "{:<15} = tf.concat([{}], {}, name = '{}')".format(
            IR_node.variable_name,
            ', '.join(self.IR_graph.get_node(s).real_variable_name for s in IR_node.in_edges),
            IR_node.layer.attr['axis'].i,
            IR_node.name))


    def emit_BatchNorm(self, IR_node):
        self.used_layers.add(IR_node.type)
        self.add_body(1, "{:<15} = batch_normalization({}, variance_epsilon={}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('epsilon'),
            IR_node.name))


    def emit_Pad(self, IR_node):
        padding = IR_node.get_attr('pads')
        padding = convert_onnx_pad_to_tf(padding)

        mode = IR_node.get_attr('mode', 'constant')
        mode = mode.lower()
        if mode == 'constant' or mode == 'reflect':
            mode = mode.upper()
        elif mode == 'edge':
            mode = 'SYMMETRIC'
        else:
            raise NotImplementedError("Not support padding mode {}.".format(mode))

        self.add_body(1, "{:<15} = tf.pad({}, {}, '{}', name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            padding,
            mode,
            IR_node.variable_name))


    def emit_Squeeze(self, IR_node):
        self.add_body(1, "{:<15} = tf.squeeze({}, [{}], name = '{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            ', '.join('%s' % axis for axis in IR_node.layer.attr['axes'].list.i),
            IR_node.name))


    def emit_ReduceMean(self, IR_node):
        self.add_body(1, "{:<15} = tf.reduce_mean({}, [{}], {}, name = '{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            ','.join('%s' % i for i in IR_node.get_attr('axes')),
            IR_node.get_attr('keepdims'),
            IR_node.name))


    def emit_LRN(self, IR_node):
        self.add_body(1, "{:<15} = tf.nn.lrn({}, {}, alpha = {}, beta = {}, name = '{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('size') - 1,
            IR_node.layer.attr['alpha'].f / (IR_node.layer.attr['size'].i * 2 - 1),
            IR_node.get_attr('beta'),
            IR_node.name))


    def emit_SeparableConv(self, IR_node):
        self.used_layers.add(IR_node.type)
        strides_str = ', '.join('%s' % i for i in IR_node.get_attr('strides'))
        input_node, padding = self._defuse_padding(IR_node)
        self.add_body(1, "{:<15} = separable_convolution({}, strides = [{}], padding = '{}', name = '{}')".format(
            IR_node.variable_name,
            input_node,
            strides_str,
            padding,
            IR_node.name))


    def emit_DepthwiseConv(self, IR_node):
        self.used_layers.add(IR_node.type)
        strides_str = ', '.join('%s' % i for i in IR_node.layer.attr['strides'].list.i)
        input_node, padding = self._defuse_padding(IR_node)
        self.add_body(1, "{:<15} = depthwise_convolution({}, strides = [{}], padding = '{}', name = '{}')".format(
            IR_node.variable_name,
            input_node,
            strides_str,
            padding,
            IR_node.name))


    def emit_Crop(self, IR_node):
        border = IR_node.get_attr('border')
        assert len(border) == 4

        output_shape = IR_node.get_attr('_output_shapes')[0]
        output_shape = shape_to_list(output_shape)

        self.add_body(1, "{:<15} = tf.image.crop_to_bounding_box({}, offset_height={}, offset_width={}, target_height={}, target_width={})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            border[0],
            border[1],
            output_shape[1],
            output_shape[2]))

    def emit_ConvTranspose(self, IR_node):
        self.used_layers.add(IR_node.type)
        output_shape = [1] + shape_to_list(IR_node.get_attr('_output_shapes')[0])[1:]
        input_node, padding = self._defuse_padding(IR_node)
        self.add_body(1, "{:<15} = convolution_transpose({}, output_shape={}, strides={}, padding='{}', name='{}')".format(
            IR_node.variable_name,
            input_node,
            output_shape,
            IR_node.get_attr('strides'),
            padding,
            IR_node.name))


    def emit_Slice(self, IR_node):
        extra_str = ""
        if IR_node.get_attr('strides'):
            extra_str += ", strides={}".format(IR_node.get_attr('strides'))
        if IR_node.get_attr('begin_mask'):
            extra_str += ", begin_mask={}".format(IR_node.get_attr('begin_mask'))
        if IR_node.get_attr('end_mask'):
            extra_str += ", end_mask={}".format(IR_node.get_attr('end_mask'))
        self.add_body(1, "{:<15} = tf.strided_slice({}, {}, {} {}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('starts'),
            IR_node.get_attr('ends'),
            extra_str,
            IR_node.name))


    def emit_Split(self, IR_node):
        self.add_body(1, "{:<15} = tf.split({}, {}, {}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('split'),
            IR_node.get_attr('axis'),
            IR_node.name))


    def _layer_Conv(self):
        self.add_body(0, """
def convolution(input, name, group, **kwargs):
    w = tf.Variable(__weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    if group == 1:
        layer = tf.nn.convolution(input, w, **kwargs)
    else:
        weight_groups = tf.split(w, num_or_size_splits=group, axis=-1)
        xs = tf.split(input, num_or_size_splits=group, axis=-1)
        convolved = [tf.nn.convolution(x, weight, **kwargs) for
                    (x, weight) in zip(xs, weight_groups)]
        layer = tf.concat(convolved, axis=-1)

    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable=is_train, name=name + "_bias")
        layer = layer + b
    return layer""")


    def _layer_PRelu(self):
        self.add_body(0, """
def prelu(input, name):
    gamma = tf.Variable(__weights_dict[name]['gamma'], name=name + "_gamma", trainable=is_train)
    return tf.maximum(0.0, input) + gamma * tf.minimum(0.0, input)
    """)


    def _layer_BatchNorm(self):
        self.add_body(0, """
def batch_normalization(input, name, **kwargs):
    mean = tf.Variable(__weights_dict[name]['mean'], name = name + "_mean", trainable = is_train)
    variance = tf.Variable(__weights_dict[name]['var'], name = name + "_var", trainable = is_train)
    offset = tf.Variable(__weights_dict[name]['bias'], name = name + "_bias", trainable = is_train) if 'bias' in __weights_dict[name] else None
    scale = tf.Variable(__weights_dict[name]['scale'], name = name + "_scale", trainable = is_train) if 'scale' in __weights_dict[name] else None
    return tf.nn.batch_normalization(input, mean, variance, offset, scale, name = name, **kwargs)
""")


    def _layer_SeparableConv(self):
        self.add_body(0, """
def separable_convolution(input, name, **kwargs):
    depthwise = tf.Variable(__weights_dict[name]['depthwise_filter'], trainable = is_train, name = name + "_df")
    pointwise = tf.Variable(__weights_dict[name]['pointwise_filter'], trainable = is_train, name = name + "_pf")
    layer = tf.nn.separable_conv2d(input, depthwise, pointwise, **kwargs)
    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable = is_train, name = name + "_bias")
        layer = layer + b
    return layer""")


    def _layer_DepthwiseConv(self):
        self.add_body(0, """
def depthwise_convolution(input, name, **kwargs):
    depthwise = tf.Variable(__weights_dict[name]['weights'], trainable = is_train, name = name + "_df")
    layer = tf.nn.depthwise_conv2d(input, depthwise, **kwargs)
    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable = is_train, name = name + "_bias")
        layer = layer + b
    return layer""")


    def _layer_ConvTranspose(self):
        self.add_body(0, """
def convolution_transpose(input, name, **kwargs):
    w = tf.Variable(__weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    dim = __weights_dict[name]['weights'].ndim - 2
    if dim == 2:
        layer = tf.nn.conv2d_transpose(input, w, **kwargs)
    elif dim == 3:
        layer = tf.nn.conv3d_transpose(input, w, **kwargs)
    else:
        raise ValueError("Error dim number {} in ConvTranspose".format(dim))

    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable=is_train, name=name + "_bias")
        layer = layer + b
    return layer""")
