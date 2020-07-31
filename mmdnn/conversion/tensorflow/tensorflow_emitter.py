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
from mmdnn.conversion.rewriter.folder import Folder


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

_weights_dict = dict()

is_train = {}

def load_weights(weight_file):
    import numpy as np

    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file = None):
    global _weights_dict
    _weights_dict = load_weights(weight_file)
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
        
        folder = Folder(self.IR_graph, self.weights_dict)
        folder.fold()

    def gen_code(self, phase):
        self.trainable = (phase == 'train')
        self.add_body(0, self.header_code)

        for layer in self.IR_graph.topological_sort:
            current_node = self.IR_graph.get_node(layer)
            node_type = current_node.type

            if hasattr(self, "emit_" + node_type):
                func = getattr(self, "emit_" + node_type)
                line = func(current_node)
                if line != None:
                    self.add_body(1, line)
            else:
                print("TensorflowEmitter has not supported operator [%s]." % (node_type))
                self.emit_UNKNOWN(current_node)


        self.add_body(1, "return {}, {}".format(
            ', '.join([self.IR_graph.get_node(name).real_variable_name for name in self.IR_graph.input_layers if self.IR_graph.get_node(name).type != 'Const' and not self.IR_graph.get_node(name).get_attr('feed_weights')]),
            ', '.join([self.IR_graph.get_node(name).real_variable_name for name in self.IR_graph.output_layers if self.IR_graph.get_node(name).type != 'Pack' and  self.IR_graph.get_node(name).type !='Shape'])))



        self.add_body(0, "")
        for i in self.used_layers:
            func = getattr(self, "_layer_" + i)
            func()

        self.add_body(0, "")
        for code in self.layers_codes.values():
            self.add_body(0, code)

        return self.body_code


    def parent_variable_name(self, IR_node, path=[0]):
        if not IR_node.in_edges and IR_node.name in self.weights_dict.keys():
            return "tf.constant(_weights_dict['{}']['weights'], name='{}')".format(
                IR_node.name,
                IR_node.name)
        return super(TensorflowEmitter, self).parent_variable_name(IR_node, path)


    @staticmethod
    def _shapeToStr(shapes):
        ret = [dim.size if dim.size != -1 else 'None' for dim in shapes.dim]
        return ', '.join('%s' % i for i in ret)


    def emit_Conv(self, IR_node):
        self.used_layers.add(IR_node.type)
        strides_str = ', '.join('%s' % i for i in IR_node.get_attr('strides')[1:-1])
        input_node, padding = self._defuse_padding(IR_node)
        data_format = IR_node.get_attr('data_format')
        code = "{:<15} = convolution({}, group={}, strides=[{}], padding='{}', name='{}')".format(
            IR_node.variable_name,
            input_node,
            IR_node.get_attr('group', 1),
            strides_str,
            padding,
            IR_node.name)
        return code

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
        if 'dtype' in IR_node.layer.attr:
            dtype_str = "{}".format(self.dtype_map[IR_node.layer.attr['dtype'].type])
        else:
            dtype_str = "tf.float32"
        code = "{:<15} = tf.constant({}, dtype={}, name='{}')".format(
            IR_node.variable_name,
            "_weights_dict['{}']['value']".format(IR_node.name) if IR_node.get_attr('value')== None else IR_node.get_attr('value'),
            dtype_str,
            IR_node.name)

        return code


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
            code = "{:<15} = tf.nn.{}{}({}, [1] + {}.get_shape().as_list()[1:-1] + [1], strides = [1] * {}, padding = 'VALID', name = '{}')".format(
                IR_node.variable_name,
                op,
                dim_str,
                self.parent_variable_name(IR_node),
                self.parent_variable_name(IR_node),
                arrlen,
                IR_node.name)
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

                code = "{:<15} = tf.nn.{}{}({}, [{}], [{}], padding='{}', name='{}')".format(
                    IR_node.variable_name,
                    op,
                    dim_str,
                    self.parent_variable_name(IR_node),
                    kernel_shape_str,
                    strides_str,
                    'SAME',
                    IR_node.name)
            else:
                kernel_shape_str = ', '.join('%s' % i for i in IR_node.get_attr('kernel_shape'))
                strides_str = ', '.join('%s' % i for i in IR_node.get_attr('strides'))
                input_node, padding = self._defuse_padding(IR_node, padding_const)
                code = "{:<15} = tf.nn.{}{}({}, [{}], [{}], padding='{}', name='{}')".format(
                    IR_node.variable_name,
                    op,
                    dim_str,
                    input_node,
                    kernel_shape_str,
                    strides_str,
                    padding,
                    IR_node.name)

        return code

    def emit_UNKNOWN(self, IR_node):
        print(IR_node.name)

    def emit_Add(self, IR_node):
        code = "{:<15} = {}".format(
            IR_node.variable_name,
            ' + '.join('%s' % self.parent_variable_name(IR_node, [idx]) for idx in range(len(IR_node.in_edges))))

        return code

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
        return code

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
            kernel_str = "kernel_initializer = tf.constant_initializer(_weights_dict['{}']['weights']), ".format(IR_node.name)
        else: kernel_str = ""

        if IR_node.name in self.weights_dict and 'bias' in self.weights_dict[IR_node.name]:
            bias_str = "bias_initializer = tf.constant_initializer(_weights_dict['{}']['bias']), ".format(IR_node.name)
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
            return code

        else:
            code = "{:<15} = tf.layers.dense({}, {}, {}{}use_bias = {})".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                IR_node.layer.attr['units'].i,
                kernel_str,
                bias_str,
                IR_node.layer.attr['use_bias'].b)
            return code


    def emit_UpSampling2D(self, IR_node):
        scales = IR_node.get_attr('scales')
        scales = tuple(scales)
        interpolation_type = IR_node.get_attr('interpolation_type')
        if interpolation_type:
            assert interpolation_type in ["nearest", "bilinear"]
        else:
            interpolation_type = "nearest"
        code = "{:<15} = tf.keras.layers.UpSampling2D(size={}, interpolation='{}')({})".format(
            IR_node.variable_name,
            scales,
            interpolation_type,
            self.parent_variable_name(IR_node))
        return code


    def emit_Flatten(self, IR_node):
        #self._emit_unary_operation(IR_node, "contrib.layers.flatten")
        code = "{:<15} = tf.contrib.layers.flatten({})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node))
        return code


    def emit_Mul(self, IR_node):

        code = "{:<15} = {}".format(
            IR_node.variable_name,
            ' * '.join('%s' % self.parent_variable_name(IR_node, [idx]) for idx in range(len(IR_node.in_edges))))
        return code


    def emit_Const(self, IR_node):
        if 'dtype' in IR_node.layer.attr:
            dtype_str = "dtype={}".format(self.dtype_map[IR_node.layer.attr['dtype'].type])
            if 'int' in dtype_str:
                code = "{:<15} = tf.constant({}, {}, shape=(1,))".format(
                    IR_node.variable_name,
                    IR_node.layer.attr['value'].i,
                    dtype_str)
            else:
                code = "{:<15} = tf.constant({}, {}, shape=(1,))".format(
                    IR_node.variable_name,
                    IR_node.layer.attr['value'].f,
                    dtype_str)
        else:
            dtype_str = "dtype=tf.float32"
            code ="{:<15} = tf.constant({}, {}, shape=(1,))".format(
                IR_node.variable_name,
                IR_node.layer.attr['value'].f,
                dtype_str)

        return code

    def emit_Transpose(self, IR_node):
        code ="{:<15} = tf.transpose(a = {}, perm = {})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node, [0]),
            self.parent_variable_name(IR_node, [1]))
        
        return code

    def emit_Gather(self, IR_node):
        variable_str = "tf.convert_to_tensor(_weights_dict['{}']['weights'])".format(IR_node.name)

        code = "{:<15} = tf.gather(params = {}, indices = {}, axis = {})".format(
            IR_node.variable_name,
            variable_str,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('axis')
            )
        
        return code

    def emit_Unstack(self, IR_node):
        code = "{:<15} = tf.unstack(value={}, num={}, axis={})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('num'),
            IR_node.get_attr('axis')
        )
        return code

    def emit_Reshape(self, IR_node):
        code = "{:<15} = tf.reshape({}, [{}], '{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            ', '.join('%s' % i for i in IR_node.get_attr('shape')),
            IR_node.name)
        
        return code


    def emit_Sub(self, IR_node):
        code = "{:<15} = {}".format(
            IR_node.variable_name,
            ' - '.join('%s' % self.parent_variable_name(IR_node, [idx]) for idx in range(len(IR_node.in_edges))))
        
        return code

    def emit_Div(self, IR_node):
        code = "{:<15} = tf.div({}, {}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            self.parent_variable_name(IR_node, [1]),
            IR_node.name
        )
        return code

    def _emit_unary_operation(self, IR_node, op_name):
        code = "{:<15} = tf.{}({}, name = '{}')".format(
            IR_node.variable_name,
            op_name,
            self.parent_variable_name(IR_node),
            IR_node.name)
        return code

    def emit_Tanh(self, IR_node):
        code = self._emit_unary_operation(IR_node, 'tanh')
        return code

    def emit_Elu(self, IR_node):
        return self._emit_unary_operation(IR_node, 'nn.elu')


    def emit_Relu(self, IR_node):
        return self._emit_unary_operation(IR_node, 'nn.relu')


    def emit_Relu6(self, IR_node):
        return self._emit_unary_operation(IR_node, 'nn.relu6')


    def emit_CRelu(self, IR_node):
        return self._emit_unary_operation(IR_node, 'nn.crelu')


    def emit_PRelu(self, IR_node):
        self.used_layers.add(IR_node.type)
        code = "{:<15} = prelu({}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.name)
        return code

    def emit_LeakyRelu(self, IR_node):
        self.add_body(1, "{:<15} = tf.nn.leaky_relu({}, alpha={}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('alpha'),
            IR_node.name
        ))


    def emit_Softmax(self, IR_node):
        return self._emit_unary_operation(IR_node, 'nn.softmax')


    def emit_Sigmoid(self, IR_node):
        code = self._emit_unary_operation(IR_node, 'sigmoid')
        return code

    def emit_Embedding(self, IR_node):
        variable_str = "tf.convert_to_tensor(_weights_dict['{}']['weights'])".format(IR_node.name)
        code = "{:<15} = tf.nn.embedding_lookup(params = {}, ids = {})".format(
            IR_node.variable_name,
            variable_str,
            self.parent_variable_name(IR_node))
        return code

    def emit_LSTM(self, IR_node):
        return self.emit_RNNs(IR_node, "LSTM")


    def emit_GRU(self, IR_node):
        return self.emit_RNNs(IR_node, "GRU")


    def emit_Concat(self, IR_node):
        
        code = "{:<15} = tf.concat([{}], {}, name = '{}')".format(
            IR_node.variable_name,
            ', '.join(self.parent_variable_name(IR_node, [idx]) for idx in range(len(IR_node.in_edges))),
            IR_node.layer.attr['axis'].i,
            IR_node.name)

        return code

    def emit_BatchNorm(self, IR_node):
        self.used_layers.add(IR_node.type)
        code = "{:<15} = batch_normalization({}, variance_epsilon={}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('epsilon'),
            IR_node.name)
        return code

    def emit_Scale(self, IR_node):
        self.used_layers.add(IR_node.type)
        code = "{:<15} = scale({}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.name)
        return code

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
        code = "{:<15} = tf.pad({}, {}, '{}', name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            padding,
            mode,
            IR_node.variable_name)
        return code

    def emit_Squeeze(self, IR_node):
        code = "{:<15} = tf.squeeze({}, [{}], name = '{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            ', '.join('%s' % axis for axis in IR_node.layer.attr['axes'].list.i),
            IR_node.name)
        return code


    def emit_ReduceMean(self, IR_node):
        code = "{:<15} = tf.reduce_mean({}, [{}], {}, name = '{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            ','.join('%s' % i for i in IR_node.get_attr('axes')),
            IR_node.get_attr('keepdims'),
            IR_node.name)
        return code

    def emit_LRN(self, IR_node):
        input_name = IR_node.variable_name
        output_name = self.parent_variable_name(IR_node)
        IR_name = IR_node.name
        size = IR_node.get_attr('size')
        depth_radius = int(IR_node.get_attr('size') / 2)
        bias = IR_node.get_attr('bias', 1)
        alpha = IR_node.get_attr('alpha') / size
        beta = IR_node.get_attr('beta')

        code = "{:<15} = tf.nn.lrn({}, depth_radius={}, bias={}, alpha={}, beta={}, name='{}')".format(
            input_name,
            output_name,
            depth_radius,
            bias,
            alpha,
            beta,
            IR_name)
        return code

    def emit_SeparableConv(self, IR_node):
        self.used_layers.add(IR_node.type)
        strides_str = ', '.join('%s' % i for i in IR_node.get_attr('strides'))
        input_node, padding = self._defuse_padding(IR_node)
        code = "{:<15} = separable_convolution({}, strides = [{}], padding = '{}', name = '{}')".format(
            IR_node.variable_name,
            input_node,
            strides_str,
            padding,
            IR_node.name)
        return code


    def emit_DepthwiseConv(self, IR_node):
        self.used_layers.add(IR_node.type)
        strides_str = ', '.join('%s' % i for i in IR_node.layer.attr['strides'].list.i)
        input_node, padding = self._defuse_padding(IR_node)
        code = "{:<15} = depthwise_convolution({}, strides = [{}], padding = '{}', name = '{}')".format(
            IR_node.variable_name,
            input_node,
            strides_str,
            padding,
            IR_node.name)
        return code

    def emit_Crop(self, IR_node):
        border = IR_node.get_attr('border')
        assert len(border) == 4

        output_shape = IR_node.get_attr('_output_shapes')[0]
        output_shape = shape_to_list(output_shape)

        code = "{:<15} = tf.image.crop_to_bounding_box({}, offset_height={}, offset_width={}, target_height={}, target_width={})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            border[0],
            border[1],
            output_shape[1],
            output_shape[2])
        
        return code

    def emit_ConvTranspose(self, IR_node):
        self.used_layers.add(IR_node.type)
        output_shape = [1] + shape_to_list(IR_node.get_attr('_output_shapes')[0])[1:]
        input_node, padding = self._defuse_padding(IR_node)
        code = "{:<15} = convolution_transpose({}, output_shape={}, strides={}, padding='{}', name='{}')".format(
            IR_node.variable_name,
            input_node,
            output_shape,
            IR_node.get_attr('strides'),
            padding,
            IR_node.name)
        return code

    def emit_Slice(self, IR_node):
        extra_str = ""
        if IR_node.get_attr('begin_mask'):
            extra_str += ", begin_mask={}".format(IR_node.get_attr('begin_mask'))
        if IR_node.get_attr('end_mask') != None:
            extra_str += ", end_mask={}".format(IR_node.get_attr('end_mask'))
        if IR_node.get_attr('shrink_axis_mask') != None:
            extra_str += ", shrink_axis_mask={}".format(IR_node.get_attr('shrink_axis_mask'))
        if IR_node.get_attr('new_axis_mask')!= None:
            extra_str += ", new_axis_mask={}".format(IR_node.get_attr('new_axis_mask'))

        if IR_node.get_attr('starts') != None:
            starts = IR_node.get_attr('starts')
        else:
            starts = self.parent_variable_name(IR_node, [1])
        
        if IR_node.get_attr('ends') != None:
            ends = IR_node.get_attr('ends')
        else:
            ends = self.parent_variable_name(IR_node, [2])
        
        if IR_node.get_attr('strides') != None:
            strides = IR_node.get_attr('strides')
        else:
            strides = self.parent_variable_name(IR_node, [3])

        code = "{:<15} = tf.strided_slice({}, {}, {}, {} {}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            starts,
            ends,
            strides,
            extra_str,
            IR_node.name)
        
        return code


    def emit_Shape(self, IR_node):
        code = "{:<15} = tf.shape({}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.name)
        return code

    def emit_Pack(self, IR_node):
        code = "{:<15} = tf.stack({}, axis={}, name='{}')".format(
            IR_node.variable_name,
            '[' +  ','.join('%s' % self.parent_variable_name(IR_node, [idx]) for idx in range(len(IR_node.in_edges))) + ']',
            IR_node.get_attr('axis'),
            IR_node.name)
        return code

    def emit_Split(self, IR_node):
        code = "{:<15} = tf.split({}, {}, {}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('split'),
            IR_node.get_attr('axis'),
            IR_node.name)
        return code

    def emit_Unsqueeze(self, IR_node):
        code = "{:<15} = tf.expand_dims({}, axis={}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('axes')[0],
            IR_node.name)
        return code

    def emit_Fill(self, IR_node):
        code = "{:<15} = tf.fill({}, {}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('value'),
            IR_node.name)
        return code

    def emit_Maximum(self, IR_node):
        code = "{:<15} = tf.maximum({}, {}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            self.parent_variable_name(IR_node, [1]),
            IR_node.name
        )
        return code

    def emit_Minimum(self, IR_node):
        code = "{:<15} = tf.minimum({}, {}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            self.parent_variable_name(IR_node, [1]),
            IR_node.name
        )
        return code

    def emit_Scope(self, IR_node):
        input_vars = [self.parent_variable_name(IR_node, [idx]) for idx in range(len(IR_node.in_edges))]
        input_vars.append('_weights_dict')
        code = "{:<15} = _{}({})".format(
            IR_node.real_variable_name,
            IR_node.pattern,
            ', '.join(input_vars))
        self._gen_scope_code(IR_node)
        return code


    def _gen_scope_code(self, scope_node):

        def _scope_func(scope_name, params, code, return_var):
            code = """
def _{}({}):
{}
    return {}
    """.format(scope_name, params, code, ', '.join(return_var))
            return code

        if not self.layers_codes.get(scope_node.pattern, None):
            body_code = str()
            for node_name in scope_node.topology_list:
                node = self.IR_graph.get_node(node_name)
                node_type = node.type

                if hasattr(self, "emit_" + node_type):
                    func = getattr(self, "emit_" + node_type)
                    line = func(node)
                    if line != None:
                        body_code += "    " + line + '\n'
                else:
                    print("TensorflowEmitter has not supported operator [%s]." % (node_type))
                    self.emit_UNKNOWN(node)

            # param_code does not need parameter slice.
            input_params = scope_node.input_params
            input_params.append("_weights_dict")
            param_code = ', '.join(input_params)
            function_code = _scope_func(scope_node.pattern, param_code, body_code, scope_node.return_variables)

            self.layers_codes[scope_node.pattern] = function_code



    def _layer_Conv(self):
        self.add_body(0, """
def convolution(input, name, group, **kwargs):
    w = tf.Variable(_weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    if group == 1:
        layer = tf.nn.convolution(input, w, name=name, **kwargs)
    else:
        weight_groups = tf.split(w, num_or_size_splits=group, axis=-1)
        xs = tf.split(input, num_or_size_splits=group, axis=-1)
        convolved = [tf.nn.convolution(x, weight, name=name, **kwargs) for
                    (x, weight) in zip(xs, weight_groups)]
        layer = tf.concat(convolved, axis=-1)

    if 'bias' in _weights_dict[name]:
        b = tf.Variable(_weights_dict[name]['bias'], trainable=is_train, name=name + "_bias")
        layer = layer + b
    return layer""")


    def _layer_PRelu(self):
        self.add_body(0, """
def prelu(input, name):
    gamma = tf.Variable(_weights_dict[name]['gamma'], name=name + "_gamma", trainable=is_train)
    return tf.maximum(0.0, input) + gamma * tf.minimum(0.0, input)
    """)


    def _layer_BatchNorm(self):
        self.add_body(0, """
def batch_normalization(input, name, **kwargs):
    mean = tf.Variable(_weights_dict[name]['mean'], name = name + "_mean", trainable = is_train)
    variance = tf.Variable(_weights_dict[name]['var'], name = name + "_var", trainable = is_train)
    offset = tf.Variable(_weights_dict[name]['bias'], name = name + "_bias", trainable = is_train) if 'bias' in _weights_dict[name] else None
    scale = tf.Variable(_weights_dict[name]['scale'], name = name + "_scale", trainable = is_train) if 'scale' in _weights_dict[name] else None
    return tf.nn.batch_normalization(input, mean, variance, offset, scale, name = name, **kwargs)
""")


    def _layer_Scale(self):
        self.add_body(0, """
def scale(input, name, **kwargs):
    mean = tf.Variable(_weights_dict[name]['scale_mean'], name = name + "_mean", trainable = is_train)
    variance = tf.Variable(_weights_dict[name]['scale_var'], name = name + "_var", trainable = is_train)
    offset = tf.Variable(_weights_dict[name]['bias'], name = name + "_bias", trainable = is_train) if 'bias' in _weights_dict[name] else None
    scale = tf.Variable(_weights_dict[name]['scale'], name = name + "_scale", trainable = is_train) if 'scale' in _weights_dict[name] else None
    return tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon = 0, name = name)
""")


    def _layer_SeparableConv(self):
        self.add_body(0, """
def separable_convolution(input, name, **kwargs):
    depthwise = tf.Variable(_weights_dict[name]['depthwise_filter'], trainable = is_train, name = name + "_df")
    pointwise = tf.Variable(_weights_dict[name]['pointwise_filter'], trainable = is_train, name = name + "_pf")
    layer = tf.nn.separable_conv2d(input, depthwise, pointwise, **kwargs)
    if 'bias' in _weights_dict[name]:
        b = tf.Variable(_weights_dict[name]['bias'], trainable = is_train, name = name + "_bias")
        layer = layer + b
    return layer""")


    def _layer_DepthwiseConv(self):
        self.add_body(0, """
def depthwise_convolution(input, name, **kwargs):
    depthwise = tf.Variable(_weights_dict[name]['weights'], trainable = is_train, name = name + "_df")
    layer = tf.nn.depthwise_conv2d(input, depthwise, **kwargs)
    if 'bias' in _weights_dict[name]:
        b = tf.Variable(_weights_dict[name]['bias'], trainable = is_train, name = name + "_bias")
        layer = layer + b
    return layer""")


    def _layer_ConvTranspose(self):
        self.add_body(0, """
def convolution_transpose(input, name, **kwargs):
    w = tf.Variable(_weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    dim = _weights_dict[name]['weights'].ndim - 2
    if dim == 2:
        layer = tf.nn.conv2d_transpose(input, w, **kwargs)
    elif dim == 3:
        layer = tf.nn.conv3d_transpose(input, w, **kwargs)
    else:
        raise ValueError("Error dim number {} in ConvTranspose".format(dim))

    if 'bias' in _weights_dict[name]:
        b = tf.Variable(_weights_dict[name]['bias'], trainable=is_train, name=name + "_bias")
        layer = layer + b
    return layer""")
