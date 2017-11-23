#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import os

from mmdnn.conversion.common.IR.IR_graph import IRGraph, IRGraphNode
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.DataStructure.emitter import Emitter

class CntkEmitter(Emitter):

    dtype_map = {
        graph_pb2.DT_FLOAT16 : "np.float16",
        graph_pb2.DT_FLOAT32 : "np.float32",
        graph_pb2.DT_FLOAT64 : "np.float64",
        graph_pb2.DT_INT16 : "np.int16",
        graph_pb2.DT_INT32 : "np.int32",
        graph_pb2.DT_INT64 : "np.int64",
        graph_pb2.DT_UINT8 : "np.uint8",
        graph_pb2.DT_UINT16 : "np.uint16"
    }


    def __init__(self, model):
        from six import string_types as _string_types
        super(CntkEmitter, self).__init__()
        if isinstance(model, _string_types):
            network_path = model
        else:
            network_path = model[0]
            self._load_weights(model[1])
        
        self.IR_graph = IRGraph(network_path)
        super(CntkEmitter, self)._build()
    

    @property
    def header_code(self):
        return """import numpy as np
import cntk
from cntk import ops, layers
from cntk.contrib.crosstalkcaffe.unimodel.cntkinstance import BlockApiSetup

__weights_dict = dict()

def load_weights(weight_file):
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

"""


    def gen_code(self, phase = 'test'):
        self.phase = phase
        self.add_body(0, self.header_code)

        for layer in self.IR_graph.topological_sort:
            current_node = self.IR_graph.get_node(layer)
            node_type = current_node.type

            if hasattr(self, "emit_" + node_type):
                func = getattr(self, "emit_" + node_type)
                func(current_node)
            else:
                print("CntkEmitter has not supported operator [%s]." % (node_type))
                self.emit_UNKNOWN(current_node)            

        self.add_body(1, "return {}".format(
            ','.join([self.IR_graph.get_node(name).real_variable_name for name in self.IR_graph.output_layers])))

        self.add_body(0, "")
        for i in self.used_layers:
            func = getattr(self, "_layer_" + i)
            func()

        return self.body_codes


    @staticmethod
    def _shapeToStr(shapes):
        new_shape = filter(lambda x:x >- 1, [dim.size for dim in shapes.dim])
        return ', '.join('%s' % i for i in new_shape)


    def emit_Convolution(self, IR_node):        
        if self.weight_loaded:
            self.used_layers.add(IR_node.type)
            dim = len(IR_node.layer.attr['strides'].list.i) - 2
            padding = [False] + [IR_node.layer.attr['padding'].s == b'SAME'] * dim
            self.add_body(1, "{:<15} = convolution({}, strides = ({},), auto_padding = [{}], name = '{}')".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                ', '.join('%s' % i for i in IR_node.layer.attr['strides'].list.i[1:-1]),
                ', '.join('%s' % i for i in padding),
                IR_node.name))
        
        else:
            self.add_body(1, "{:<15} = Convolution(name = '{}', num_filters = {}, filter_shape = ({}), strides = ({},), pad = {}, bias = {})({})\n".format(
                IR_node.variable_name,
                IR_node.name,
                IR_node.layer.attr["filter"].list.i[-1],
                ', '.join('%s' % i for i in IR_node.layer.attr["kernel_size"].list.i[-2]),
                ', '.join('%s' % i for i in IR_node.layer.attr['strides'].list.i[1:-1]),
                IR_node.layer.attr['padding'].s == b'SAME',
                IR_node.layer.attr['use_bias'].b,
                self.parent_variable_name(IR_node)))


    def emit_Pool(self, IR_node):
        input_node = self.IR_graph.get_node(IR_node.in_edges[0]).real_variable_name
        if IR_node.layer.attr['global_pooling'].b:
            self.used_layers.add('GlobalPooling')
            self.add_body(1, "{:<15} = global_pooling({}, '{}', name = '{}')".format(
                IR_node.variable_name,
                input_node,
                IR_node.layer.attr['pooling_type'].s.decode('utf-8'),
                IR_node.name))
        else:
            for e in IR_node.IR_layer.attr["dilation_rate"].list.i:
                assert e == 1
            
            pool_size = ', '.join('%s' % id for id in IR_node.layer.attr['window_shape'].list.i[1:-1])
            strides = ', '.join('%s' % id for id in IR_node.layer.attr['strides'].list.i[1:-1])
            padding = IR_node.layer.attr['padding'].s == b'SAME'
            
            if self.weight_loaded:
                self.used_layers.add(IR_node.type)
                self.add_body(1, "{:<15} = pooling({}, '{}', filter_shape = ({}), strides = ({}), pad = {}, name = '{}')".format(
                    IR_node.variable_name,
                    input_node,
                    IR_node.layer.attr['pooling_type'].s.decode('utf-8'),
                    pool_size,
                    strides,
                    padding,
                    IR_node.name))

            else:
                raise NotImplementedError("")


    def emit_UNKNOWN(self, IR_node):
        print(IR_node.IR_layer.name)


    def emit_DataInput(self, IR_node):        
        shape_str = self._shapeToStr(IR_node.IR_layer.attr["shape"].shape)
        dtype_str = ", dtype = {}".format(self.dtype_map[IR_node.layer.attr['dtype'].type]) if 'dtype' in IR_node.layer.attr else ""
        self.add_body(1, "{:<15} = cntk.input_variable(({},) {}, name = '{}')\n".format(
            IR_node.variable_name,            
            shape_str,
            dtype_str,
            IR_node.name))


    def emit_Dropout(self, IR_node):
        parent = self.IR_graph.get_parent(IR_node.name, [0])
        if self.phase == 'train':            
            self.add_body(1, "{:<15} = Dropout({}, name = '{}')({})".format(
                IR_node.variable_name,
                1 - IR_node.IR_layer.attr["keep_prob"].f,
                IR_node.name,
                parent.real_variable_name))
        else:
            IR_node.real_name = parent.real_name


    def emit_FullyConnected(self, IR_node):                        
        input_node = self.parent_variable_name(IR_node)
        if self.weight_loaded:
            self.used_layers.add(IR_node.type)
            self.add_body(1, "{:<15} = dense({}, name = '{}')".format(
                IR_node.variable_name,
                input_node,
                IR_node.name))

        else:
            self.add_body(1, "{:<15} = Dense({}, bias = {}, name = '{}')({})".format(
                IR_node.variable_name,
                IR_node.layer.attr["units"].i,
                IR_node.layer.attr['use_bias'].b,
                IR_node.name,
                input_node))


    def emit_Flatten(self, IR_node):        
        self.add_body(1, "{:<15} = ops.reshape({}, (-1,), name = '{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.name))


    def emit_Reshape(self, IR_node):        
        self.add_body(1, "{:<15} = cntk.reshape({}, shape = ({},) name = '{}')".format(
            IR_node.variable_name,
            self.IR_graph.get_node(IR_node.in_edges[0]).real_variable_name,
            ', '.join('%s' % i for i in IR_node.layer.attr["shape"].list.i),
            IR_node.name))


    def _emit_activation(self, IR_node, op_name):
        self.add_body(1, "{:<15} = layers.Activation(activation = {}, name = '{}')({})".format(
            IR_node.variable_name,
            op_name,
            IR_node.name,
            self.parent_variable_name(IR_node)))

    
    def emit_Tanh(self, IR_node):
        self._emit_activation(IR_node, 'ops.tanh')


    def emit_Relu(self, IR_node):
        self._emit_activation(IR_node, 'ops.relu')


    def emit_Softmax(self, IR_node):
        self._emit_activation(IR_node, 'ops.softmax')


    def emit_Sigmoid(self, IR_node):
        self._emit_activation(IR_node, 'ops.sigmoid')
        

    def emit_RNNs(self, IR_node, func):
        assert False        


    def emit_LSTM(self, IR_node):
        return self.emit_RNNs(IR_node, "LSTM")


    def emit_GRU(self, IR_node):
        return self.emit_RNNs(IR_node, "GRU")


    def emit_Add(self, IR_node):
        if len(IR_node.in_edges) > 1:
            inputs = '+ '.join(self.IR_graph.get_node(i).real_variable_name for i in IR_node.in_edges)
            self.add_body(1, "{:<15} = {}".format(
                IR_node.variable_name,
                inputs))


    def emit_Concat(self, IR_node):
        inputs = ', '.join(self.IR_graph.get_node(i).real_variable_name for i in IR_node.in_edges)
        self.add_body(1, "{:<15} = cntk.splice({}, axis = {}, name = '{}')".format(
            IR_node.variable_name,
            inputs,
            IR_node.layer.attr['axis'].i - 1,
            IR_node.name))


    def emit_BatchNorm(self, IR_node):        
        self.used_layers.add(IR_node.type)
        self.add_body(1, "{:<15} = batch_normalization({}, epsilon = {}, name = '{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.layer.attr['epsilon'].f,
            IR_node.name))
        

    def emit_Pad(self, IR_node):        
        if IR_node.layer.attr['mode'].s == b'CONSTANT':
            mode = 'mode = ops.CONSTANT_PAD, constant_value = {}'.format(IR_node.layer.attr['constant_values'].f)
        elif IR_node.layer.attr['mode'].s == b'REFLECT':
            mode = 'mode = ops.REFLECT_PAD'
        elif IR_node.layer.attr['mode'].s == b'SYMMETRIC':
            mode = 'mode = ops.SYMMETRIC_PAD'
        else:
            assert False

        padding_str = ', '.join('(%s, %s)' % 
            (IR_node.layer.attr['paddings'].list.i[idx],
             IR_node.layer.attr['paddings'].list.i[idx + 1]) 
             for idx in range(2, len(IR_node.layer.attr['paddings'].list.i), 2))

        self.add_body(1, "{:<15} = ops.pad({}, pattern = [{}], {})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            padding_str,
            mode))


    def emit_Squeeze(self, IR_node):
        IR_node.real_name = self.IR_graph.get_node(IR_node.in_edges[0]).real_name


    def emit_ReduceMean(self, IR_node):
        self.add_body(1, "{:<15} = ops.reduce_mean({}, axis = ({}), name = '{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            ', '.join('%s' % (i - 1) for i in IR_node.layer.attr['axes'].list.i),
            IR_node.name))


    def emit_LRN(self, IR_node):
        self.used_layers.add(IR_node.type)
        self.add_body(1, "{:<15} = lrn({}, k = 1, n = {}, alpha = {}, beta = {}, name = '{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.layer.attr['size'].i,
            IR_node.layer.attr['alpha'].f,
            IR_node.layer.attr['beta'].f,
            IR_node.name))            


    def _layer_LRN(self):
        self.add_body(0, """
def lrn(input, **kwargs):
    dim = len(input.output.shape)
    input = cntk.transpose(input, [dim - 1] + list(range(0, dim - 1)))
    layer = BlockApiSetup.lrn(**kwargs)(input)
    layer = cntk.transpose(layer, list(range(1, dim)) + [0])
    return layer
""")


    def _layer_FullyConnected(self):
        self.add_body(0, """
def dense(input, name, **kwargs):
    w = __weights_dict[name]['weights']
    b = __weights_dict[name]['bias'] if 'bias' in __weights_dict[name] else None
    return BlockApiSetup.linear(output_shape = w.shape[1], input_shape = w.shape[0], scale_init = w, bias_init = b, name = name, **kwargs)(input)
""")


    def _layer_Convolution(self):
        self.add_body(0, """
def convolution(input, name, **kwargs):    
    dim = __weights_dict[name]['weights'].ndim
    
    weight = np.transpose(__weights_dict[name]['weights'], [dim - 1, dim - 2] + list(range(0, dim - 2)))
    w = cntk.Parameter(init = weight, name = name + '_weight')

    input = cntk.transpose(input, [dim - 2] + list(range(0, dim - 2)))

    layer = ops.convolution(w, input, **kwargs)
    if 'bias' in __weights_dict[name]:
        bias = np.reshape(__weights_dict[name]['bias'], [-1] + [1] * (dim - 2))
        b = cntk.Parameter(init = bias, name = name + '_bias')
        layer = layer + b
    layer = cntk.transpose(layer, list(range(1, dim - 1)) + [0])
    return layer
""")


    def _layer_Pool(self):
        self.add_body(0, """
def pooling(input, type, name, **kwargs):
    dim = len(input.output.shape)
    input = cntk.transpose(input, [dim - 1] + list(range(0, dim - 1)))
    layer = layers.MaxPooling(**kwargs)(input) if type == 'MAX' else layers.AveragePooling(**kwargs)(input)
    layer = cntk.transpose(layer, list(range(1, dim)) + [0])
    return layer
""")


    def _layer_GlobalPooling(self):
        self.add_body(0, """
def global_pooling(input, type, **kwargs):
    dim = len(input.output.shape)
    input = cntk.transpose(input, [dim - 1] + list(range(0, dim - 1)))
    layer = layers.GlobalMaxPooling(**kwargs)(input) if type == 'MAX' else layers.GlobalAveragePooling(**kwargs)(input)
    layer = cntk.transpose(layer, list(range(1, dim)) + [0])
    return layer
""")


    def _layer_BatchNorm(self):
        self.add_body(0, """
def batch_normalization(input, name, epsilon, **kwargs):
    mean = cntk.Parameter(init = __weights_dict[name]['mean'],
        name = name + "_mean")
    var = cntk.Parameter(init = __weights_dict[name]['var'],
        name = name + "_var")
    
    layer = (input - mean) / cntk.sqrt(var + epsilon)
    if 'scale' in __weights_dict[name]:
        scale = cntk.Parameter(init = __weights_dict[name]['scale'],
            name = name + "_scale")
        layer = scale * layer

    if 'bias' in __weights_dict[name]:
        bias = cntk.Parameter(init = __weights_dict[name]['bias'], 
            name = name + "_bias")
        layer = layer + bias
    
    return layer
""")
