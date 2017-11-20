#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import os

import numpy as np
from six import string_types as _string_types
from mmdnn.conversion.common.IR.IR_graph import IRGraph, IRGraphNode
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.DataStructure.emitter import Emitter

class PytorchEmitter(Emitter):
   
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

    # Base Functions
    def __init__(self, model):        
        super(PytorchEmitter, self).__init__()
        if isinstance(model, _string_types):
            network_path = model
        else:
            network_path = model[0]
            weight_path = model[1]

        self.init_codes = str()
        self.IR_graph = IRGraph(network_path)
        self.IR_graph.build()
        self._load_weights(weight_path)


    def run(self, dstNetworkPath, dstWeightPath = None, phase = 'test'):
        super(PytorchEmitter, self).run(dstNetworkPath, dstWeightPath, phase)
        if self.weight_loaded:
            self.save_weights(self.weights_dict, dstWeightPath)


    def add_init(self, indent, codes):
        if isinstance(codes, _string_types):
            codes = [codes]
        for code in codes:
            self.init_codes += ("    " * indent) + code + '\n'


    @property
    def header_code(self):
        return """import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return
    
    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):    
"""

    def gen_codes(self, phase):
        self.add_init(1, """
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)
""")
        
        self.add_body(1, "def forward(self, x):")

        for layer in self.IR_graph.topological_sort:            
            current_node = self.IR_graph.get_node(layer)
            node_type = current_node.type

            if hasattr(self, "emit_" + node_type):
                func = getattr(self, "emit_" + node_type)
                line = func(current_node)

            else:
                print("Pytorch Emitter has not supported operator [%s]." % (node_type))
                self.emit_UNKNOWN(current_node)

        self.add_body(2, "return {}".format(
            ','.join([self.IR_graph.get_node(name).real_variable_name for name in self.IR_graph.output_layers])))

        self.add_body(0, "")
        for i in self.used_layers:
            func = getattr(self, "_layer_" + i)
            func()
        
        return self.header_code + '\n' + self.init_codes + '\n' + self.body_codes

        
    def emit_Convolution(self, IR_node):
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
        self.used_layers.add(IR_node.type)

        dim = len(IR_node.IR_layer.attr["strides"].list.i) - 2

        in_channels = IR_node.IR_layer.attr["filter"].list.i[-2]
        
        filter = IR_node.IR_layer.attr["filter"].list.i[-1]
        
        kernel = IR_node.IR_layer.attr["filter"].list.i[:-2]

        strides = IR_node.IR_layer.attr["strides"].list.i[1:-1]        

        use_bias = IR_node.IR_layer.attr["use_bias"].b

        if IR_node.IR_layer.attr["padding"].s == b'VALID':
            padding = 0
        else:
            calculate_same_pad
                
        self.add_init(2, "self.{} = self.__convolution({}, name = '{}', in_channels = {}, out_channels = {}, kernel_size = ({}), stride = ({}), padding = {}, bias = {})".format(
            IR_node.variable_name,
            dim,
            IR_node.name, 
            in_channels, 
            filter, 
            ','.join('%s' % id for id in kernel),
            ','.join('%s' % id for id in strides),
            padding,
            use_bias))
                        
        self.add_body(2, "{:<15} = self.{}({})".format(
            IR_node.variable_name,
            IR_node.variable_name,
            self.IR_graph.get_node(IR_node.in_edges[0]).real_variable_name))

        if self.weight_loaded:
            self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], [dim + 1, dim] + list(range(0, dim)))

    
    def emit_Pool(self, IR_node):
        dim = len(IR_node.IR_layer.attr["strides"].list.i) - 2

        if IR_node.layer.attr['pooling_type'].s == b"MAX":
            pool_name = "max_pool{}d".format(dim)
        elif IR_node.layer.attr['pooling_type'].s == b"AVG":
            pool_name = "avg_pool{}d".format(dim)
        else:
            assert False
        
        if IR_node.layer.attr['global_pooling'].b:
            raise NotImplementedError("Not Global Pooling support!")
            
        else:
            for e in IR_node.IR_layer.attr["dilation_rate"].list.i:
                assert e == 1
            
            if IR_node.IR_layer.attr["padding"].s == b'VALID':
                padding = 0
            else:
                # Kit TODO: to handle padding
                padding = 1

            pool_size = IR_node.IR_layer.attr['window_shape'].list.i[1:-1]            
            strides = IR_node.IR_layer.attr['strides'].list.i[1:-1]                            
            
            self.add_body(2, "{:<15} = F.{}(input = {}, kernel_size = ({}), stride = ({}), padding = {})".format(
                IR_node.variable_name,
                pool_name,
                self.IR_graph.get_parent(IR_node.name, [0]).real_variable_name,
                ','.join([str(id) for id in pool_size]),
                ','.join([str(id) for id in strides]),
                padding))


    def emit_UNKNOWN(self, IR_node):
        print(IR_node.name)


    def emit_DataInput(self, IR_node):
        # Ignore it in Pytorch
        IR_node.real_name = 'x'        


    def emit_Dropout(self, IR_node):
        self.add_body(2, "{:<15} = F.dropout(input = {}, p = {}, training = self.training, inplace = True)".format(
            IR_node.variable_name,
            self.IR_graph.get_parent(IR_node.name, [0]).real_variable_name,
            IR_node.layer.attr["keep_prob"].f))

    def check_if_need_transpose(self, IR_node):
        parent = self.IR_graph.get_parent(IR_node.name, [0])
        while parent.type == 'Flatten':
            parent = self.IR_graph.get_parent(parent.name, [0])
        dim = len(parent.layer.attr['_output_shapes'].list.shape[0].dim)
        if dim > 2:
            original_dims = self.weights_dict[IR_node.name]['weights'].shape
            dims = [i.size for i in parent.layer.attr['_output_shapes'].list.shape[0].dim[1:]] + [-1]
            self.weights_dict[IR_node.name]['weights'] = np.reshape(self.weights_dict[IR_node.name]['weights'], dims)
            self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], [dim - 2] + list(range(0, dim - 2)) + [dim - 1])
            self.weights_dict[IR_node.name]['weights'] = np.reshape(self.weights_dict[IR_node.name]['weights'], original_dims)


    def emit_FullyConnected(self, IR_node):
        self.used_layers.add(IR_node.type)
        in_features = 1
        for i in self.IR_graph.get_parent(IR_node.name, [0]).layer.attr['_output_shapes'].list.shape[0].dim[1:]:
            in_features *= i.size
        
        self.add_init(2, "self.{} = self.__dense(name = '{}', in_features = {}, out_features = {}, bias = {})".format(
            IR_node.variable_name,
            IR_node.name,
            in_features,
            IR_node.layer.attr["units"].i,
            IR_node.IR_layer.attr["use_bias"].b))

        self.add_body(2, "{:<15} = self.{}({})".format(
            IR_node.variable_name,
            IR_node.variable_name,            
            self.IR_graph.get_parent(IR_node.name, [0]).real_variable_name))

        if self.weight_loaded:
            self.check_if_need_transpose(IR_node)
            self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], (1, 0))


    def emit_Flatten(self, IR_node):
        parent = self.IR_graph.get_parent(IR_node.name, [0]).real_variable_name
        self.add_body(2, "{:<15} = {}.view({}.size(0), -1)".format(
            IR_node.variable_name,
            parent,
            parent))


    def emit_Reshape(self, IR_node):
        shape_str = IRGraph.shapeToStr(IR_node.IR_layer.attr["shape"].shape, True)                
        self.add_body(1, "{:<15} = Reshape(name = \"{}\", target_shape = ({}))({})".format(
            IR_node.variable_name,
            IR_node.name, 
            shape_str, 
            self.IR_graph.get_node(IR_node.in_edges[0]).real_variable_name))


    def emit_Tanh(self, IR_node):
        code = "{:<15} = Activation(name = '{}', activation = 'tanh')({})".format(
                IR_node.replace_scope(IR_node.name),
                IR_node.name,
                IR_node.replace_scope(IR_node.in_edges[0]))
        return code


    def emit_Relu(self, IR_node):
        self.add_body(2, "{:<15} = F.relu({})".format(        
            IR_node.variable_name,            
            self.IR_graph.get_parent(IR_node.name, [0]).real_variable_name))
      

    def emit_Softmax(self, IR_node):
        self.add_body(2, "{:<15} = F.softmax({})".format(        
            IR_node.variable_name,            
            self.IR_graph.get_parent(IR_node.name, [0]).real_variable_name))


    def emit_Sigmoid(self, IR_node):
        code = "{:<15} = Activation(name = '{}', activation = 'sigmoid')({})".format(
                IR_node.replace_scope(IR_node.name), 
                IR_node.name,
                IR_node.replace_scope(IR_node.in_edges[0]))
        return code


    def emit_Embedding(self, IR_node):
        ret = "{:<15} = Embedding(input_dim = {}, output_dim = {}, mask_zero = {})({})".format(
                IR_node.name, 
                IR_node.IR_layer.attr['input_dim'].i,
                IR_node.IR_layer.attr['output_dim'].i,
                IR_node.IR_layer.attr['mask_zero'].b,
                IR_node.in_edges[0])

        return ret


    def emit_RNNs(self, IR_node, func):
        # for Keras
        if "dropout" in IR_node.IR_layer.attr:
            dropout_str = ",dropout = {}, recurrent_dropout = {}".format(
                    IR_node.IR_layer.attr['dropout'].f,
                    IR_node.IR_layer.attr['recurrent_dropout'].f)
        else:
            dropout_str = ""
        
        code = "{:<15} = {}(units = {}, use_bias = {} {})({})".format(
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


    def emit_Add(self, IR_node):
        code = Keras2Emitter._emit_merge(IR_node, "add")
        return code


    def emit_Concat(self, IR_node):
        code = Keras2Emitter._emit_merge(IR_node, "concatenate")
        return code


    def emit_BatchNorm(self, IR_node):
        code = "{:<15} = BatchNormalization(name = '{}', axis = {}, center = {}, scale = {})({})".format(
                IR_node.variable_name,
                IR_node.name,                
                IR_node.IR_layer.attr['axis'].i,
                IR_node.IR_layer.attr['center'].b,
                IR_node.IR_layer.attr['scale' ].b,
                IR_node.replace_scope(IR_node.in_edges[0]))
        return code


    def emit_pad(self, IR_node):
        if IR_node.IR_layer.attr['mode'].s == b"CONSTANT":
            func = "ZeroPadding"

        dim = len(IR_node.IR_layer.attr['padding'].list.i) // 2

        padding_str = ""
        for idx in range(0, dim):
            padding_str += "({}, {}),".format(
                    IR_node.IR_layer.attr['padding'].list.i[idx + idx],
                    IR_node.IR_layer.attr['padding'].list.i[idx + idx + 1])

        code = "{:<15} = {}{}D(name = \"{}\", padding = ({}))({})".format(
                IR_node.variable_name,
                func,
                dim,
                IR_node.name,
                padding_str,
                IR_node.replace_scope(IR_node.in_edges[0]))

        return code


    def emit_Squeeze(self, IR_node):
        raise NotImplementedError()
        input_name = IR_node.replace_scope(self.IR_graph.layer_name_map[IR_node.in_edges[0]])
        self.forward_code += "        {} = {}.view({}.size(0), -1)\n".format(
            IR_node.replace_scope(IR_node.name),
            input_name, 
            input_name)


    def emit_Pad(self, IR_node):        
        if IR_node.layer.attr['mode'].s == b'CONSTANT':
            mode = "mode = 'constant', value = {}".format(0)
        elif IR_node.layer.attr['mode'].s == b'REFLECT':
            mode = "mode = 'reflect'"
        elif IR_node.layer.attr['mode'].s == b'SYMMETRIC':
            mode = "mode = 'replicate'"
        else:
            assert False

        padding_str = ', '.join('%s' % i for i in IR_node.layer.attr['paddings'].list.i[2:-2])

        self.add_body(2, "{:<15} = F.pad({}, ({}), {})".format(
            IR_node.variable_name,
            self.IR_graph.get_node(IR_node.in_edges[0]).real_variable_name,
            padding_str,
            mode))


    def _layer_Convolution(self):
        self.add_body(0, """
    @staticmethod
    def __convolution(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()
                
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer""")


    def _layer_FullyConnected(self):
        self.add_body(0, """
    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer""")