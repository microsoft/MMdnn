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
from mmdnn.conversion.common.utils import *
from mmdnn.conversion.rewriter.folder import Folder

class PytorchEmitter(Emitter):

    dtype_map = {
        graph_pb2.DT_FLOAT16 : "torch.float16",
        graph_pb2.DT_FLOAT32 : "torch.float32",
        graph_pb2.DT_FLOAT64 : "torch.float64",
        graph_pb2.DT_INT16 : "torch.int16",
        graph_pb2.DT_INT32 : "torch.int32",
        graph_pb2.DT_INT64 : "torch.int64",
        graph_pb2.DT_UINT8 : "torch.uint8",
        graph_pb2.DT_UINT16 : "torch.uint16"
    }

    # Base Functions
    def __init__(self, model):
        super(PytorchEmitter, self).__init__()
        if isinstance(model, _string_types):
            network_path = model
        else:
            network_path = model[0]
            weight_path = model[1]

        self.init_code = str()
        self.IR_graph = IRGraph(network_path)
        self.IR_graph.build()
        self._load_weights(weight_path)

        folder = Folder(self.IR_graph, self.weights_dict)
        folder.fold()

    def run(self, dstNetworkPath, dstWeightPath = None, phase = 'test'):
        super(PytorchEmitter, self).run(dstNetworkPath, dstWeightPath, phase)
        if self.weight_loaded:
            self.save_weights(self.weights_dict, dstWeightPath)


    def add_init(self, indent, codes):
        if isinstance(codes, _string_types):
            codes = [codes]
        for code in codes:
            self.init_code += ("    " * indent) + code + '\n'

    def parent_variable_name(self, IR_node, path=[0], weight_type='weights'):
        if not IR_node.in_edges and IR_node.name in self.weights_dict.keys():
            self.weights_dict[IR_node.name][weight_type] = self.weights_dict[IR_node.name][weight_type]
            return "torch.from_numpy(_weights_dict['{}']['{}'])".format(IR_node.name, weight_type)

        return super(PytorchEmitter, self).parent_variable_name(IR_node, path)


    @property
    def header_code(self):
        return """import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

_weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):
"""

    def gen_code(self, phase):
        self.add_init(1, """
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global _weights_dict
        _weights_dict = load_weights(weight_file)
""")

        self.add_body(1, "def forward(self, x):")

        for layer in self.IR_graph.topological_sort:
            current_node = self.IR_graph.get_node(layer)
            node_type = current_node.type

            if hasattr(self, "emit_" + node_type):
                func = getattr(self, "emit_" + node_type)
                line = func(current_node)
                if line:
                    self.add_body(2, line)

            else:
                print("Pytorch Emitter has not supported operator [%s]." % (node_type))
                self.emit_UNKNOWN(current_node)

        self.add_body(2, "return {}".format(
            ', '.join([self.IR_graph.get_node(name).real_variable_name for name in self.IR_graph.output_layers if self.IR_graph.get_node(name).type != 'Pack'])))

        self.add_body(0, "")
        for i in self.used_layers:
            func = getattr(self, "_layer_" + i)
            func()

        self.add_body(0, "")
        for code in self.layers_codes.values():
            self.add_body(0, code)

        return self.header_code + '\n' + self.init_code + '\n' + self.body_code


    def _defuse_padding(self, IR_node, extra_str = ""):
        input_node = self.parent_variable_name(IR_node)
        if IR_node.get_attr('auto_pad') == 'VALID':
            return input_node

        if is_valid_padding(IR_node.get_attr("pads")) == True:
            return input_node

        padding = self._convert_padding(IR_node)
        input_node = IR_node.variable_name + '_pad'
        self.add_body(2, "{:<15} = F.pad({}, {}{})".format(
            input_node,
            self.parent_variable_name(IR_node),
            padding,
            extra_str
        ))

        return input_node


    def emit_Conv(self, IR_node):
        self.used_layers.add('Conv')

        dim = len(IR_node.get_attr('strides')) - 2

        in_channels = IR_node.get_attr('kernel_shape')[-2]
        filter = IR_node.get_attr('kernel_shape')[-1]
        kernel = IR_node.get_attr('kernel_shape')[:-2]
        strides = IR_node.get_attr('strides')[1:-1]

        if IR_node.type == 'DepthwiseConv':
            group = in_channels
            filter = group

        else:
            group = IR_node.get_attr('group', 1)

        self.add_init(2, "self.{} = self.__conv({}, name='{}', in_channels={}, out_channels={}, kernel_size={}, stride={}, groups={}, bias={})".format(
            IR_node.variable_name,
            dim,
            IR_node.name,
            in_channels,
            filter,
            tuple(kernel),
            tuple(strides),
            # padding,
            group,
            IR_node.get_attr('use_bias')))

        input_node = self._defuse_padding(IR_node)

        code = "{:<15} = self.{}({})".format(
            IR_node.variable_name,
            IR_node.variable_name,
            input_node)

        if self.weight_loaded:
            if IR_node.type == 'DepthwiseConv':
                self.weights_dict[IR_node.name]['weights'] = np.swapaxes(self.weights_dict[IR_node.name]['weights'], -1, -2)
            self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], [dim + 1, dim] + list(range(0, dim)))

        return code


    @staticmethod
    def is_ceil_mode(pads):
        lens = len(pads)
        for i in range(lens // 2 + 1, lens - 1):
            if pads[i] == pads[i - lens // 2]:
                return False
        else:
            return True


    def emit_Pool(self, IR_node):
        dim = len(IR_node.get_attr('strides')) - 2

        if IR_node.get_attr('pooling_type') == "MAX":
            pool_name = "max_pool{}d".format(dim)
            # exstr = ", value=float('-Inf')"
        elif IR_node.get_attr('pooling_type') == "AVG":
            pool_name = "avg_pool{}d".format(dim)
            # exstr = ""
        else:
            raise ValueError()

        if IR_node.layer.attr['global_pooling'].b:
            code = "{:<15} = F.{}(input = {}, kernel_size = {}.size()[2:])".format(
                IR_node.variable_name,
                pool_name,
                self.parent_variable_name(IR_node),
                self.parent_variable_name(IR_node)
            )
            return code

        else:
            if IR_node.get_attr('pooling_type') == "MAX":
                # Change to padding defuse
                input_node = self._defuse_padding(IR_node,", value=float('-inf')")
                for e in IR_node.get_attr('dilations', []):
                    assert e == 1

                pool_size = IR_node.get_attr('kernel_shape')[1:-1]
                strides = IR_node.get_attr('strides')[1:-1]

                code = "{}, {}_idx = F.{}({}, kernel_size={}, stride={}, padding={}, ceil_mode={}, return_indices={})".format(
                    IR_node.variable_name,
                    IR_node.variable_name,
                    pool_name,
                    input_node,
                    tuple(pool_size),
                    tuple(strides),
                    0,
                    False,
                    True
                    )
                return code

            elif IR_node.get_attr('pooling_type') == "AVG":

                for e in IR_node.get_attr('dilations', []):
                    assert e == 1

                pool_size = IR_node.get_attr('kernel_shape')[1:-1]
                strides = IR_node.get_attr('strides')[1:-1]

                padding = IR_node.get_attr('pads')[1:dim]
                ceil_mode = self.is_ceil_mode(IR_node.get_attr('pads'))

                # input_node = self._defuse_padding(IR_node, exstr)
                code = "{:<15} = F.{}({}, kernel_size={}, stride={}, padding={}, ceil_mode={}, count_include_pad=False)".format(
                    IR_node.variable_name,
                    pool_name,
                    self.parent_variable_name(IR_node),
                    tuple(pool_size),
                    tuple(strides),
                    tuple(padding),
                    ceil_mode
                    )
                return code
            else:
                raise ValueError()

    def emit_Unpool(self, IR_node):
        dim = len(IR_node.get_attr('strides')) - 2

        # Change to padding defuse
        input_node = self.parent_variable_name(IR_node)
        index_node = self.parent_variable_name(IR_node,[1])
        pool_name = "max_unpool{}d".format(dim)
        pool_size = IR_node.get_attr('kernel_shape')[1:-1]
        strides = IR_node.get_attr('strides')[1:-1]

        code = "{:<15} = F.{}({},{}_idx, kernel_size={}, stride={}, padding={})".format(
            IR_node.variable_name,
            pool_name,
            input_node,
            index_node,
            tuple(pool_size),
            tuple(strides),
            0
            )
        return code


    def emit_UNKNOWN(self, IR_node):
        print(IR_node.name)


    def emit_DataInput(self, IR_node):
        # Ignore it in Pytorch
        IR_node.real_name = 'x'


    def emit_Dropout(self, IR_node):
        code = "{:<15} = F.dropout(input = {}, p = {}, training = self.training, inplace = True)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.layer.attr["keep_prob"].f)
        return code


    def check_if_need_transpose(self, IR_node):
        parent = self.IR_graph.get_parent(IR_node.name, [0])
        while parent.type == 'Flatten' or parent.type == 'Dropout':
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

        if IR_node.get_attr('in_features') != None:
            in_features = IR_node.get_attr('in_features')

        self.add_init(2, "self.{} = self.__dense(name = '{}', in_features = {}, out_features = {}, bias = {})".format(
            IR_node.variable_name,
            IR_node.name,
            in_features,
            IR_node.layer.attr["units"].i,
            IR_node.IR_layer.attr["use_bias"].b))

        input_node = self.parent_variable_name(IR_node)
        if len(self.IR_graph.get_parent(IR_node.name, [0]).get_attr('_output_shapes')[0].dim) > 2:
            input_node = "{}.view({}.size(0), -1)".format(input_node, input_node)
        
        code = "{:<15} = self.{}({})".format(
            IR_node.variable_name,
            IR_node.variable_name,
            input_node)

        if self.weight_loaded:
            self.check_if_need_transpose(IR_node)
            self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], (1, 0))

        return code

    def emit_Flatten(self, IR_node):
        parent = self.IR_graph.get_parent(IR_node.name, [0]).real_variable_name
        code = "{:<15} = {}.view({}.size(0), -1)".format(
            IR_node.variable_name,
            parent,
            parent)
        return code


    def emit_Reshape(self, IR_node):
        shape_list = IR_node.get_attr('shape')
        shape_str = ','.join([str(int(i)) for i in shape_list])
        code = "{:<15} = torch.reshape(input = {}, shape = ({}))".format(
            IR_node.variable_name,
            self.IR_graph.get_node(IR_node.in_edges[0]).real_variable_name,
            shape_str)
        return code


    def emit_Tanh(self, IR_node):
        code = "{:<15} = F.tanh({})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node, [0]))
        return code


    def emit_Relu(self, IR_node):
        code = "{:<15} = F.relu({})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node, [0]))
        return code


    def emit_LeakyRelu(self, IR_node):
        code = "{:<15} = F.leaky_relu({}, negative_slope={})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node, [0]),
            IR_node.get_attr('alpha'))
        return code


    def emit_Relu6(self, IR_node):
        code = "{:<15} = F.relu6({})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node, [0]))
        return code


    def emit_Softmax(self, IR_node):
        code = "{:<15} = F.softmax({})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node, [0]))
        return code


    def emit_Sigmoid(self, IR_node):
        code = "{:<15} = F.sigmoid({})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node)
        )
        return code


    def emit_Embedding(self, IR_node):
        self.used_layers.add("Embedding")
        self.add_init(2, "self.{} = self.__embedding('{}', num_embeddings={}, embedding_dim={})".format(
            IR_node.variable_name,
            IR_node.name,
            IR_node.get_attr('input_dim'),   #2-D
            IR_node.get_attr('output_dim')
            ))
        
        code = "{:<15} = self.{}({})".format(
            IR_node.variable_name,
            IR_node.variable_name,
            "torch.LongTensor(np.array({}))".format(self.parent_variable_name(IR_node))
        )
        return code


    def emit_RNNs(self, IR_node, func):
        raise NotImplementedError()
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
        code = "{:<15} = {} + {}".format(
            IR_node.variable_name,
             self.parent_variable_name(IR_node),
             self.parent_variable_name(IR_node, [1]))
        return code


    def emit_Sub(self, IR_node):
        code = "{:<15} = {}".format(
            IR_node.variable_name,
            ' - '.join(self.parent_variable_name(IR_node, [idx]) for idx in range(len(IR_node.in_edges))))
        return code


    def emit_Mul(self, IR_node):
        code = "{:<15} = {}".format(
            IR_node.variable_name,
            ' * '.join(self.parent_variable_name(IR_node, [idx]) for idx in range(len(IR_node.in_edges))))
        return code


    def emit_MatMul(self, IR_node):
        code = "{:<15} = torch.matmul({})".format(
            IR_node.variable_name,
            ' , '.join('%s' % self.IR_graph.get_node(s).real_variable_name for s in IR_node.in_edges))
        return code


    def emit_Constant(self, IR_node):
        if IR_node.get_attr('value'):
            value = IR_node.get_attr('value')
            if not isinstance(value, list):
                value = [value]
            code = "self.{:<15} = torch.autograd.Variable(torch.Tensor({}), requires_grad=False)".format(
                IR_node.variable_name,
                value)
        else:
            code = "self.{:<15} = torch.autograd.Variable(torch.from_numpy(_weights_dict['{}']['value']), requires_grad=False)".format(
                IR_node.variable_name,
                IR_node.name)
        
        # self.add_init(2, "self.{:<15} = torch.from_numpy(_weights_dict['{}']['value'])".format(
        #     IR_node.variable_name,
        #     IR_node.name))
        IR_node.real_name = "self." + IR_node.variable_name
        return code


    def _convert_axis(self, IR_node, axis):
        ndim = len(self.IR_graph.get_parent(IR_node.name, [0]).get_attr('_output_shapes')[0].dim)
        if axis == 0:
            return 0
        elif axis == ndim - 1:
            return 1
        else:
            return axis + 1


    def emit_Concat(self, IR_node):
        axis = self._convert_axis(IR_node, IR_node.get_attr('axis'))
        code = "{:<15} = torch.cat(({},), {})".format(
            IR_node.variable_name,
            ', '.join(self.parent_variable_name(IR_node, [idx]) for idx in range(len(IR_node.in_edges))),
            axis,
        )
        return code


    def emit_BatchNorm(self, IR_node):
        self.used_layers.add(IR_node.type)
        dim = len(IR_node.layer.attr['_output_shapes'].list.shape[0].dim) - 2

        output_shape = IR_node.layer.attr['_output_shapes'].list.shape[0]
        if IR_node.get_attr('data_format', "NHWC") == "NCHW":
            num_features = output_shape.dim[1].size
        else:
            num_features = output_shape.dim[-1].size

        self.add_init(2, "self.{} = self.__batch_normalization({}, '{}', num_features={}, eps={}, momentum={})".format(
             IR_node.variable_name,
             dim,
             IR_node.name,
             num_features,
             IR_node.layer.attr['epsilon'].f,
             IR_node.layer.attr['momentum'].f,
        ))

        code = "{:<15} = self.{}({})".format(
            IR_node.variable_name,
            IR_node.variable_name,
            self.parent_variable_name(IR_node)
        )
        return code


    def emit_Scale(self, IR_node):
        self.used_layers.add(IR_node.type)
        dim = len(IR_node.layer.attr['_output_shapes'].list.shape[0].dim) - 2

        self.add_init(2, "self.{} = self.__scale({}, '{}', num_features={})".format(
             IR_node.variable_name,
             dim,
             IR_node.name,
             IR_node.layer.attr['_output_shapes'].list.shape[0].dim[-1].size
        ))

        code = "{:<15} = self.{}({})".format(
            IR_node.variable_name,
            IR_node.variable_name,
            self.parent_variable_name(IR_node)
        )
        return code


    def emit_Squeeze(self, IR_node):
        code = "{:<15} = torch.squeeze({})".format(
            IR_node.variable_name, self.parent_variable_name(IR_node)
        )
        return code


    @staticmethod
    def _convert_padding(IR_node):
        padding = IR_node.get_attr('pads')
        padding = convert_onnx_pad_to_tf(padding)[1:-1]
        new_padding = []
        for pad in padding:
            new_padding.insert(0, pad)
        return tuple(np.array(new_padding).reshape(-1).tolist())


    def emit_Pad(self, IR_node):
        if IR_node.get_attr('mode').lower() == 'constant':
            mode = "mode = 'constant', value = {}".format(0)
        elif IR_node.get_attr('mode').lower() == 'reflect':
            mode = "mode = 'reflect'"
        elif IR_node.get_attr('mode').upper() == 'SYMMETRIC':
            mode = "mode = 'replicate'"
        else:
            assert False

        padding = self._convert_padding(IR_node)
        code = "{:<15} = F.pad({}, {}, {})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            padding,
            mode)
        return code


    def emit_ReduceMean(self, IR_node):
        axes = [self._convert_axis(IR_node, x) for x in IR_node.get_attr('axes')]
        input_node = self.parent_variable_name(IR_node)
        codes = []
        for axis in sorted(axes, reverse=True):
            code = "{:<15} = torch.mean({}, {}, {})".format(
                IR_node.variable_name,
                input_node,
                axis,
                IR_node.get_attr("keepdims")
            )
            codes.append(code)
            input_node = IR_node.variable_name
        return codes


    def emit_LRN(self, IR_node):
        output_name = IR_node.variable_name
        input_name = self.parent_variable_name(IR_node)
        size = IR_node.get_attr('size')
        alpha = IR_node.get_attr('alpha')
        beta = IR_node.get_attr('beta')
        bias = IR_node.get_attr('bias', 1)

        code =  "{:<15} = F.local_response_norm({}, size={}, alpha={}, beta={}, k={})".format(
            output_name,
            input_name,
            size,
            alpha,
            beta,
            bias
        )
        return code


    def emit_DepthwiseConv(self, IR_node):
        return self.emit_Conv(IR_node)


    def emit_Const(self, IR_node):
        if 'dtype' in IR_node.layer.attr:
            dtype_str = "dtype={}".format(self.dtype_map[IR_node.layer.attr['dtype'].type])
            if 'int' in dtype_str:
                code = "{:<15} = torch.tensor({}, {})".format(
                    IR_node.variable_name,
                    IR_node.layer.attr['value'].i,
                    dtype_str)
            else:
                code = "{:<15} = torch.tensor({}, {})".format(
                    IR_node.variable_name,
                    IR_node.layer.attr['value'].f,
                    dtype_str)

        else:
            dtype_str = "dtype=torch.float32"
            code = "{:<15} = torch.tensor({}, {})".format(
                IR_node.variable_name,
                IR_node.layer.attr['value'].f,
                dtype_str)
        return code


    def emit_Shape(self, IR_node):
        code = "{:<15} = torch.Tensor(list({}.size()))".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node)
            )
        return code


    def emit_Pack(self, IR_node):
        code = "{:<15} = {}".format(
            IR_node.variable_name,
            '[' +  ','.join('%s' % self.IR_graph.get_node(s).real_variable_name for s in IR_node.in_edges) + ']',
            )
        return code


    def emit_Slice(self, IR_node):
        starts = IR_node.get_attr('starts')
        if len(starts) > 1:
            starts = [starts[0], starts[-1]] + starts[1:-1]
        ends = IR_node.get_attr('ends')
        if len(ends) > 1:
            ends = [ends[0], ends[-1]] + ends[1:-1]
        extra_str = ""
        for idx, _ in enumerate(starts):
            if idx:
                extra_str += ", "
            extra_str += "{}:".format(starts[idx])
            if ends[idx]:
                extra_str += "{}".format(ends[idx])
        
        shrink_mask = IR_node.get_attr('shrink_axis_mask')

        if shrink_mask:
            mask = [int(s) for s in bin(shrink_mask)[2:][::-1]]
            shrink_str = '[' + ','.join(':' if bit==0 else '0' for bit in mask) + ']'
        else:
            shrink_str = ''
        code = "{:<15} = {}[{}]{}".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            extra_str,
            shrink_str
        )
        return code


    def emit_Split(self, IR_node):

        if isinstance(IR_node.get_attr('split'), list):
            split_str = IR_node.get_attr('split')
        else:
            num_split = IR_node.get_attr('split')
            split_str = "math.ceil({}.shape[{}]/{})".format(
                self.parent_variable_name(IR_node), 
                IR_node.get_attr('axis'),
                num_split)
        code = "{:<15} = torch.split({}, {}, dim={})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            split_str,
            IR_node.get_attr('axis'),
        )
        return code


    def emit_Unstack(self, IR_node):
        code = "{:<15} = torch.unbind({}, dim={})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('axis')
        )
        return code


    def emit_Fill(self, IR_node):
        code = "{:<15} = torch.full({}.int().numpy().tolist(), {})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('value')
        )
        return code


    def emit_Gather(self, IR_node):
        pass


    def emit_Unsqueeze(self, IR_node):
        code = "{:<15} = {}.unsqueeze({})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('axes')[0]
        )
        return code


    def emit_Transpose(self, IR_node):
        code = "{:<15} = {}.permute({})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            self.parent_variable_name(IR_node, [1]))
        return code


    def emit_Minimum(self, IR_node):
        code = "{:<15} = torch.min({}, {})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            self.parent_variable_name(IR_node, [1]))
        return code


    def emit_Maxmum(self, IR_node):
        code = "{:<15} = torch.max({}, {})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            self.parent_variable_name(IR_node, [1]))
        return code


    def emit_Square(self, IR_node):
        code = "{:<15} = {}.pow(2)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node))
        return code


    def emit_PRelu(self, IR_node):
        code = "{:<15} = F.prelu({}, torch.from_numpy(_weights_dict['{}']['weights']))".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node, [0]),
            IR_node.name)
        
        if self.weight_loaded:
            self.weights_dict[IR_node.name]['weights'] = self.weights_dict[IR_node.name]['gamma']
        
        return code


    def emit_Cast(self, IR_node):
        dstType = IR_node.get_attr('dstType')

        if dstType == 'float':
            dst = 'torch.FloatTensor'
        elif dstType == 'double':
            dst = 'torch.DoubleTensor'
        elif dstType == 'int':
            dst = 'torch.IntTensor'
        
        code = "{:<15} = {}.type({})".format(
            IR_node.real_variable_name,
            self.parent_variable_name(IR_node),
            dst)

        return code


    def emit_Scope(self, IR_node):
        input_vars = [self.parent_variable_name(IR_node, [idx]) for idx in range(len(IR_node.in_edges))]
        code = "{:<15} = self.__{}({})".format(
            IR_node.real_variable_name,
            IR_node.pattern,
            ', '.join(input_vars))
        self._gen_scope_code(IR_node)
        return code


    def _gen_scope_code(self, scope_node):

        def _scope_func(scope_name, params, code, return_var):
            code = """
    def __{}({}):
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
                        body_code += "        " + line + '\n'
                else:
                    print("PytorchEmitter has not supported operator [%s]." % (node_type))
                    self.emit_UNKNOWN(node)

            # param_code does not need parameter slice.
            input_params = scope_node.input_params
            input_params.insert(0, "self")
            param_code = ', '.join(input_params)
            function_code = _scope_func(scope_node.pattern, param_code, body_code, scope_node.return_variables)

            self.layers_codes[scope_node.pattern] = function_code


    def _layer_Embedding(self):
        self.add_body(0,"""
    @staticmethod
    def __embedding(name, **kwargs):
        layer = nn.Embedding(**kwargs) #shape
        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        return layer
        """)


    def _layer_Conv(self):
        self.add_body(0, """
    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer""")


    def _layer_FullyConnected(self):
        self.add_body(0, """
    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer""")


    def _layer_BatchNorm(self):
        self.add_body(0, """
    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in _weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(_weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(_weights_dict[name]['var']))
        return layer""")


    def _layer_Scale(self):
        self.add_body(0, """
    # from torch.nn.parameter import Parameter

    class _Scale(nn.Module):

        def __init__(self, num_features, affine=True):
            super(KitModel._Scale, self).__init__()
            self.num_features = num_features
            self.affine = affine

            self.running_mean = torch.zeros(num_features)
            self.running_var = torch.ones(num_features)
            self.training = False
            self.eps = 1e-5
            if self.affine:
                self.weight = nn.Parameter(torch.Tensor(num_features))
                self.bias = nn.Parameter(torch.Tensor(num_features))
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)
            self.reset_parameters()


        def reset_parameters(self):
            if self.affine:
                self.weight.data.uniform_()
                self.bias.data.zero_()

        def _check_input_dim(self, input):
            raise NotImplementedError

        def forward(self, input):
            self._check_input_dim(input)

            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training,
                0 , self.eps)


    class Scale1d(_Scale):

        def _check_input_dim(self, input):
            if input.dim() != 2 and input.dim() != 3:
                raise ValueError('expected 2D or 3D input (got {}D input)'
                                .format(input.dim()))



    class Scale2d(_Scale):


        def _check_input_dim(self, input):
            if input.dim() != 4:
                raise ValueError('expected 4D input (got {}D input)'
                                .format(input.dim()))


    class Scale3d(_Scale):

        def _check_input_dim(self, input):
            if input.dim() != 5:
                raise ValueError('expected 5D input (got {}D input)'
                                .format(input.dim()))


    @staticmethod
    def __scale(dim, name, **kwargs):
        if   dim == 1:  layer = KitModel.Scale1d(**kwargs)
        elif dim == 2:  layer = KitModel.Scale2d(**kwargs)
        elif dim == 3:  layer = KitModel.Scale3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in _weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        return layer""")
