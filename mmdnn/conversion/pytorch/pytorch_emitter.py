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

        self.init_code = str()
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
            self.init_code += ("    " * indent) + code + '\n'


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

    def gen_code(self, phase):
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
            filter *= group

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
        self.add_body(2, "{:<15} = self.{}({})".format(
            IR_node.variable_name,
            IR_node.variable_name,
            input_node))

        if self.weight_loaded:
            if IR_node.type == 'DepthwiseConv':
                self.weights_dict[IR_node.name]['weights'] = np.swapaxes(self.weights_dict[IR_node.name]['weights'], -1, -2)
            self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], [dim + 1, dim] + list(range(0, dim)))


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
            self.add_body(2, "{:<15} = F.{}(input = {}, kernel_size = {}.size()[2:])".format(
                IR_node.variable_name,
                pool_name,
                self.parent_variable_name(IR_node),
                self.parent_variable_name(IR_node)
            ))

        else:

            if IR_node.get_attr('pooling_type') == "MAX":
                # Change to padding defuse
                input_node = self._defuse_padding(IR_node,", value=float('-inf')")
                for e in IR_node.get_attr('dilations', []):
                    assert e == 1

                pool_size = IR_node.get_attr('kernel_shape')[1:-1]
                strides = IR_node.get_attr('strides')[1:-1]

                self.add_body(2, "{:<15} = F.{}({}, kernel_size={}, stride={}, padding={}, ceil_mode={})".format(
                    IR_node.variable_name,
                    pool_name,
                    input_node,
                    tuple(pool_size),
                    tuple(strides),
                    0,
                    False
                    ))

            elif IR_node.get_attr('pooling_type') == "AVG":

                for e in IR_node.get_attr('dilations', []):
                    assert e == 1

                pool_size = IR_node.get_attr('kernel_shape')[1:-1]
                strides = IR_node.get_attr('strides')[1:-1]

                padding = IR_node.get_attr('pads')[1:dim]
                ceil_mode = self.is_ceil_mode(IR_node.get_attr('pads'))

                # input_node = self._defuse_padding(IR_node, exstr)
                self.add_body(2, "{:<15} = F.{}({}, kernel_size={}, stride={}, padding={}, ceil_mode={})".format(
                    IR_node.variable_name,
                    pool_name,
                    self.parent_variable_name(IR_node),
                    tuple(pool_size),
                    tuple(strides),
                    tuple(padding),
                    ceil_mode
                    ))

            else:
                raise ValueError()


    def emit_UNKNOWN(self, IR_node):
        print(IR_node.name)


    def emit_DataInput(self, IR_node):
        # Ignore it in Pytorch
        IR_node.real_name = 'x'


    def emit_Dropout(self, IR_node):
        self.add_body(2, "{:<15} = F.dropout(input = {}, p = {}, training = self.training, inplace = True)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.layer.attr["keep_prob"].f))


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

        self.add_init(2, "self.{} = self.__dense(name = '{}', in_features = {}, out_features = {}, bias = {})".format(
            IR_node.variable_name,
            IR_node.name,
            in_features,
            IR_node.layer.attr["units"].i,
            IR_node.IR_layer.attr["use_bias"].b))

        input_node = self.parent_variable_name(IR_node)
        if len(self.IR_graph.get_parent(IR_node.name, [0]).get_attr('_output_shapes')[0].dim) > 2:
            input_node = "{}.view({}.size(0), -1)".format(input_node, input_node)
        self.add_body(2, "{:<15} = self.{}({})".format(
            IR_node.variable_name,
            IR_node.variable_name,
            input_node))

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
        shape_list = IR_node.get_attr('shape')
        shape_str = ','.join([str(int(i)) for i in shape_list])
        self.add_body(2, "{:<15} = torch.reshape(input = {}, shape = ({}))".format(
            IR_node.variable_name,
            self.IR_graph.get_node(IR_node.in_edges[0]).real_variable_name,
            shape_str))


    def emit_Tanh(self, IR_node):
        self.add_body(2, "{:<15} = F.tanh({})".format(
            IR_node.variable_name,
            self.IR_graph.get_parent(IR_node.name, [0]).real_variable_name))


    def emit_Relu(self, IR_node):
        self.add_body(2, "{:<15} = F.relu({})".format(
            IR_node.variable_name,
            self.IR_graph.get_parent(IR_node.name, [0]).real_variable_name))


    def emit_LeakyRelu(self, IR_node):
        self.add_body(2, "{:<15} = F.leaky_relu({}, negative_slope={})".format(
            IR_node.variable_name,
            self.IR_graph.get_parent(IR_node.name, [0]).real_variable_name,
            IR_node.get_attr('alpha')))



    def emit_Relu6(self, IR_node):
        self.add_body(2, "{:<15} = F.relu6({})".format(
            IR_node.variable_name,
            self.IR_graph.get_parent(IR_node.name, [0]).real_variable_name))


    def emit_Softmax(self, IR_node):
        self.add_body(2, "{:<15} = F.softmax({})".format(
            IR_node.variable_name,
            self.IR_graph.get_parent(IR_node.name, [0]).real_variable_name))


    def emit_Sigmoid(self, IR_node):
        code = "{:<15} = Activation(name = '{}', activation = 'sigmoid')({})".format(
                IR_node.variable_name,
                IR_node.name,
                self.IR_graph.get_parent(IR_node.name, [0]).real_variable_name)
        return code


    def emit_Embedding(self, IR_node):
        raise NotImplementedError()
        ret = "{:<15} = Embedding(input_dim = {}, output_dim = {}, mask_zero = {})({})".format(
                IR_node.name,
                IR_node.IR_layer.attr['input_dim'].i,
                IR_node.IR_layer.attr['output_dim'].i,
                IR_node.IR_layer.attr['mask_zero'].b,
                IR_node.in_edges[0])

        return ret


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
        self.add_body(2, "{:<15} = {}".format(
            IR_node.variable_name,
            ' + '.join('%s' % self.IR_graph.get_node(s).real_variable_name for s in IR_node.in_edges)))

    def emit_Sub(self, IR_node):
        self.add_body(2, "{:<15} = {}".format(
            IR_node.variable_name,
            ' - '.join('%s' % self.IR_graph.get_node(s).real_variable_name for s in IR_node.in_edges)))

    def emit_Mul(self, IR_node):
        self.add_body(2, "{:<15} = {}".format(
            IR_node.variable_name,
            ' * '.join('%s' % self.IR_graph.get_node(s).real_variable_name for s in IR_node.in_edges)))


    def emit_Constant(self, IR_node):
        self.add_init(2, "self.{:<15} = torch.autograd.Variable(torch.Tensor(__weights_dict['{}']['value']), requires_grad=False)".format(
            IR_node.variable_name,
            IR_node.name))

        # self.add_init(2, "self.{:<15} = torch.from_numpy(__weights_dict['{}']['value'])".format(
        #     IR_node.variable_name,
        #     IR_node.name))
        IR_node.real_name = "self." + IR_node.variable_name


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
        self.add_body(2, "{:<15} = torch.cat(({}), {})".format(
            IR_node.variable_name,
            ', '.join(self.IR_graph.get_node(s).real_variable_name for s in IR_node.in_edges),
            axis,
        ))


    def emit_BatchNorm(self, IR_node):
        self.used_layers.add(IR_node.type)
        dim = len(IR_node.layer.attr['_output_shapes'].list.shape[0].dim) - 2

        self.add_init(2, "self.{} = self.__batch_normalization({}, '{}', num_features={}, eps={}, momentum={})".format(
             IR_node.variable_name,
             dim,
             IR_node.name,
             IR_node.layer.attr['_output_shapes'].list.shape[0].dim[-1].size,
             IR_node.layer.attr['epsilon'].f,
             IR_node.layer.attr['momentum'].f,
        ))

        self.add_body(2, "{:<15} = self.{}({})".format(
            IR_node.variable_name,
            IR_node.variable_name,
            self.parent_variable_name(IR_node)
        ))

    def emit_Scale(self, IR_node):
        self.used_layers.add(IR_node.type)
        dim = len(IR_node.layer.attr['_output_shapes'].list.shape[0].dim) - 2

        self.add_init(2, "self.{} = self.__scale({}, '{}', num_features={})".format(
             IR_node.variable_name,
             dim,
             IR_node.name,
             IR_node.layer.attr['_output_shapes'].list.shape[0].dim[-1].size
        ))

        self.add_body(2, "{:<15} = self.{}({})".format(
            IR_node.variable_name,
            IR_node.variable_name,
            self.parent_variable_name(IR_node)
        ))


    def emit_Squeeze(self, IR_node):
        self.add_body(2, "{:<15} = torch.squeeze({})".format(
            IR_node.variable_name, self.parent_variable_name(IR_node)
        ))


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
        self.add_body(2, "{:<15} = F.pad({}, {}, {})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            padding,
            mode))


    def emit_ReduceMean(self, IR_node):
        axes = [self._convert_axis(IR_node, x) for x in IR_node.get_attr('axes')]
        input_node = self.parent_variable_name(IR_node)
        for axis in sorted(axes, reverse=True):
            self.add_body(2, "{:<15} = torch.mean({}, {}, {})".format(
                IR_node.variable_name,
                input_node,
                axis,
                IR_node.get_attr("keepdims")
            ))
            input_node = IR_node.variable_name


    def emit_LRN(self, IR_node):
        self.used_layers.add(IR_node.type)
        self.add_body(2, "{:<15} = self.LRN(size = {}, alpha = {}, beta = {})({})".format(
            IR_node.variable_name,
            IR_node.layer.attr['size'].i * 2 - 1,
            IR_node.layer.attr['alpha'].f,
            IR_node.layer.attr['beta'].f,
            self.parent_variable_name(IR_node)
        ))


    def emit_DepthwiseConv(self, IR_node):
        self.emit_Conv(IR_node)


    def emit_Slice(self, IR_node):
        starts = IR_node.get_attr('starts')
        starts = [starts[0], starts[-1]] + starts[1:-1]
        ends = IR_node.get_attr('ends')
        ends = [ends[0], ends[-1]] + ends[1:-1]
        extra_str = ""
        for idx, _ in enumerate(starts):
            if idx:
                extra_str += ", "
            extra_str += "{}:".format(starts[idx])
            if ends[idx]:
                extra_str += "{}".format(ends[idx])

        self.add_body(2, "{:<15} = {}[{}]".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            extra_str
        ))


    def _layer_Conv(self):
        self.add_body(0, """
    @staticmethod
    def __conv(dim, name, **kwargs):
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


    def _layer_BatchNorm(self):
        self.add_body(0, """
    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in __weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(__weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(__weights_dict[name]['var']))
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

        if 'scale' in __weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        return layer""")




    def _layer_LRN(self):
        self.add_body(0, """
    class LRN(nn.Module):
        def __init__(self, size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=False):
            super(KitModel.LRN, self).__init__()
            self.ACROSS_CHANNELS = ACROSS_CHANNELS
            if self.ACROSS_CHANNELS:
                self.average=nn.AvgPool3d(kernel_size=(size, 1, 1),
                        stride=1,
                        padding=(int((size-1.0)/2), 0, 0))
            else:
                self.average=nn.AvgPool2d(kernel_size=size,
                        stride=1,
                        padding=int((size-1.0)/2))
            self.alpha = alpha
            self.beta = beta

        def forward(self, x):
            if self.ACROSS_CHANNELS:
                div = x.pow(2).unsqueeze(1)
                div = self.average(div).squeeze(1)
                div = div.mul(self.alpha).add(1.0).pow(self.beta)
            else:
                div = x.pow(2)
                div = self.average(div)
                div = div.mul(self.alpha).add(1.0).pow(self.beta)
            x = x.div(div)
            return x""")
