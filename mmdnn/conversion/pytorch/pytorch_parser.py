#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import os
import numpy as np
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.utils import *
from mmdnn.conversion.common.DataStructure.parser import Parser
from mmdnn.conversion.pytorch.pytorch_graph import PytorchGraph040
from mmdnn.conversion.pytorch.pytorch_graph import PytorchGraph151
import torch
import torchvision

class PytorchParser(Parser):

    layer_map = {
    'onnx::Conv': 'Conv',
    'onnx::Flatten': 'Flatten',
    'onnx::Gemm': 'FullyConnected',
    'onnx::MaxPool': 'Maxpool',
    'onnx::AveragePool': 'Avgpool',
    'onnx::GlobalAveragePool': 'GAvgpool',
    'onnx::Dropout': 'Dropout',
    'onnx::BatchNormalization': 'BatchNormalization',
    'onnx::Add': 'Add',
    'onnx::Concat': 'Concat',
    'onnx::Relu': 'Relu',
    'onnx::Tanh': 'Tanh',
    'onnx::Sigmoid': 'Sigmoid',
    'onnx::Mul': 'Mul',
    'onnx::Pad': 'Pad'


    # TODO
    # 'max_pool2d': convert_maxpool,
    # 'onnx::Mul': convert_elementwise_mul,
    # 'onnx::Sub': convert_elementwise_sub,
    # 'onnx::ConvTranspose': convert_convtranspose,
    # 'onnx::LeakyRelu': convert_lrelu,
    # 'onnx::Sigmoid': convert_sigmoid,
    # 'onnx::Softmax': convert_softmax,
    # 'onnx::Selu': convert_selu,
    # 'onnx::Transpose': convert_transpose,
    # 'onnx::Reshape': convert_reshape,
    # 'onnx::MatMul': convert_matmul,
    # 'onnx::Gather': convert_gather,
    # 'onnx::ReduceSum': convert_reduce_sum,
    # 'onnx::Constant': convert_constant,
    # 'onnx::Upsample': convert_upsample,
    # 'onnx::Pad': convert_padding,
}


    ############
    # property #
    ############

    @property
    def src_graph(self):
        return self.pytorch_graph

    def get_weight_name(self, node):
        pass

    ####################
    # Public Functions #
    ####################

    def __init__(self, model_file_name, input_shape):
        super(PytorchParser, self).__init__()
        if not os.path.exists(model_file_name):
            print("Pytorch model file [{}] is not found.".format(model_file_name))
            assert False
        # test

        # cpu: https://github.com/pytorch/pytorch/issues/5286
        try:
            model = torch.load(model_file_name)
        except:
            model = torch.load(model_file_name, map_location='cpu')

        self.weight_loaded = True
        self.model = model
        # Build network graph
        self.pytorch_graph = None

    def build_graph(self, input_shape):
        self.input_shape = tuple([1] + input_shape)
        self.pytorch_graph.build(self.input_shape)
        self.state_dict = self.pytorch_graph.state_dict
        self.shape_dict = self.pytorch_graph.shape_dict

    def gen_IR(self):
        for layer in self.src_graph.topological_sort:
            current_node = self.src_graph.get_node(layer)
            onnx_node_type = current_node.type
            if onnx_node_type not in PytorchParser.layer_map.keys():
                print("PyTorch parser has not supported operator [%s]. IR network strucuture may lost info."
                        % (onnx_node_type))
                return
            node_type = PytorchParser.layer_map[onnx_node_type]


            if hasattr(self, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                func(current_node)

            else:
                self.rename_UNKNOWN(current_node)

        self.gen_Input()



    def _set_output_shape(self, source_node, IR_node):

        shape = graph_pb2.TensorShape()


        layer_name = source_node.name

        shape_pytorch = self.shape_dict[layer_name]


        new_dim = shape.dim.add()

        if not shape_pytorch:
            print("Warning: Pytorch cannot inference outputshape of \"{}\" with operator \"{}\". Setting outputshape manually in json file is alternative .".format(source_node.name, source_node.type))
            IR_node.attr["_output_shapes"].list.shape.extend([shape])
            return 

        # (batch, C, H, W)  & NHWC
        if len(shape_pytorch) == 4:

            if shape_pytorch[0] == 1:
                new_dim.size = -1
            else:
                new_dim.size = shape_pytorch[0]
            for index in [2, 3, 1]:
                new_dim = shape.dim.add()
                dim = shape_pytorch[index]
                new_dim.size = dim if dim else -1
        elif len(shape_pytorch) == 2:
            if shape_pytorch[0] == 1:
                new_dim.size = -1
            else:
                new_dim.size = shape_pytorch[0]
            for _ in range(2):
                new_dim = shape.dim.add()
                new_dim.size = 1
            new_dim = shape.dim.add()
            dim = shape_pytorch[1]
            new_dim.size = dim if dim else -1


        IR_node.attr["_output_shapes"].list.shape.extend([shape])

    ##########
    # Layers #
    ##########
    def rename_UNKNOWN(self, source_node):
        print("PyTorch parser has not supported operator [%s] with name [%s]."
              % (source_node.type, source_node.name))
        assert False      
        print(source_node.layer)
        print(source_node.layer.data.size())
        
        


    def gen_Input(self):
        IR_node = self.IR_graph.node.add()
        IR_node.name = 'input'
        IR_node.op = "DataInput"

        for node in self.IR_graph.node:
            if node.name in self.src_graph.input_layers:
                node.input.append('input')

        assert len(self.input_shape) == 4
        new_dim = IR_node.attr["shape"].shape.dim.add()
        if self.input_shape[0] == 1:
            new_dim.size = -1
        else:
            new_dim.size = self.input_shape[0]
        for index in [2, 3, 1]:
            new_dim = IR_node.attr["shape"].shape.dim.add()
            new_dim.size = self.input_shape[index]

        shape = graph_pb2.TensorShape()
        new_dim = shape.dim.add()
        shape_pytorch = self.input_shape

        if len(shape_pytorch) == 4:

            if shape_pytorch[0] == 1:
                new_dim.size = -1
            else:
                new_dim.size = shape_pytorch[0]
            for index in [2, 3, 1]:
                new_dim = shape.dim.add()
                dim = shape_pytorch[index]
                new_dim.size = dim if dim else -1
        elif len(shape_pytorch) == 2:
            if shape_pytorch[0] == 1:
                new_dim.size = -1
            else:
                new_dim.size = shape_pytorch[0]
            for _ in range(2):
                new_dim = shape.dim.add()
                new_dim.size = 1
            new_dim = shape.dim.add()
            dim = shape_pytorch[1]
            new_dim.size = dim if dim else -1


        IR_node.attr["_output_shapes"].list.shape.extend([shape])


    def rename_Conv(self, source_node):

        attr = source_node.attrs
        kwargs = dict()

        # dilation
        if 'dilations' in attr:
            kwargs['dilations'] = [1] + attr['dilations'] + [1]
        else:
            kwargs['dilations'] = [1] + [1, 1] + [1]

        if len(attr['pads']) == 4:
            kwargs['pads'] = [0] + attr['pads'][0:2] + [0, 0] + attr['pads'][2:] + [0]
        elif len(attr['pads']) == 2:
            kwargs['pads'] = ( [0] + attr['pads'][0:2] + [0] ) *2

        if 'strides' not in attr:
            kwargs['strides'] = [1] + [1, 1] + [1]
        else:
            kwargs['strides'] = [1] + attr['strides'] + [1]

        kwargs['group'] = attr['group']

        weights_scope = self.get_weight_name(source_node)

        bias_name = '{0}.bias'.format(weights_scope)
        weights_name = '{0}.weight'.format(weights_scope)
        weight = self.state_dict[weights_name]

        weight = weight.numpy()
        dim = weight.ndim - 2


        IR_node = self._convert_identity_operation(source_node, new_op="Conv")
        weight = np.transpose(weight, list(range(2, dim + 2)) + [1, 0])

        self.set_weight(source_node.name, 'weights', weight)
        kwargs['kernel_shape'] = list(weight.shape)


        # handle bias
        if bias_name in self.state_dict:
            bias = self.state_dict[bias_name].numpy()
            self.set_weight(source_node.name, 'bias', bias)
            kwargs['use_bias'] = True
        else:
            kwargs['use_bias'] = False


        assign_IRnode_values(IR_node, kwargs)


    def rename_BatchNormalization(self, source_node):
        # TODO
        # output_shape

        IR_node = self._convert_identity_operation(source_node, new_op="BatchNorm")


        attr = source_node.attrs
        # epsilon
        IR_node.attr['epsilon'].f = attr['epsilon']
        weights_scope = self.get_weight_name(source_node)

        bias_name = '{0}.bias'.format(weights_scope)
        weights_name = '{0}.weight'.format(weights_scope)
        mean_name = '{0}.running_mean'.format(weights_scope)
        var_name = '{0}.running_var'.format(weights_scope)



        if bias_name in self.state_dict:
            beta = self.state_dict[bias_name].numpy()
            IR_node.attr['bias'].b = True
        else:
            IR_node.attr['bias'].b = False

        if weights_name in self.state_dict:
            gamma = self.state_dict[weights_name].numpy()
            IR_node.attr['scale'].b = True
        else:
            IR_node.attr['scale'].b = False

        mean = self.state_dict[mean_name].numpy()
        variance = self.state_dict[var_name].numpy()



        if IR_node.attr['scale'].b:
            self.set_weight(source_node.name, "scale", gamma)

        if IR_node.attr['bias'].b:
            self.set_weight(source_node.name, "bias", beta)

        # mean
        self.set_weight(source_node.name, "mean", mean)

        # var
        self.set_weight(source_node.name, "var", variance)

    def rename_Pad(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op="Pad")
        attr = source_node.attrs
        kwargs = dict()
        kwargs['mode'] = attr['mode']
        kwargs['pads'] = attr['pads']
        kwargs['constant_values'] = attr['value']
        assign_IRnode_values(IR_node, kwargs)

    def rename_Relu(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op="Relu")

    def rename_Tanh(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op="Tanh")

    def rename_Sigmoid(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op="Sigmoid")

    def rename_Mul(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op="Mul")

    def rename_Maxpool(self, source_node):
        attr = source_node.attrs
        kwargs = dict()
        kwargs['strides'] = [1] + attr['strides'] + [1]
        if 'dilations' not in attr:
            kwargs['dilations'] = [1] + [1, 1] + [1]
        else:
            kwargs['dilations'] = [1] + attr['dilations'] + [1]
        kwargs['pads'] = [0] + attr['pads'][0:2] + [0, 0] + attr['pads'][2:] + [0]
        kwargs['kernel_shape'] = [1] + attr['kernel_shape'] + [1]
        IR_node = self._convert_identity_operation(source_node, new_op="Pool")

        kwargs['pooling_type'] = 'MAX'

        assign_IRnode_values(IR_node, kwargs)

    def rename_Avgpool(self, source_node):
        attr = source_node.attrs
        kwargs = dict()
        kwargs['strides'] = [1] + attr['strides'] + [1]
        if 'dilations' not in attr:
            kwargs['dilations'] = [1] + [1, 1] + [1]
        else:
            kwargs['dilations'] = [1] + attr['dilations'] + [1]
        if 'pads' in attr:
            kwargs['pads'] = [0] + attr['pads'][0:2] + [0, 0] + attr['pads'][2:] + [0]
        else:
            kwargs['pads'] = [0, 0, 0, 0, 0, 0, 0, 0]
        kwargs['kernel_shape'] = [1] + attr['kernel_shape'] + [1]
        IR_node = self._convert_identity_operation(source_node, new_op="Pool")

        kwargs['pooling_type'] = 'AVG'

        assign_IRnode_values(IR_node, kwargs)

    def rename_GAvgpool(self, source_node):
        attr = source_node.attrs
        input_shape = self.pytorch_graph.shape_dict[source_node.in_edges[0]]
        kwargs = dict()
        kwargs['strides'] = [1, 1, 1, 1]
        kwargs['dilations'] = [1] + [1, 1] + [1]
        kwargs['pads'] = [0, 0, 0, 0, 0, 0, 0, 0]
        kwargs['kernel_shape'] = [1] + input_shape[2:] + [1]
        IR_node = self._convert_identity_operation(source_node, new_op="Pool")

        kwargs['pooling_type'] = 'AVG'

        assign_IRnode_values(IR_node, kwargs)

    def rename_Flatten(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op="Flatten")

    def rename_FullyConnected(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op="FullyConnected")
        weights_scope = self.get_weight_name(source_node)
        bias_name = '{0}.bias'.format(weights_scope)
        weights_name = '{0}.weight'.format(weights_scope)


        W = self.state_dict[weights_name].numpy().transpose()
        input_channels, output_channels = W.shape

        # Kit weight tranpose
        # weight: N x M -> C x H x W x M -> H x W x C x M -> N x M
        if self.weight_loaded:
            parent = self.src_graph.get_parent(source_node.name, [0])
            if parent:
                while parent.type == 'onnx::Flatten' or parent.type == 'onnx::Dropout':
                    parent = self.src_graph.get_parent(parent.name, [0])
                if len(self.shape_dict[parent.name]) == 4:
                    #
                    original_shape = W.shape
                    channel_first_list = self.shape_dict[parent.name][1:]
                    dim = len(channel_first_list) + 1
                    weight = W.reshape(channel_first_list + [original_shape[1]])
                    assert dim > 2
                    weight = weight.transpose(list(range(1, dim-1)) + [0, dim-1])
                    W = weight.reshape(original_shape)

        # weights
        self.set_weight(source_node.name, 'weights', W )

        # use_bias
        if bias_name in self.state_dict:
            IR_node.attr['use_bias'].b = True
            bias = self.state_dict[bias_name].numpy()
            self.set_weight(source_node.name, 'bias', bias )
        else:
            IR_node.attr['use_bias'].b = False

        # units
        IR_node.attr['units'].i = output_channels


    def rename_Dropout(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='Dropout')
        IR_node.attr['keep_prob'].f = source_node.attrs['ratio']

    def rename_Concat(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='Concat')

        if source_node.attrs['axis'] == 1:
            IR_node.attr['axis'].i = len(self.shape_dict[source_node.name]) - 1
        else:
            IR_node.attr['axis'].i = source_node.attrs['axis']

    def rename_Add(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='Add')


    def rename_MaxPool2d(self, source_node):
        self._convert_pooling(source_node)


    def rename_View(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='Reshape')
        assign_IRnode_values(IR_node, {'shape' : list(source_node.get_attr('new_sizes'))[1:]})


    def rename_Addmm(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='FullyConnected')
        kwargs = dict()

        # handle weight
        weight = source_node.get_attr('next_functions')[2][0].next_functions[0][0].variable.data.numpy()
        weight = np.transpose(weight)
        kwargs['units'] = weight.shape[1]
        self.set_weight(source_node.name, 'weights', weight)

        # handle bias
        if source_node.get_attr('next_functions')[0][0]:
            bias = source_node.get_attr('next_functions')[0][0].variable.data.numpy()
            kwargs['use_bias'] = True
            self.set_weight(source_node.name, 'bias', weight)

        assign_IRnode_values(IR_node, kwargs)



    ####################
    # Helper Functions #
    ####################

    @staticmethod
    def _copy_and_reop(source_node, IR_node, new_op = None):
        if new_op == None: new_op = source_node.type
        IR_node.name = source_node.name
        IR_node.op = new_op


    def _convert_identity_operation(self, source_node, in_edge_count = None, new_op = None):
        IR_node = self.IR_graph.node.add()
        PytorchParser._copy_and_reop(source_node, IR_node, new_op)
        self.convert_inedge(source_node, IR_node, 0, in_edge_count)
        self._set_output_shape(source_node, IR_node)
        return IR_node

    def _convert_pooling(self, source_node):
        kwargs = dict()
        kwargs['strides'] = [1] + list(source_node.get_attr('stride')) + [1]
        kwargs['dilations'] = [1] + list(source_node.get_attr('dilation')) + [1]
        kwargs['pads'] = ([0] + list(source_node.get_attr('padding')) + [0]) * 2
        kwargs['kernel_shape'] = [1] + list(source_node.get_attr('kernel_size')) + [1]
        IR_node = self._convert_identity_operation(source_node, new_op="Pool")

        if source_node.name.startswith('Max'):
            kwargs['pooling_type'] = 'MAX'
        elif source_node.name.startswith('Avg'):
            kwargs['pooling_type'] = 'AVG'
        else:
            raise ValueError('Unknown pooling type')

        assign_IRnode_values(IR_node, kwargs)

class PytorchParser040(PytorchParser):

    def __init__(self, model_file_name, input_shape):
        super(PytorchParser040, self).__init__(model_file_name, input_shape)
        self.pytorch_graph = PytorchGraph040(self.model)
        self.build_graph(input_shape)

    def get_weight_name(self, node):
        return node.weights_name

class PytorchParser151(PytorchParser):

    def __init__(self, model_file_name, input_shape):
        super(PytorchParser151, self).__init__(model_file_name, input_shape)
        self.pytorch_graph = PytorchGraph151(self.model)
        self.build_graph(input_shape)

    def get_weight_name(self, node):
        return self.pytorch_graph.layer_weight_map[node.name]
    
