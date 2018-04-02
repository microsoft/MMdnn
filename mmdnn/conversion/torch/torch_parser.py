#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import os
import numpy as np
from torch.utils.serialization import load_lua
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.utils import *
from mmdnn.conversion.common.DataStructure.parser import Parser
from mmdnn.conversion.torch.torch_graph import TorchGraph


class TorchParser(Parser):

    ############
    # property #
    ############

    @property
    def src_graph(self):
        return self.torch_graph


    ####################
    # Public Functions #
    ####################

    def __init__(self, model_file_name, input_shape):
        super(TorchParser, self).__init__()
        if not os.path.exists(model_file_name):
            raise ValueError("Torch7 model file [{}] is not found.".format(model_file_name))
        model = load_lua(model_file_name)
        if type(model).__name__=='hashable_uniq_dict':
            model = model.model
        model.evaluate()
        self.weight_loaded = True

        # Build network graph
        self.torch_graph = TorchGraph(model)
        self.torch_graph.build([[1] + list(map(int, input_shape))])


    def gen_IR(self):
        print ("OK")
        assert False
        print (self.torch_graph.model.childrens())
        for layer in self.src_graph.topological_sort:
            current_node = self.src_graph.get_node(layer)
            node_type = current_node.type

            if hasattr(self, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                func(current_node)

            else:
                self.rename_UNKNOWN(current_node)

    ##########
    # Layers #
    ##########
    def rename_UNKNOWN(self, source_node):
        print (source_node.layer)
        print (source_node.layer.data.size())
        assert False
        print("PyTorch parser has not supported operator [%s] with name [%s]."
              % (source_node.type, source_node.name))

    def rename_NoneType(self, source_node):
        assert source_node.name in self.src_graph.input_layers
        IR_node = self._convert_identity_operation(source_node, new_op="DataInput")
        for dim in self.input_shape:
            new_dim = IR_node.attr["shape"].shape.dim.add()
            if dim == None:
                new_dim.size = -1
            else:
                new_dim.size = dim


    def rename_ConvNd(self, source_node):
        kwargs = dict()
        kwargs['dilations'] = [1] + list(source_node.get_attr('dilation')) + [1]
        kwargs['pads'] = ([0] + list(source_node.get_attr('padding')) + [0]) * 2
        kwargs['strides'] = [1] + list(source_node.get_attr('stride')) + [1]
        kwargs['group'] = source_node.get_attr('groups')

        # handle weight
        weight = source_node.get_attr('next_functions')[1][0].variable.data.numpy()
        dim = weight.ndim - 2

        if source_node.get_attr('transposed'):
            IR_node = self._convert_identity_operation(source_node, new_op="ConvTranpose")
            weight = np.transpose(weight, list(range(2, dim + 2)) + [0, 1])
        else:
            IR_node = self._convert_identity_operation(source_node, new_op="Conv")
            weight = np.transpose(weight, list(range(2, dim + 2)) + [1, 0])

        self.set_weight(source_node.name, 'weights', weight)
        kwargs['kernel_shape'] = list(weight.shape)

        # handle bias
        if source_node.get_attr('next_functions')[2][0]:
            bias = source_node.get_attr('next_functions')[2][0].variable.data.numpy()
            self.set_weight(source_node.name, 'bias', weight)
            kwargs['use_bias'] = True
        else:
            kwargs['use_bias'] = False

        assign_IRnode_values(IR_node, kwargs)


    def rename_Threshold(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='Relu')


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

        print(IR_node)


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
        PyTorchParser._copy_and_reop(source_node, IR_node, new_op)
        self.convert_inedge(source_node, IR_node, 0, in_edge_count)
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
            kwargs['pooling_type'] = 'MAX'
        else:
            raise ValueError('Unknown pooling type')

        assign_IRnode_values(IR_node, kwargs)
