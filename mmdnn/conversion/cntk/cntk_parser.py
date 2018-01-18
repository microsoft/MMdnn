#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import os
import numpy as np
import cntk as _cntk
from mmdnn.conversion.cntk.cntk_graph import CntkGraph
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.utils import *
from mmdnn.conversion.common.DataStructure.parser import Parser


class CntkParser(Parser):

    dtype_map = {
        0 : graph_pb2.DT_UNDEFINED,
        np.float32 : graph_pb2.DT_FLOAT32,
        np.float64 : graph_pb2.DT_FLOAT64,
        3 : graph_pb2.DT_INT32,
        4 : graph_pb2.DT_UINT8,
        5 : graph_pb2.DT_INT16,
        6 : graph_pb2.DT_INT8,
        7 : graph_pb2.DT_STRING,
        9 : graph_pb2.DT_INT64
    }


    @property
    def src_graph(self):
        return self.cntk_graph


    def __init__(self, model, dest_nodes = None):
        super(CntkParser, self).__init__()

        if not os.path.exists(model):
            raise ValueError('Cntk model [{}] can not be found!'.format(model))
        model = _cntk.Function.load(model)
        self.weight_loaded = True

        # Build network graph
        self.cntk_graph = CntkGraph(model)
        self.cntk_graph.build()


    @staticmethod
    def _convert_padding_to_IR(kernel_shape, auto_pad):
        lower = []
        upper = []
        for idx in range(0, len(kernel_shape)):
            if auto_pad[idx] == False:
                lower += [0]
                upper += [0]
            else:
                q = kernel_shape[idx] // 2
                lower += [q] if kernel_shape[idx] % 2 else [q - 1]
                upper += [q]

        return [0] + lower + [0, 0] + upper + [0]


    def _convert_identity_operation(self, source_node, start_edge=0, end_edge=None, new_op=None, shape_transpose=False):
        IR_node = self.IR_graph.node.add()
        CntkParser._copy_and_reop(source_node, IR_node, new_op, shape_transpose)
        self.convert_inedge(source_node, IR_node, start_edge, end_edge)
        return IR_node


    def gen_IR(self):
        for layer in self.src_graph.topological_sort:
            current_node = self.src_graph.get_node(layer)
            node_type = current_node.type
            if hasattr(self, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                func(current_node)
            else:
                self.rename_UNKNOWN(current_node)


    @staticmethod
    def _copy_and_reop(source_node, IR_node, new_op=None, shape_transpose=False):
        if new_op == None: new_op = source_node.type
        IR_node.name = source_node.real_name
        IR_node.op = new_op

        kwargs = {}

        if hasattr(source_node.layer, 'dtype'):
            assert source_node.layer.dtype in CntkParser.dtype_map, 'type [{}] is unknown.'.format(source_node.layer.dtype)
            IR_node.attr["dtype"].type = CntkParser.dtype_map[source_node.layer.dtype]

        if hasattr(source_node.layer, 'shape'):
            shape =  (-1,) + source_node.layer.shape[1:]
            if shape_transpose:
                shape = CntkParser.channel_first_shape_to_IR(shape)
            shape = list_to_shape(shape)
            kwargs['_output_shapes'] = [shape]

        assign_IRnode_values(IR_node, kwargs)


    def _fuse_bias_node(self, source_node):
        next_node = self.src_graph.get_son(source_node.name, [0])
        if next_node is None or next_node.type != 'Plus' or not next_node.layer.parameters:
            return False

        next_node.covered = True
        next_node.real_name = source_node.real_name
        B = next_node.layer.parameters[0].asarray()
        self.set_weight(source_node.name, 'bias', B)

        return True


    def rename_UNKNOWN(self, source_node):
        print("Cntk Parser has not supported operator [%s] with name [%s]."
              % (source_node.type, source_node.name))

        print (source_node.layer)
        print (source_node.layer.parameters)
        print (source_node.layer.attributes)
        print (source_node.layer.inputs)

        assert False


    @staticmethod
    def get_ndarray(variable):
        if variable.is_parameter:
            return variable.as_parameter().asarray()

        elif variable.is_constant:
            return variable.as_constant().asarray()

        else:
            raise ValueError("Unknown variable [{}].".format(variable))


    def rename_Convolution(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op="Conv")

        for input in source_node.layer.inputs:
            if input.name.endswith("W"):
                W = self.get_ndarray(input)
                break

        W = self.channel_first_conv_kernel_to_IR(W)
        self.set_weight(source_node.name, 'weights', W)

        kwargs = dict()

        kwargs['strides'] = [1] + list(source_node.get_attr('strides'))[1:] + [1]
        kwargs['dilations'] = [1] + list(source_node.get_attr('dilation'))[1:] + [1]
        kwargs['kernel_shape'] = list(W.shape)
        padding = source_node.get_attr('autoPadding')[1:]

        for pad in padding:
            assert pad == padding[0]

        kwargs['auto_pad'] = 'SAME_UPPER' if padding[0] else 'VALID'
        kwargs['pads'] = self._convert_padding_to_IR(kwargs['kernel_shape'][:-2], padding)

        kwargs['use_bias'] = self._fuse_bias_node(source_node)

        assign_IRnode_values(IR_node, kwargs)


    def rename_ReLU(self, source_node):
        self._convert_identity_operation(source_node, new_op='Relu')


    def rename_Relu6(self, source_node):
        self._convert_identity_operation(source_node)


    def rename_Plus(self, source_node):
        if not source_node.covered:
            assert not source_node.layer.parameters
            IR_node = self._convert_identity_operation(source_node, new_op='Add')


    def rename_Minus(self, source_node):
        if not source_node.covered:
            assert not source_node.layer.parameters
            IR_node = self._convert_identity_operation(source_node, new_op='Sub')


    def rename_Sub(self, source_node):
        self._convert_identity_operation(source_node)


    def rename_Reshape(self, source_node):
        IR_node = self._convert_identity_operation(source_node, shape_transpose=True)
        new_shape = source_node.get_attr('newShape')
        kwargs = {'shape' : self.channel_first_shape_to_IR(new_shape)}
        assign_IRnode_values(IR_node, kwargs)


    def rename_Times(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='FullyConnected')

        W = source_node.layer.parameters[0].asarray().squeeze()
        self.set_weight(source_node.name, 'weights', W)

        kwargs = dict()
        kwargs['unit'] = W.shape[-1]
        assign_IRnode_values(IR_node, kwargs)

        kwargs['use_bias'] = self._fuse_bias_node(source_node)


    def rename_MaxPooling(self, source_node):
        self.rename_Pooling(source_node)


    def rename_AveragePooling(self, source_node):
        self.rename_Pooling(source_node)


    def rename_Splice(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='Concat', shape_transpose=True)
        assign_IRnode_values(IR_node, {'axis' : source_node.get_attr('axis')[-1]})


    def rename_Pooling(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='Pool')
        kwargs = {}

        # strides
        kwargs['strides'] = [1] + list(source_node.get_attr('strides'))[1:] + [1]

        # window_shape
        kwargs['kernel_shape'] = [1] + list(source_node.get_attr('poolingWindowShape'))[1:] + [1]

        # pool type
        pool_type = source_node.get_attr('poolingType')
        if pool_type == _cntk.MAX_POOLING:
            kwargs['pooling_type'] = 'MAX'
        elif pool_type == _cntk.AVG_POOLING:
            kwargs['pooling_type'] = 'AVG'
        else:
            raise ValueError("Unknown pooling type [{}].".format(pool_type))

        # padding
        padding = source_node.get_attr('autoPadding')[1:]
        for pad in padding:
            assert pad == padding[0]
        kwargs['auto_pad'] = 'SAME_UPPER' if padding[0] else 'VALID'
        kwargs['pads'] = self._convert_padding_to_IR(kwargs['kernel_shape'][1:-1], padding)

        assign_IRnode_values(IR_node, kwargs)


    def rename_DataInput(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='DataInput')
        shape = [-1] + list(source_node.layer.shape)
        assign_IRnode_values(IR_node, {'shape' : list_to_shape(self.channel_first_shape_to_IR(shape))})


    def rename_BatchNormalization(self, source_node):
        for param in source_node.layer.inputs:
            if param.name.endswith('scale'):
                self.set_weight(source_node.name, 'scale', self.get_ndarray(param))

            elif param.name.endswith('bias'):
                self.set_weight(source_node.name, 'bias', self.get_ndarray(param))

            elif param.name.endswith('Mean'):
                self.set_weight(source_node.name, 'mean', self.get_ndarray(param))

            elif param.name.endswith('Variance'):
                self.set_weight(source_node.name, 'var', self.get_ndarray(param))

        IR_node = self._convert_identity_operation(source_node, end_edge=1, new_op='BatchNorm')

        kwargs = dict()
        kwargs['epsilon'] = source_node.get_attr('epsilon')
        kwargs['axis'] = -1

        assign_IRnode_values(IR_node, kwargs)


    def rename_ElementTimes(self, source_node):
        self._convert_identity_operation(source_node, new_op='Mul')


    def rename_Log(self, source_node):
        self._convert_identity_operation(source_node)


    def rename_Exp(self, source_node):
        self._convert_identity_operation(source_node)


    def rename_Reciprocal(self, source_node):
        self._convert_identity_operation(source_node)

    def rename_Dropout(self, source_node):
        source_node.real_name = self.src_graph.get_parent(source_node.name, [0]).real_name

    def rename_Dense(self, source_node):
        for param in source_node.layer.inputs:
            if param.name.endswith('W'):
                self.set_weight(source_node.name, 'weights', self.get_ndarray(param))

            elif param.name.endswith('b'):
                self.set_weight(source_node.name, 'bias', self.get_ndarray(param))

        IR_node = self._convert_identity_operation(source_node, new_op='FullyConnected')
