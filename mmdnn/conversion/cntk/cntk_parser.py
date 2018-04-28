#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import os
import numpy as np
from six.moves import xrange
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


    def _convert_identity_operation(self, source_node, start_edge=0, end_edge=None, new_op=None, shape_transpose=True):
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
            shape =  (-1,) + source_node.layer.shape
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


    @staticmethod
    def _print_layer(source_node):
        print ("Layer: ", source_node.layer)
        print ("Parameters: ", source_node.layer.parameters)
        print ("Attributes: ", source_node.layer.attributes)
        for in_node in source_node.layer.inputs:
            print (in_node)


    def rename_UNKNOWN(self, source_node):
        print("Cntk Parser has not supported operator [%s] with name [%s]."
              % (source_node.type, source_node.name))

        self._print_layer(source_node)
        assert False


    @staticmethod
    def get_ndarray(variable):
        if variable.is_parameter:
            return variable.as_parameter().asarray()

        elif variable.is_constant:
            return variable.as_constant().asarray()

        else:
            raise ValueError("Unknown variable [{}].".format(variable))


    @staticmethod
    def _get_attribute(source_node, attribute_name):
        if attribute_name in source_node.attributes:
            return source_node.attributes

        node = source_node.block_root
        while not attribute_name in node.attributes:
            node = node.inputs[0].owner

        return node.attributes


    def rename_Convolution(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op="Conv")

        for input in source_node.layer.inputs:
            if input.name.endswith("W"):
                W = self.get_ndarray(input)
                break

        W = self.channel_first_conv_kernel_to_IR(W)
        self.set_weight(source_node.name, 'weights', W)

        attributes = CntkParser._get_attribute(source_node.layer, 'strides')

        kwargs = dict()
        kwargs['strides'] = [1] + list(attributes['strides'])[1:] + [1]
        kwargs['dilations'] = [1] + list(attributes['dilation'])[1:] + [1]
        kwargs['kernel_shape'] = list(W.shape)
        padding = attributes['autoPadding'][1:]

        for pad in padding:
            assert pad == padding[0]

        kwargs['auto_pad'] = 'SAME_LOWER' if padding[0] else 'VALID'
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
            self._convert_binary_operator(source_node, new_op='Sub')


    def rename_Sub(self, source_node):
        self._convert_identity_operation(source_node)


    def rename_Reshape(self, source_node):
        IR_node = self._convert_identity_operation(source_node)
        new_shape = source_node.get_attr('newShape')
        kwargs = {'shape' : self.channel_first_shape_to_IR(new_shape)}
        assign_IRnode_values(IR_node, kwargs)


    def rename_Times(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='FullyConnected')

        W = source_node.layer.parameters[0].asarray().squeeze()
        self.set_weight(source_node.name, 'weights', W)

        kwargs = dict()
        kwargs['units'] = W.shape[-1]
        kwargs['use_bias'] = self._fuse_bias_node(source_node)
        assign_IRnode_values(IR_node, kwargs)


    def rename_MaxPooling(self, source_node):
        if source_node.layer.is_block:
            source_node.layer = source_node.layer.block_root.owner

        self.rename_Pooling(source_node)


    def rename_AveragePooling(self, source_node):
        self.rename_Pooling(source_node)


    def rename_Splice(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='Concat')
        assign_IRnode_values(IR_node, {'axis' : source_node.get_attr('axis')[-1] + 1})


    def rename_Pooling(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='Pool')
        dim = len(IR_node.attr['_output_shapes'].list.shape[0].dim)
        kwargs = {}

        # strides
        kwargs['strides'] = list(source_node.get_attr('strides')) + [1]
        if len(kwargs['strides']) < dim:
            kwargs['strides'] = [1] + kwargs['strides']

        # window_shape
        kwargs['kernel_shape'] = list(source_node.get_attr('poolingWindowShape')) + [1]
        if len(kwargs['kernel_shape']) < dim:
            kwargs['kernel_shape'] = [1] + kwargs['kernel_shape']

        # pool type
        pool_type = source_node.get_attr('poolingType')
        if pool_type == _cntk.MAX_POOLING:
            kwargs['pooling_type'] = 'MAX'
        elif pool_type == _cntk.AVG_POOLING:
            kwargs['pooling_type'] = 'AVG'
        else:
            raise ValueError("Unknown pooling type [{}].".format(pool_type))

        # padding
        padding = source_node.get_attr('autoPadding')
        if len(padding) >= dim - 1:
            padding = padding[1:]
        elif len(padding) < dim - 2:
            padding.extend([padding[-1]] * (dim - len(padding) - 2))
        for pad in padding:
            assert pad == padding[-1]
        kwargs['auto_pad'] = 'SAME_LOWER' if padding[0] else 'VALID'
        kwargs['pads'] = self._convert_padding_to_IR(kwargs['kernel_shape'][1:-1], padding)

        assign_IRnode_values(IR_node, kwargs)


    def rename_DataInput(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='DataInput')
        shape = [-1] + list(source_node.layer.shape)
        assign_IRnode_values(IR_node, {'shape' : list_to_shape(self.channel_first_shape_to_IR(shape))})


    def rename_BatchNormalization(self, source_node):
        kwargs = dict()
        kwargs['scale'] = False
        kwargs['bias'] = False
        for param in source_node.layer.inputs:
            if param.name.endswith('scale'):
                self.set_weight(source_node.name, 'scale', self.get_ndarray(param).flatten())
                kwargs['scale'] = True

            elif param.name.endswith('bias'):
                self.set_weight(source_node.name, 'bias', self.get_ndarray(param).flatten())
                kwargs['bias'] = True

            elif param.name.lower().endswith('mean'):
                self.set_weight(source_node.name, 'mean', self.get_ndarray(param).flatten())

            elif param.name.lower().endswith('variance'):
                self.set_weight(source_node.name, 'var', self.get_ndarray(param).flatten())

        IR_node = self._convert_identity_operation(source_node, end_edge=1, new_op='BatchNorm')
        kwargs['epsilon'] = source_node.get_attr('epsilon')
        kwargs['axis'] = -1
        assign_IRnode_values(IR_node, kwargs)


    def _add_constant_node(self, constant_node, IR_node):
        new_node = self.IR_graph.node.add()
        new_node.name = constant_node.uid
        new_node.op = 'Constant'
        value = np.atleast_1d(self.get_ndarray(constant_node))
        self.set_weight(new_node.name, 'value', value)
        IR_node.input.append(new_node.name)


    def _convert_binary_operator(self, source_node, new_op):
        IR_node = self._convert_identity_operation(source_node, new_op=new_op)
        for in_node in source_node.layer.inputs:
            if in_node.is_constant:
                self._add_constant_node(in_node, IR_node)


    def rename_ElementTimes(self, source_node):
        if source_node.layer.inputs[0] == source_node.layer.inputs[1]:
            # TODO: Handle square
            pass

        self._convert_binary_operator(source_node, 'Mul')


    def rename_Log(self, source_node):
        self._convert_identity_operation(source_node)


    def rename_Exp(self, source_node):
        self._convert_identity_operation(source_node)


    def rename_Reciprocal(self, source_node):
        self._convert_identity_operation(source_node)


    def rename_Dropout(self, source_node):
        # self._print_layer(source_node)
        # print (source_node.name)
        # print (self.src_graph.get_parent(source_node.name, [0]).real_name)
        # assert False
        source_node.real_name = self.src_graph.get_parent(source_node.name, [0]).real_name


    def rename_Dense(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='FullyConnected', shape_transpose=False)
        for param in source_node.layer.inputs:
            if param.name.endswith('W'):
                w = np.squeeze(self.get_ndarray(param))
                if w.ndim > 2:
                    w = np.transpose(w, list(range(1, w.ndim - 1)) + [0, -1])
                    w = np.reshape(w, [-1, w.shape[-1]])
                self.set_weight(source_node.name, 'weights', w)
                assign_IRnode_values(IR_node, {'units' : w.shape[-1] })

            elif param.name.endswith('b'):
                self.set_weight(source_node.name, 'bias', self.get_ndarray(param))
                assign_IRnode_values(IR_node, {'use_bias' : True })


    def rename_Convolution2D(self, source_node):
        assert source_node.layer.is_block

        # Convolution
        kwargs = dict()
        conv_IR_node = self.IR_graph.node.add()
        conv_node = source_node.layer.block_root.inputs[0].owner.inputs[0].owner

        conv_IR_node.name = conv_node.uid
        conv_IR_node.op = 'Conv'
        conv_IR_node.input.append(self.get_parent(source_node.name, [0]).real_name)

        # Kernel
        conv_weight = source_node.layer.block_root.inputs[0].owner.inputs[0].owner.inputs[0]
        conv_weight = self.get_ndarray(conv_weight)
        W = self.channel_first_conv_kernel_to_IR(conv_weight)
        self.set_weight(conv_IR_node.name, 'weights', W)

        # Attributes
        conv_attr = source_node.layer.block_root.inputs[0].owner.inputs[0].owner.attributes

        kwargs['strides'] = [1] + list(conv_attr['strides'])[1:] + [1]
        kwargs['dilations'] = [1] + list(conv_attr['dilation'])[1:] + [1]
        kwargs['kernel_shape'] = list(W.shape)
        padding = conv_attr['autoPadding'][1:]

        for pad in padding:
            assert pad == padding[0]

        kwargs['auto_pad'] = 'SAME_LOWER' if padding[0] else 'VALID'
        kwargs['pads'] = self._convert_padding_to_IR(kwargs['kernel_shape'][:-2], padding)

        kwargs['use_bias'] = True

        assign_IRnode_values(conv_IR_node, kwargs)

        # Bias
        plus = source_node.layer.block_root.inputs[0].owner.inputs[1]
        plus = np.squeeze(self.get_ndarray(plus))
        self.set_weight(conv_IR_node.name, 'bias', plus)

        # Activation
        activation = source_node.layer.block_root.owner.op_name

        activation_IR = self.IR_graph.node.add()
        activation_IR.name = source_node.name
        activation_IR.input.append(conv_IR_node.name)
        if (activation == 'ReLU'):
            activation_IR.op = 'Relu'
        else:
            raise ValueError()


    def rename_Activation(self, source_node):
        assert source_node.layer.is_block

        op = source_node.layer.root_function.owner.name

        if op.startswith('relu'):
            new_op= 'Relu'
        else:
            raise ValueError()

        self._convert_identity_operation(source_node, new_op=new_op)