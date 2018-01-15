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
        2 : graph_pb2.DT_FLOAT64,
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


    def _convert_reduction_operators(self, source_node, new_op = None):
        IR_node = self._convert_identity_operation(source_node, 1, new_op)

        # keep dims
        IR_node.attr['keepdims'].b = source_node.layer.attr['keep_dims'].b

        # axes
        axes = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor
        axes = tensor_util.MakeNdarray(axes)
        IR_node.attr['axes'].list.i.extend(axes)


    def _convert_layers_batchnorm(self, source_node):
        # name, op
        IR_node = self.IR_graph.node.add()
        CntkParser._copy_and_reop(source_node, IR_node, 'BatchNorm')

        # epsilon
        epsilon = self.get_parent(source_node.name, [1])
        IR_node.attr['epsilon'].f = epsilon.layer.attr['value'].tensor.float_val[0]

        # moving variance (var)
        moving_variance = self.get_parent(source_node.name, [0, 0])
        if self.weight_loaded and moving_variance.name in self.ckpt_data.keys():
            self.set_weight(source_node.name, 'var', self.ckpt_data[moving_variance.name])

        # gamma (scale)
        gamma = self.get_son(source_node.name, [0, 0], True)
        gamma = self.get_parent(gamma.name, [1, 0], True)
        if gamma is None or not gamma.type.startswith('Variable'):
            IR_node.attr['scale'].b = False
            output_node = self.get_son(source_node.name, [0, 0, 0], True)
        else:
            IR_node.attr['scale'].b = True
            if self.weight_loaded:
                self.set_weight(source_node.name, 'scale', self.ckpt_data[gamma.name])
            output_node = self.get_son(source_node.name, [0, 0, 0, 0], True)

        # mean
        mean = self.get_parent(output_node.name, [1, 1, 0, 0], True)
        if self.weight_loaded and mean.name in self.ckpt_data.keys():
            self.set_weight(source_node.name, 'mean', self.ckpt_data[mean.name])

        # bias
        bias = self.get_parent(output_node.name, [1, 0, 0], True)
        if bias is None or not bias.type.startswith('Variable'):
            IR_node.attr['bias'].b = False
        else:
            IR_node.attr['bias'].b = True
            if self.weight_loaded:
                self.set_weight(source_node.name, 'bias', self.ckpt_data[bias.name])

        # input node
        assert output_node.type == 'Add'
        input_node = self.get_parent(output_node.name, [0, 0])
        IR_node.input.append(input_node.real_name)

        # output node
        output_node.real_name = source_node.name


    def __init__(self, model, dest_nodes = None):
        super(CntkParser, self).__init__()

        if not os.path.exists(model):
            raise ValueError('Cntk model [{}] can not be found!'.format(model))
        model = _cntk.Function.load(model)

        # Build network graph
        self.cntk_graph =  CntkGraph(model)
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


    def gen_IR(self):
        for layer in self.src_graph.topological_sort:
            current_node = self.src_graph.get_node(layer)
            node_type = current_node.type
            print (current_node.name, node_type)
            if hasattr(self, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                func(current_node)
            else:
                self.rename_UNKNOWN(current_node)


    @staticmethod
    def _copy_and_reop(source_node, IR_node, new_op = None):
        if new_op == None: new_op = source_node.type
        IR_node.name = source_node.real_name
        IR_node.op = new_op

        kwargs = {}
        # if 'data_format' in source_node.layer.attr:
        #     kwargs['data_format'] = source_node.get_attr('data_format')

        if hasattr(source_node.layer, 'dtype'):
            assert source_node.layer.dtype in CntkParser.dtype_map, 'type [{}] is unknown.'.format(source_node.layer.dtype)
            IR_node.attr["dtype"].type = CntkParser.dtype_map[source_node.layer.dtype]

        if hasattr(source_node.layer, 'shape'):
            shape =  (-1,) + source_node.layer.shape
            shape = list_to_shape(shape)
            kwargs['_output_shapes'] = [shape]

        assign_IRnode_values(IR_node, kwargs)


    def rename_UNKNOWN(self, source_node):
        print("Cntk Parser has not supported operator [%s] with name [%s]."
              % (source_node.type, source_node.name))

        print (source_node.layer)
        print (source_node.layer.parameters)
        print (source_node.attributes)
        # for input in source_node.layer.inputs:
        #     print (input)
        #     print (dir(input))

        assert False


    def rename_Convolution(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op="Conv")

        W = source_node.layer.parameters[0].asarray()
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

        kwargs['use_bias'] = False

        assign_IRnode_values(IR_node, kwargs)


    def _convert_identity_operation(self, source_node, in_edge_count = None, new_op = None):
        IR_node = self.IR_graph.node.add()
        CntkParser._copy_and_reop(source_node, IR_node, new_op)
        self.convert_inedge(source_node, IR_node, 0, in_edge_count)
        return IR_node


    def rename_ReLU(self, source_node):
        self._convert_identity_operation(source_node, new_op='Relu')


    def rename_Relu6(self, source_node):
        self._convert_identity_operation(source_node)


    def rename_Plus(self, source_node):
        if not source_node.covered:
            assert not source_node.layer.parameters
            IR_node = self._convert_identity_operation(source_node, new_op='Add')


    def rename_Sub(self, source_node):
        self._convert_identity_operation(source_node)


    def rename_Reshape(self, source_node):
        IR_node = self._convert_identity_operation(source_node, 1)
        kwargs = {'shape' : self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]}
        assign_IRnode_values(IR_node, kwargs)


    def rename_Times(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='FullyConnected')

        W = source_node.layer.parameters[0].asarray().squeeze()
        self.set_weight(source_node.name, 'weights', W)

        kwargs = dict()
        kwargs['unit'] = W.shape[-1]
        assign_IRnode_values(IR_node, kwargs)

        next_node = self.src_graph.get_son(source_node.name, [0])
        if next_node.type == 'Plus':
            next_node.covered = True
            next_node.real_name = source_node.name
            kwargs['use_bias'] = True
            B = next_node.layer.parameters[0].asarray()
            self.set_weight(source_node.name, 'bias', B)

        else:
            kwargs['use_bias'] = False


    # def rename_MatMul(self, source_node):
    #     """
    #     weights: name_weights, name_bias
    #     """
    #     IR_node = self._convert_identity_operation(source_node, 1)

    #     # units
    #     units = source_node.layer.attr['_output_shapes'].list.shape[-1].dim[-1].size
    #     IR_node.attr['units'].i = units

    #     # Weights
    #     W = self.tf_graph.get_node(self.tf_graph.get_node(source_node.in_edges[1]).in_edges[0])
    #     if self.weight_loaded:
    #         self.set_weight(source_node.name, 'weights', self.ckpt_data[W.name])

    #     if source_node.out_edges:
    #         add_node = self.tf_graph.get_node(source_node.out_edges[0])
    #         if add_node.type == 'Add':
    #             add_node.covered = True
    #             add_node.real_name = source_node.real_name
    #             # FullyConnected Layer
    #             # name, op
    #             CntkParser._copy_and_reop(source_node, IR_node, 'FullyConnected')

    #             # get Bias
    #             B = self.tf_graph.get_node(self.tf_graph.get_node(source_node.out_edges[0]).in_edges[1]).in_edges[0]
    #             if self.weight_loaded:
    #                 self.set_weight(source_node.name, 'bias', self.ckpt_data[B])
    #             IR_node.attr['use_bias'].b = True

    #         else:
    #             raise NotImplementedError("Not implemented yet. Please submit a issue in github and provide your models for reproduce.")

    #     else:
    #         # Matmul Layer
    #         CntkParser._copy_and_reop(source_node, IR_node, 'FullyConnected')
    #         assign_IRnode_values(IR_node, {'use_bias' : False})


    def rename_Pooling(self, source_node):
        IR_node = self._convert_identity_operation(source_node)
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


    # def rename_Identity(self, source_node):
    #     source_node.real_name =  self.src_graph.get_node(source_node.in_edges[0]).real_name


    # def rename_Squeeze(self, source_node):
    #     IR_node = self._convert_identity_operation(source_node)
    #     IR_node.attr['axes'].MergeFromString(source_node.layer.attr['squeeze_dims'].SerializeToString())


    def rename_DataInput(self, source_node):
        IR_node = self._convert_identity_operation(source_node, 0, 'DataInput')
        IR_node.attr['shape'].shape.MergeFromString(IR_node.attr['_output_shapes'].list.shape[0].SerializeToString())


    # def rename_Pad(self, source_node):
    #     IR_node = self._convert_identity_operation(source_node, 1, 'Pad')
    #     kwargs = {}
    #     kwargs['mode'] = 'constant'
    #     kwargs['constant_values'] = 0.0

    #     # paddings
    #     padding = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor
    #     shapes = tensor_util.MakeNdarray(padding)
    #     kwargs['pads'] = convert_tf_pad_to_onnx(shapes)

    #     assign_IRnode_values(IR_node, kwargs)


    # def rename_Mean(self, source_node):
    #     self._convert_reduction_operators(source_node, new_op = 'ReduceMean')


    # def rename_ConcatV2(self, source_node):
    #     n = len(source_node.in_edges) - 1
    #     IR_node = self._convert_identity_operation(source_node, n, 'Concat')
    #     axis = self.tf_graph.get_parent(source_node.name, [n])
    #     IR_node.attr['axis'].i = axis.layer.attr['value'].tensor.int_val[0]


    # def rename_DepthwiseConv2dNative(self, source_node):
    #     IR_node = self._convert_identity_operation(source_node, 1, 'DepthwiseConv')
    #     kwargs = {}
    #     kwargs['strides'] = source_node.get_attr('strides')

    #     input_node = self.src_graph.get_parent(source_node.name, [1])
    #     kwargs['kernel_shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

    #     self._convert_padding(source_node, IR_node, kwargs['kernel_shape'][:-2])

    #     if self.weight_loaded:
    #         weight = self.src_graph.get_parent(source_node.name, [1, 0])
    #         self.set_weight(source_node.name, 'weights', self.ckpt_data[weight.name])

    #     assign_IRnode_values(IR_node, kwargs)


    def rename_BatchNormalization(self, source_node):
        for param in source_node.layer.parameters:
            if param.name.endswith('scale'):
                self.set_weight(source_node.name, 'scale', param.asarray())

            elif param.name.endswith('bias'):
                self.set_weight(source_node.name, 'bias', param.asarray())

            else:
                raise ValueError("Unknown BN layer parameter [{}].".format(param.name))

        IR_node = self._convert_identity_operation(source_node, 1, 'BatchNorm')

        kwargs = dict()
        kwargs['epsilon'] = source_node.get_attr('epsilon')
        kwargs['axis'] = 1

        assign_IRnode_values(IR_node, kwargs)


    # def rename_Transpose(self, source_node):
    #     IR_node = self._convert_identity_operation(source_node, 1)
    #     perm = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor
    #     perm = tensor_util.MakeNdarray(perm).tolist()
    #     assign_IRnode_values(IR_node, {'perm' : perm})

    # def rename_Sigmoid(self, source_node):
    #     IR_node = self._convert_identity_operation(source_node)


    # def rename_Mul(self, source_node):
    #     IR_node = self._convert_identity_operation(source_node)
