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
    def tensor_shape_to_list(shapes):
        if isinstance(shapes, attr_value_pb2.AttrValue):
            return [dim.size for dim in shapes.shape.dim]

        else:
            ret = []
            for shape in shapes:
                this_one = [dim.size for dim in shape.dim]
                ret.append(this_one)
            return ret


    def _convert_padding(self, source_node, IR_node, kernel_size):
        # TODO: Fused conv and pool with padding is different from defused operators
        input_node = self.get_parent(source_node.name, [0])
        input_shape = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        if source_node.get_attr('padding') == 'VALID':
            dims = len(input_shape)
            assign_IRnode_values(IR_node, {'auto_pad' : "VALID", 'pads' : [0, 0] * dims})

        elif source_node.get_attr('padding') == 'SAME':
            padding = compute_tf_same_padding(
                input_shape,
                kernel_size,
                source_node.get_attr('strides'))
            assign_IRnode_values(IR_node, {'auto_pad' : "SAME_LOWER", 'pads' : padding})

        else:
            assert False


    def _convert_pooling(self, source_node, pool_type):
        IR_node = self._convert_identity_operation(source_node, new_op='Pool')
        kwargs = {}

        # strides
        kwargs['strides'] = source_node.get_attr('strides')

        # window_shape
        kwargs['kernel_shape'] = source_node.get_attr('ksize')

        # pool type
        kwargs['pooling_type'] = pool_type

        # padding
        self._convert_padding(source_node, IR_node, kwargs['kernel_shape'][1:-1])

        assign_IRnode_values(IR_node, kwargs)


    def gen_IR(self):
        for layer in self.src_graph.topological_sort:
            current_node = self.src_graph.get_node(layer)
            node_type = current_node.type
            # print (current_node.name, node_type)
            # if hasattr(self, "rename_" + node_type):
            #     func = getattr(self, "rename_" + node_type)
            #     func(current_node)
            # else:
            self.rename_UNKNOWN(current_node)


    @staticmethod
    def _copy_and_reop(source_node, IR_node, new_op = None):
        if new_op == None: new_op = source_node.type
        IR_node.name = source_node.name
        IR_node.op = new_op

        kwargs = {}
        if 'data_format' in source_node.layer.attr:
            kwargs['data_format'] = source_node.get_attr('data_format')

        if 'dtype' in source_node.layer.attr:
            assert source_node.layer.attr['dtype'].type in CntkParser.dtype_map, 'type [{}] is unknown.'.format(source_node.layer.attr['dtype'].type)
            IR_node.attr["dtype"].type = CntkParser.dtype_map[source_node.layer.attr['dtype'].type]

        if '_output_shapes' in source_node.layer.attr:
            IR_node.attr["_output_shapes"].MergeFromString(source_node.layer.attr['_output_shapes'].SerializeToString())

        assign_IRnode_values(IR_node, kwargs)


    @staticmethod
    def _copy_shape(source_node, IR_node):
        assert 'shape' in source_node.layer.attr
        IR_node.attr['shape'].shape.MergeFromString(source_node.layer.attr['shape'].shape.SerializeToString())


    def rename_UNKNOWN(self, source_node):
        print("Cntk Parser has not supported operator [%s] with name [%s]."
              % (source_node.type, source_node.name))
        print (source_node.layer)
        for input in source_node.layer.inputs:
            print (input)
            print (dir(input))

        assert False


    def rename_Conv2D(self, source_node):
        """
        weights: name_weights, name_bias
        """
        IR_node = self._convert_identity_operation(source_node, 1, 'Conv')

        kwargs = {}

        # strides
        kwargs['strides'] = source_node.get_attr('strides')

        # input[1] : W
        # filter
        W = self.tf_graph.get_node(source_node.layer.input[1])
        W = self.tf_graph.get_node(W.layer.input[0]).layer
        kwargs['kernel_shape'] = self.tensor_shape_to_list(W.attr['shape'])

        # padding
        self._convert_padding(source_node, IR_node, kwargs['kernel_shape'][:-2])

        if self.weight_loaded:
            self.set_weight(source_node.name, 'weights', self.ckpt_data[W.name])

        assign_IRnode_values(IR_node, kwargs)

        # output[0] : B
        self._get_bias(source_node, IR_node)


    def _convert_identity_operation(self, source_node, in_edge_count = None, new_op = None):
        IR_node = self.IR_graph.node.add()
        CntkParser._copy_and_reop(source_node, IR_node, new_op)
        self._convert_inedge(source_node, IR_node, 0, in_edge_count)
        return IR_node


    def rename_Relu(self, source_node):
        self._convert_identity_operation(source_node)


    def rename_Relu6(self, source_node):
        self._convert_identity_operation(source_node)


    def rename_Add(self, source_node):
        if not source_node.covered:
            scopes = self._get_scopes(source_node.name)
            if len(scopes) < 3:
                self._convert_identity_operation(source_node)

            elif scopes[-2] == 'dropout':
                # converted [dropout]
                pass

            elif scopes[-2] == 'batchnorm':
                # convert [tf.contrib.layers.batch_norm]
                self._convert_layers_batchnorm(source_node)

            else:
                # normal Add
                self._convert_identity_operation(source_node)


    def rename_Sub(self, source_node):
        self._convert_identity_operation(source_node)


    def rename_Reshape(self, source_node):
        IR_node = self._convert_identity_operation(source_node, 1)
        kwargs = {'shape' : self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]}
        assign_IRnode_values(IR_node, kwargs)


    def rename_MatMul(self, source_node):
        """
        weights: name_weights, name_bias
        """
        IR_node = self._convert_identity_operation(source_node, 1)

        # units
        units = source_node.layer.attr['_output_shapes'].list.shape[-1].dim[-1].size
        IR_node.attr['units'].i = units

        # Weights
        W = self.tf_graph.get_node(self.tf_graph.get_node(source_node.in_edges[1]).in_edges[0])
        if self.weight_loaded:
            self.set_weight(source_node.name, 'weights', self.ckpt_data[W.name])

        if source_node.out_edges:
            add_node = self.tf_graph.get_node(source_node.out_edges[0])
            if add_node.type == 'Add':
                add_node.covered = True
                add_node.real_name = source_node.real_name
                # FullyConnected Layer
                # name, op
                CntkParser._copy_and_reop(source_node, IR_node, 'FullyConnected')

                # get Bias
                B = self.tf_graph.get_node(self.tf_graph.get_node(source_node.out_edges[0]).in_edges[1]).in_edges[0]
                if self.weight_loaded:
                    self.set_weight(source_node.name, 'bias', self.ckpt_data[B])
                IR_node.attr['use_bias'].b = True

            else:
                raise NotImplementedError("Not implemented yet. Please submit a issue in github and provide your models for reproduce.")

        else:
            # Matmul Layer
            CntkParser._copy_and_reop(source_node, IR_node, 'FullyConnected')
            assign_IRnode_values(IR_node, {'use_bias' : False})


    def rename_RealDiv(self, source_node):
        scopes = self._get_scopes(source_node.name)

        # Deal Dropout
        if scopes[-2] == 'dropout':
            IR_node = self._convert_identity_operation(source_node, 1, 'Dropout')

            # keep prob
            if 'value' in self.tf_graph.get_node(source_node.layer.input[1]).layer.attr:
                IR_node.attr['keep_prob'].f = self.tf_graph.get_node(source_node.layer.input[1]).layer.attr['value'].tensor.float_val[0]
            else:
                IR_node.attr['keep_prob'].f = 1.0

            # Remove nodes
            # Mul
            mul_node = self.tf_graph.get_node(source_node.out_edges[0])
            assert mul_node.type == "Mul"
            mul_node.covered = True
            mul_node.real_name = source_node.name

            # Floor
            floor_node = self.tf_graph.get_node(mul_node.in_edges[1])
            assert floor_node.type == "Floor"
            floor_node.covered = True
        else:
            assert False


    def rename_Floor(self, source_node):
        scopes = self._get_scopes(source_node.name)
        assert scopes[-2] == 'dropout'


    def rename_MaxPool(self, source_node):
        self._convert_pooling(source_node, b'MAX')


    def rename_AvgPool(self, source_node):
        self._convert_pooling(source_node, b'AVG')


    def rename_Identity(self, source_node):
        source_node.real_name =  self.src_graph.get_node(source_node.in_edges[0]).real_name


    def rename_Squeeze(self, source_node):
        IR_node = self._convert_identity_operation(source_node)
        IR_node.attr['axes'].MergeFromString(source_node.layer.attr['squeeze_dims'].SerializeToString())


    def rename_QueueDequeueManyV2(self, source_node):
        IR_node = self._convert_identity_operation(source_node, 0, 'DataInput')
        IR_node.attr['shape'].shape.MergeFromString(source_node.layer.attr['_output_shapes'].list.shape[0].SerializeToString())
        IR_node.attr['shape'].shape.dim[0].size = -1
        IR_node.attr['dtype'].type = self.dtype_map[source_node.layer.attr['component_types'].list.type[0]]


    def rename_Pad(self, source_node):
        IR_node = self._convert_identity_operation(source_node, 1, 'Pad')
        kwargs = {}
        kwargs['mode'] = 'constant'
        kwargs['constant_values'] = 0.0

        # paddings
        padding = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor
        shapes = tensor_util.MakeNdarray(padding)
        kwargs['pads'] = convert_tf_pad_to_onnx(shapes)

        assign_IRnode_values(IR_node, kwargs)


    def rename_Mean(self, source_node):
        self._convert_reduction_operators(source_node, new_op = 'ReduceMean')


    def rename_ConcatV2(self, source_node):
        n = len(source_node.in_edges) - 1
        IR_node = self._convert_identity_operation(source_node, n, 'Concat')
        axis = self.tf_graph.get_parent(source_node.name, [n])
        IR_node.attr['axis'].i = axis.layer.attr['value'].tensor.int_val[0]


    def rename_DepthwiseConv2dNative(self, source_node):
        IR_node = self._convert_identity_operation(source_node, 1, 'DepthwiseConv')
        kwargs = {}
        kwargs['strides'] = source_node.get_attr('strides')

        input_node = self.src_graph.get_parent(source_node.name, [1])
        kwargs['kernel_shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        self._convert_padding(source_node, IR_node, kwargs['kernel_shape'][:-2])

        if self.weight_loaded:
            weight = self.src_graph.get_parent(source_node.name, [1, 0])
            self.set_weight(source_node.name, 'weights', self.ckpt_data[weight.name])

        assign_IRnode_values(IR_node, kwargs)


    def rename_FusedBatchNorm(self, source_node):
        IR_node = self._convert_identity_operation(source_node, 1, 'BatchNorm')
        IR_node.attr['epsilon'].f = source_node.get_attr('epsilon', 0)

        # gamma (scale)
        scale = self.get_parent(source_node.name, [1], True)

        if scale.type == 'Const':
            value = scale.get_attr('value')
            shape = value.tensor_shape
            assert len(shape.dim) == 1
            shape = shape.dim[0].size

            assert len(value.float_val) == 1
            value = value.float_val[0]

            if np.isclose(value, 1.0):
                IR_node.attr['scale'].b = False
            else:
                IR_node.attr['scale'].b = True
                if self.weight_loaded:
                    self.set_weight(source_node.name, 'scale', np.array([value] * shape))

        else:
            scale = self.get_parent(scale.name, [0], True)
            if self.weight_loaded:
                self.set_weight(source_node.name, 'scale', self.ckpt_data[scale.name])
            IR_node.attr['scale'].b = True

        # bias
        bias = self.get_parent(source_node.name, [2, 0], True)
        IR_node.attr['bias'].b = True

        # Mean
        mean = self.get_parent(source_node.name, [3, 0], True)

        # Var
        var = self.get_parent(source_node.name, [4, 0], True)

        if self.weight_loaded:
            self.set_weight(source_node.name, 'bias', self.ckpt_data[bias.name])
            self.set_weight(source_node.name, 'mean', self.ckpt_data[mean.name])
            self.set_weight(source_node.name, 'var', self.ckpt_data[var.name])


    def rename_Transpose(self, source_node):
        IR_node = self._convert_identity_operation(source_node, 1)
        perm = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor
        perm = tensor_util.MakeNdarray(perm).tolist()
        assign_IRnode_values(IR_node, {'perm' : perm})

    def rename_Sigmoid(self, source_node):
        IR_node = self._convert_identity_operation(source_node)


    def rename_Mul(self, source_node):
        IR_node = self._convert_identity_operation(source_node)
