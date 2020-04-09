from __future__ import absolute_import
from __future__ import division
import numpy as np

from mmdnn.conversion.caffe.errors import ConversionError
from mmdnn.conversion.caffe.common_graph import Node
from mmdnn.conversion.caffe.network import DEFAULT_PADDING
from mmdnn.conversion.caffe.utils import get_lower_case
from mmdnn.conversion.common.IR.graph_pb2 import TensorShape


def get_handler_name(node_kind):
    if node_kind is None:
        return node_kind
    else:
        if len(node_kind) <= 4:
            return node_kind.lower()
        else:
            return get_lower_case(node_kind)


class NodeMapper(object):

    @classmethod
    def _convert_output_shape(cls, kwargs, node):
        shape = TensorShape()
        dim = shape.dim.add()
        dim.size = -1

        if len(node.output_shape) > 2:
            for i in node.output_shape[2:]:
                dim = shape.dim.add()
                dim.size = i
            dim = shape.dim.add()
            dim.size = node.output_shape.channels
        else:
            dim = shape.dim.add()
            dim.size = node.output_shape[1]
        kwargs['_output_shapes'] = [shape]

    @classmethod
    def get_kernel_params(cls, node, input_shape):
        kwargs = {}

        if node.kernel_parameters.global_pooling:
            kwargs['kernel_shape'] = [1, input_shape.height, input_shape.width, 1]
            kwargs['pads'] = [0] * 8

        else:
            from mmdnn.conversion.caffe.graph import NodeKind
            if node.kind == NodeKind.Pooling:
                kwargs['kernel_shape'] = [1, node.kernel_parameters.k_h, node.kernel_parameters.k_w, 1]
            elif node.kind in [NodeKind.Convolution, NodeKind.Deconvolution]:
                pass
            else:
                raise ValueError

            dilation = node.parameters.dilation[0] if hasattr(node.parameters, 'dilation') and node.parameters.dilation else 1
            o_h_caffe = node.output_shape.height
            o_w_caffe = node.output_shape.width
            ko_h = dilation * (int(node.kernel_parameters.k_h) - 1) + 1
            ko_w = dilation * (int(node.kernel_parameters.k_w) - 1) + 1

            if node.kind == NodeKind.Deconvolution:
                o_h_tf = int(node.kernel_parameters.s_h) * (input_shape.height - 1) + ko_h - 2 * int(node.kernel_parameters.p_h)
                o_w_tf = int(node.kernel_parameters.s_w) * (input_shape.width - 1) + ko_w - 2 * int(node.kernel_parameters.p_w)
            else:
                o_h_tf = (input_shape.height + node.kernel_parameters.p_h * 2 - ko_h + 1) // node.kernel_parameters.s_h
                o_w_tf = (input_shape.width + node.kernel_parameters.p_w * 2 - ko_w + 1) // node.kernel_parameters.s_w
            
            kwargs['pads'] = [0, node.kernel_parameters.p_h, node.kernel_parameters.p_w, 0] + \
                    [0, node.kernel_parameters.p_h + o_h_caffe - o_h_tf, node.kernel_parameters.p_w + o_w_caffe - o_w_tf, 0]

        kwargs['strides'] = [1, node.kernel_parameters.s_h, node.kernel_parameters.s_w, 1]
        cls._convert_output_shape(kwargs, node)

        return kwargs


    @classmethod
    def map_data(cls, node):
        # TODO: We need to identify whether this is 4D image data, otherwise we shouldn't change the dimension order
        shape = TensorShape()
        dim = shape.dim.add()
        dim.size = -1
        for i in node.output_shape[2:]:
            dim = shape.dim.add()
            dim.size = i
        dim = shape.dim.add()
        dim.size = node.output_shape.channels

        kwargs = {'shape': shape} # Ignore the dimension of batch size
        cls._convert_output_shape(kwargs, node)
        return Node.create('DataInput', **kwargs)


    @classmethod
    def map_input(cls, node):
        return cls.map_data(node)

    @classmethod
    def map_convolution(cls, node):
        parent, _ = node.get_only_parent()
        kwargs = cls.get_kernel_params(node, parent.output_shape)
        kwargs['kernel_shape'] = [node.kernel_parameters.k_h, node.kernel_parameters.k_w, parent.output_shape.channels, node.parameters.num_output]
        kwargs['use_bias'] = node.parameters.bias_term
        if node.parameters.dilation:
            dilation = node.parameters.dilation[0]
            if dilation != 1:
                kwargs['dilations'] = [1, dilation, dilation, 1]
        kwargs['group'] = node.parameters.group
        return Node.create('Conv', **kwargs)


    @classmethod
    def map_deconvolution(cls, node):
        parent, _ = node.get_only_parent()
        kwargs = cls.get_kernel_params(node, parent.output_shape)

        kwargs['kernel_shape'] = [node.kernel_parameters.k_h, node.kernel_parameters.k_w, node.parameters.num_output, parent.output_shape.channels]
        kwargs['use_bias'] = node.parameters.bias_term
        if node.parameters.dilation:
            dilation = node.parameters.dilation[0]
            if dilation != 1:
                kwargs['dilations'] = [1, dilation, dilation, 1]
        kwargs['group'] = node.parameters.group
        return Node.create('ConvTranspose', **kwargs)


    @classmethod
    def map_crop(cls, node):
        kwargs = {}
        cls._convert_output_shape(kwargs, node)
        offset = node.parameters.offset
        if offset:
            if len(offset) == 1:
                kwargs['border'] = [offset[0], offset[0], 0, 0]
            else:
                kwargs['border'] = [offset[0], offset[1], 0, 0]

        return Node.create('Crop', **kwargs)


    @classmethod
    def map_elu(cls, node):
        kwargs = {}
        cls._convert_output_shape(kwargs, node)
        return Node.create('ELU', **kwargs)


    @classmethod
    def map_relu(cls, node):
        kwargs = {}
        cls._convert_output_shape(kwargs, node)
        return Node.create('Relu', **kwargs)


    @classmethod
    def map_p_re_lu(cls, node):
        # print(node.parameters)
        # assert False
        try:
            scale_value = float(node.parameters.filler.value)
            kwargs = {'gamma' : scale_value}
        except ConversionError:
            kwargs = {'gamma' : 0.25}
        cls._convert_output_shape(kwargs, node)
        return Node.create('PRelu', **kwargs)


    @classmethod
    def map_pooling(cls, node):
        parent, _ = node.get_only_parent()
        kwargs = cls.get_kernel_params(node, parent.output_shape)
        if node.parameters.pool == 0:
            kwargs['pooling_type'] = 'MAX'
        elif node.parameters.pool == 1:
            kwargs['pooling_type'] = 'AVG'
        else:
            # Stochastic pooling, for instance.
            raise ConversionError('Unsupported pooling type.')
        cls._convert_output_shape(kwargs, node)
        return Node.create('Pool', **kwargs)


    @classmethod
    def map_unpooling(cls, node):
        kwargs = {}
        kwargs['kernel_shape'] = [1, node.kernel_parameters.k_h, node.kernel_parameters.k_w, 1]
        kwargs['pads'] = [0, node.kernel_parameters.p_h, node.kernel_parameters.p_w, 0]
        kwargs['strides'] = [1, node.kernel_parameters.s_h, node.kernel_parameters.s_w, 1]
        cls._convert_output_shape(kwargs, node)
        return Node.create('Unpool', **kwargs)


    @classmethod
    def _add_flatten_layer(cls, node):
        shape = TensorShape()
        dim = shape.dim.add()
        dim.size = -1

        dim = shape.dim.add()
        dim.size = 1
        for i in node.output_shape[1:]:
            dim.size *= i
        kwargs = {'_output_shapes' : [shape]}
        return Node.create('Flatten', **kwargs)

    @classmethod
    def map_inner_product(cls, node):
        #TODO: Axis
        assert node.parameters.axis == 1
        #TODO: Unbiased
        shape = TensorShape()
        dim = shape.dim.add()
        dim.size = -1
        dim = shape.dim.add()
        dim.size = 1
        for i in node.output_shape[1:]:
            dim.size *= i
        kwargs = {'use_bias' : node.parameters.bias_term, 'units' : node.parameters.num_output,
                '_output_shapes': [shape]}

        # check if need the Flatten layer
        parent, _ = node.get_only_parent()
        ret = []

        # if parent.output_shape.height > 1 or parent.output_shape.width > 1:
        ret.append(cls._add_flatten_layer(parent))
        ret.append(Node.create('FullyConnected', **kwargs))
        return ret

    @classmethod
    def map_softmax(cls, node):
        kwargs = {}
        cls._convert_output_shape(kwargs, node)
        return Node.create('Softmax', **kwargs)

    @classmethod
    def map_lrn(cls, node):
        params = node.parameters
        assert params.local_size % 2 == 1
        kwargs = {'size': int(params.local_size), 'alpha': params.alpha, 'beta': params.beta, 'bias' : params.k}
        cls._convert_output_shape(kwargs, node)
        return Node.create('LRN', **kwargs)

    @classmethod
    def map_concat(cls, node):
        kwargs = {'axis': (2, 3, 1, 0)[node.parameters.axis]}
        cls._convert_output_shape(kwargs, node)
        return Node.create('Concat', **kwargs)

    @classmethod
    def map_dropout(cls, node):
        kwargs = {'keep_prob': node.parameters.dropout_ratio}
        cls._convert_output_shape(kwargs, node)
        return Node.create('Dropout', **kwargs)

    @classmethod
    def map_batch_norm(cls, node):
        kwargs = {'scale' : len(node.data) >= 3, 'bias' : len(node.data) == 4}
        epsilon = node.parameters.eps
        kwargs['epsilon'] = epsilon
        cls._convert_output_shape(kwargs, node)
        return Node.create('BatchNorm', **kwargs)

    @classmethod
    def map_scale(cls, node):
        # TODO: The gamma parameter has to be set (in node.data?) and this should work.
        # Also, mean should be set to 0, and var to 1, just to be safe.
        if node.data:
            scale_value = float(node.parameters.filler.value)
            if node.parameters.bias_term:
                bias_value = float(node.parameters.bias_filler.value)
                kwargs = {'use_scale' : True, 'use_bias' : node.parameters.bias_term, 'gamma' : scale_value, 'beta': bias_value, 'epsilon': 0}
            else:
                kwargs = {'use_scale' : True, 'use_bias' : node.parameters.bias_term, 'gamma' : scale_value, 'epsilon': 0}

            cls._convert_output_shape(kwargs, node)
            return Node.create('Affine', **kwargs)
        else:
            return Node.create('Mul')


    @classmethod
    def map_eltwise(cls, node):
        operations = {0: 'Mul', 1: 'Add', 2: 'Max'}
        op_code = node.parameters.operation
        try:
            kwargs = {}
            cls._convert_output_shape(kwargs, node)
            return Node.create(operations[op_code], **kwargs)
        except KeyError:
            raise ConversionError('Unknown elementwise operation: {}'.format(op_code))

    @classmethod
    def map_abs_val(cls, node):
        return Node.create('Abs')

    @classmethod
    def map_tanh(cls, node):
        return Node.create('Tanh')

    @classmethod
    def map_sigmoid(cls, node):
        return Node.create('Sigmoid')

    @classmethod
    def map_reshape(cls, node):
        kwargs = {'shape' : [dim for dim in node.output_shape]}
        cls._convert_output_shape(kwargs, node)
        return Node.create('Reshape', **kwargs)

    @classmethod
    def map_flatten(cls, node):
        return cls._add_flatten_layer(node)

    @classmethod
    def map_split(cls, node):
        # skip the split node
        return

    @classmethod
    def map_elu(cls, node):
        kwargs = {}
        cls._convert_output_shape(kwargs, node)
        return Node.create('ELU', **kwargs)
