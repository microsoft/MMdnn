#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from six.moves import xrange

import cntk
from mmdnn.conversion.common.IR.IR_graph import IRGraph, IRGraphNode
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.DataStructure.emitter import Emitter
from mmdnn.conversion.common.utils import *
from mmdnn.conversion.rewriter.folder import *

class CntkEmitter(Emitter):
    
    dtype_map = {
        graph_pb2.DT_FLOAT16 : "np.float16",
        graph_pb2.DT_FLOAT32 : "np.float32",
        graph_pb2.DT_FLOAT64 : "np.float64",
        graph_pb2.DT_INT16 : "np.float16",  # Cntk does not support Int.
        graph_pb2.DT_INT32 : "np.float32",  # Cntk does not support Int.
        graph_pb2.DT_INT64 : "np.float64",  # Cntk does not support Int.
        graph_pb2.DT_UINT8 : "np.uint8",
        graph_pb2.DT_UINT16 : "np.uint16"
    }


    naive_scope_pattern = ['gru_cell', 'lstm_cell']

    def __init__(self, model):
        from six import string_types as _string_types
        super(CntkEmitter, self).__init__()
        if isinstance(model, _string_types):
            network_path = model
        else:
            network_path = model[0]
            self._load_weights(model[1])

        self.IR_graph = IRGraph(network_path)
        super(CntkEmitter, self)._build()
        self.yolo_parameter = []
        folder = Folder(self.IR_graph, self.weights_dict)
        folder.fold()


    @property
    def header_code(self):
        return """import numpy as np
import cntk
from cntk import ops, layers
from cntk.contrib.crosstalkcaffe.unimodel.cntkinstance import BlockApiSetup

_weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file = None):
    global _weights_dict
    _weights_dict = load_weights(weight_file)

"""


    def gen_code(self, phase = 'test'):
        self.phase = phase
        self.add_body(0, self.header_code)

        for layer in self.IR_graph.topological_sort:
            current_node = self.IR_graph.get_node(layer)
            node_type = current_node.type

            if hasattr(self, "emit_" + node_type):
                func = getattr(self, "emit_" + node_type)
                line = func(current_node)
                if line:
                    self.add_body(1, line)
            else:
                print("CntkEmitter has not supported operator [%s]." % (node_type))
                self.emit_UNKNOWN(current_node)

        self.add_body(1, "return {}".format(
            ','.join([self.IR_graph.get_node(name).real_variable_name for name in self.IR_graph.output_layers])))

        self.add_body(0, "")
        for i in self.used_layers:
            func = getattr(self, "_layer_" + i)
            func()

        self.add_body(0, "")
        for code in self.layers_codes.values():
            self.add_body(0, code)

        return self.body_code


    @staticmethod
    def _shapeToStr(shapes):
        new_shape = filter(lambda x:x >- 1, [dim.size for dim in shapes.dim])
        return ', '.join('%s' % i for i in new_shape)


    @staticmethod
    def is_valid_padding(auto_pad, pads):
        """
        different from utils.is_valid_padding
        """
        if auto_pad:
            if auto_pad == 'VALID':
                return True
            elif auto_pad.startswith('SAME'):
                return False
            else:
                raise ValueError("Unknown padding type{}.".format(auto_pad))

        else:
            lens = len(pads)
            assert lens % 2 == 0
            for i in range(0, lens // 2):
                if pads[i] != 0:
                    return False
            return True

    @staticmethod
    def is_ceil_mode(pads):
        lens = len(pads)
        for i in range(lens // 2 + 1, lens - 1):
            if pads[i] == pads[i - lens // 2]:
                return False
        else:
            return True


    def _defuse_padding(self, IR_node):
        auto_pad = IR_node.get_attr('auto_pad')
        if auto_pad:
            input_node = self.parent_variable_name(IR_node)
            if auto_pad == 'VALID':
                padding = False
            elif auto_pad.startswith("SAME"):
                padding = True
            else:
                raise ValueError("Unknown padding type [{}].".format(auto_pad))

            return input_node, padding

        else:
            padding = IR_node.get_attr('pads')
            if not is_valid_padding(padding):
                dim = len(padding) // 2
                padding_str = list()
                for i in xrange(1, dim):
                    padding_str.append((padding[i], padding[i + dim]))
                input_node = IR_node.variable_name + '_pad'
                self.add_body(1, "{:<15} = cntk.pad({}, pattern={})".format(
                    input_node,
                    self.parent_variable_name(IR_node),
                    padding_str))

            else:
                input_node = self.parent_variable_name(IR_node)

            return input_node, False



    def emit_Conv(self, IR_node):
        codes = list()
        if self.weight_loaded:
            self.used_layers.add('Conv')
            input_node, padding = self._defuse_padding(IR_node)

            dim = len(IR_node.get_attr('strides')) - 2
            padding = [False] + [padding] * dim

            if IR_node.type == 'DepthwiseConv':
                groups = IR_node.get_attr('kernel_shape')[-2]
                codes.append("_weights_dict['{}']['weights'] = np.swapaxes(_weights_dict['{}']['weights'], -1, -2)".format(
                    IR_node.real_name, IR_node.real_name))
            else:
                groups = IR_node.get_attr('group', 1)

            codes.append("{:<15} = convolution({}, is_transpose={}, strides={}, auto_padding={}, dilation={}, groups={}, name='{}')".format(
                IR_node.variable_name,
                input_node,
                IR_node.type == 'ConvTranspose',
                tuple(IR_node.get_attr('strides')[1:-1]),
                padding,
                tuple(IR_node.get_attr('dilations', [1])),
                groups,
                IR_node.name))

        else:
            codes.append("{:<15} = Convolution(name = '{}', num_filters = {}, filter_shape = ({}), strides = ({},), pad = {}, bias = {})({})\n".format(
                IR_node.variable_name,
                IR_node.name,
                IR_node.get_attr('kernel_shape')[-1],
                ', '.join('%s' % i for i in IR_node.layer.attr["kernel_shape"].list.i[:-2]),
                ', '.join('%s' % i for i in IR_node.layer.attr['strides'].list.i[1:-1]),
                IR_node.get_attr('auto_pad') != 'VALID',
                IR_node.get_attr('use_bias'),
                self.parent_variable_name(IR_node)))
        return codes


    def emit_Pool(self, IR_node):
        input_node = self.IR_graph.get_node(IR_node.in_edges[0]).real_variable_name
        if IR_node.layer.attr['global_pooling'].b:
            self.used_layers.add('GlobalPooling')
            code = "{:<15} = global_pooling({}, '{}', name = '{}')".format(
                IR_node.variable_name,
                input_node,
                IR_node.get_attr('pooling_type'),
                IR_node.name)
        else:
            for e in IR_node.get_attr('dilations', []):
                assert e == 1

            dim = len(IR_node.get_attr('kernel_shape')) - 2
            padding = not self.is_valid_padding(IR_node.get_attr('auto_pad'), IR_node.get_attr('pads'))
            padding = [False] + [padding] * dim
            ceil_out_dim = self.is_ceil_mode(IR_node.get_attr('pads'))

            pooling_type = IR_node.get_attr('pooling_type')
            if pooling_type == 'MAX':
                pooling_type = cntk.MAX_POOLING
            elif pooling_type == 'AVG':
                pooling_type = cntk.AVG_POOLING
            else:
                raise ValueError

            if self.weight_loaded:
                self.used_layers.add(IR_node.type)
                code = "{:<15} = pooling({}, pooling_type={}, pooling_window_shape={}, strides={}, auto_padding={}, ceil_out_dim={})".format(
                    IR_node.variable_name,
                    input_node,
                    pooling_type,
                    tuple(IR_node.get_attr('kernel_shape')[1:-1]),
                    tuple(IR_node.get_attr('strides')[1:-1]),
                    padding,
                    ceil_out_dim
                    )
            else:
                raise NotImplementedError
        return code


    def emit_UNKNOWN(self, IR_node):
        print(IR_node.IR_layer.name)


    def emit_DataInput(self, IR_node):

        shape_str = self._shapeToStr(IR_node.IR_layer.attr["shape"].shape)
        
        dtype_str = ", dtype = {}".format(self.dtype_map[IR_node.layer.attr['dtype'].type]) if 'dtype' in IR_node.layer.attr else ""
        code = "{:<15} = cntk.sequence.input_variable(({},) {}, name='{}')".format(
            IR_node.variable_name,
            shape_str,
            dtype_str,
            IR_node.name)
        return code


    def emit_Dropout(self, IR_node):
        parent = self.IR_graph.get_parent(IR_node.name, [0])
        if self.phase == 'train':
            code = "{:<15} = Dropout({}, name = '{}')({})".format(
                IR_node.variable_name,
                1 - IR_node.get_attr('keep_prob'),
                IR_node.name,
                parent.real_variable_name)
            return code
        else:
            IR_node.real_name = parent.real_name


    def emit_FullyConnected(self, IR_node):
        input_node = self.parent_variable_name(IR_node)
        if self.weight_loaded:
            self.used_layers.add(IR_node.type)
            code = "{:<15} = dense({}, name = '{}')".format(
                IR_node.variable_name,
                input_node,
                IR_node.name)

        else:
            code = "{:<15} = Dense({}, bias = {}, name = '{}')({})".format(
                IR_node.variable_name,
                IR_node.layer.attr["units"].i,
                IR_node.layer.attr['use_bias'].b,
                IR_node.name,
                input_node)
        return code


    def emit_Flatten(self, IR_node):
        code = "{:<15} = ops.reshape({}, (-1,), name = '{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.name)
        return code


    def emit_Reshape(self, IR_node):
        code = "{:<15} = cntk.reshape({}, shape={}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            tuple(IR_node.get_attr('shape')),
            IR_node.name)
        return code


    def _emit_activation(self, IR_node, op_name):
        code = "{:<15} = layers.Activation(activation = {}, name = '{}')({})".format(
            IR_node.variable_name,
            op_name,
            IR_node.name,
            self.parent_variable_name(IR_node))
        return code


    def emit_Tanh(self, IR_node):
        return self._emit_activation(IR_node, 'ops.tanh')


    def emit_Relu(self, IR_node):
        return self._emit_activation(IR_node, 'ops.relu')


    def emit_Softmax(self, IR_node):
        return self._emit_activation(IR_node, 'ops.softmax')


    def emit_Sigmoid(self, IR_node):
        return self._emit_activation(IR_node, 'ops.sigmoid')


    def emit_RNNs(self, IR_node, func):
        assert False


    def emit_LSTM(self, IR_node):
        return self.emit_RNNs(IR_node, "LSTM")

    def emit_GRU(self, IR_node):
        return self.emit_RNNs(IR_node, "GRU")


    def emit_Add(self, IR_node):
        if len(IR_node.in_edges) > 1:
            inputs = ' + '.join(self.parent_variable_name(IR_node, i) for i in IR_node.in_edges)
            code = "{:<15} = {}".format(
                IR_node.variable_name,
                inputs)
            return code


    def emit_Sub(self, IR_node):
        if len(IR_node.in_edges) > 1:
            inputs = ' - '.join(self.parent_variable_name(IR_node, i) for i in IR_node.in_edges)
            code = "{:<15} = {}".format(
                IR_node.variable_name,
                inputs)
            return code


    def emit_Mul(self, IR_node):
        if len(IR_node.in_edges) > 1:
            inputs = ' * '.join(self.parent_variable_name(IR_node, i) for i in IR_node.in_edges)
            code = "{:<15} = {}".format(
                IR_node.variable_name,
                inputs)
            return code


    def emit_Constant(self, IR_node):
        if IR_node.get_attr('value'):
            code = "{:<15} = cntk.Constant(value={})".format(
            IR_node.variable_name, IR_node.get_attr('value'))
        else:
            code = "{:<15} = cntk.Constant(value=_weights_dict['{}']['value'])".format(
                IR_node.variable_name, IR_node.name)
        return code


    def emit_Concat(self, IR_node):
        inputs = ', '.join(self.parent_variable_name(IR_node, i) for i in IR_node.in_edges)
        for s in IR_node.in_edges:
            node = self.IR_graph.get_node(s)

        code = "{:<15} = cntk.splice({}, axis={}, name='{}')".format(
            IR_node.variable_name,
            inputs,
            IR_node.get_attr('axis') -1 , # why -1 ?
            IR_node.name)
        return code


    def emit_BatchNorm(self, IR_node):
        self.used_layers.add(IR_node.type)
        code = "{:<15} = batch_normalization({}, epsilon={}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('epsilon'),
            IR_node.name)
        return code


    def emit_Pad(self, IR_node):
        if IR_node.get_attr('mode') == 'constant':
            mode = 'mode = ops.CONSTANT_PAD, constant_value = {}'.format(IR_node.get_attr('constant_values', 0.0))
        elif IR_node.get_attr('mode') == 'reflect':
            mode = 'mode = ops.REFLECT_PAD'
        elif IR_node.get_attr('mode') == 'SYMMETRIC':
            mode = 'mode = ops.SYMMETRIC_PAD'
        else:
            assert False

        padding = IR_node.get_attr('pads')
        padding = convert_onnx_pad_to_tf(padding)[1:]

        code = "{:<15} = ops.pad({}, pattern={}, {})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            padding,
            mode)
        return code


    def emit_Squeeze(self, IR_node):
        IR_node.real_name = self.IR_graph.get_node(IR_node.in_edges[0]).real_name


    def emit_Log(self, IR_node):
        code = "{:<15} = _cntk.log({}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.name)
        return code


    def emit_Exp(self, IR_node):
        code = "{:<15} = _cntk.exp({}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.name)
        return code


    def emit_Embedding(self, IR_node):
        
        codes = list()
        codes.append("{}_P = cntk.one_hot({}, _weights_dict['{}']['weights'].shape[0])".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.name))
        
        codes.append("{:<15} = layers.Embedding(weights=_weights_dict['{}']['weights'])({}_P)".format(
            IR_node.variable_name,
            # IR_node.get_attr('output_dim'),
            IR_node.name,
            IR_node.variable_name))

        return codes


    def emit_Reciprocal(self, IR_node):
        code = "{:<15} = _cntk.reciprocal({}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.name)
        return code


    def emit_ReduceMean(self, IR_node):
        code = "{:<15} = ops.reduce_mean({}, axis = ({}), name = '{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            ', '.join('%s' % (i - 1) for i in IR_node.get_attr('axes')),
            IR_node.name)
        return code


    def emit_LRN(self, IR_node):
        self.used_layers.add(IR_node.type)
        output_name = IR_node.variable_name
        input_name = self.parent_variable_name(IR_node)
        IR_name = IR_node.name
        size = IR_node.get_attr('size')
        depth_radius = int(size / 2)
        alpha = IR_node.get_attr('alpha')
        #alpha = alpha / size
        beta = IR_node.get_attr('beta')
        bias = IR_node.get_attr('bias')

        code = "{:<15} = lrn({}, k={}, n={}, alpha={}, beta={}, name='{}')".format(
            output_name,
            input_name,
            bias,
            depth_radius + 1,
            alpha,
            beta,
            IR_name)
        return code

    # ??
    def emit_LeakRelu(self, IR_node):
        code = "{:<15} = _cntk.relu({}) - {} * _cntk.relu(-{})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('alpha'),
            self.parent_variable_name(IR_node))
        return code


    def emit_LeakyRelu(self, IR_node):
        self.used_layers.add(IR_node.type)
        code = "{:<15} = _leaky_relu({}, {}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('alpha'),
            IR_node.name)
        return code


    def emit_UpSampling2D(self, IR_node):
        self.used_layers.add(IR_node.type)
        code = "{:<15} = Upsampling2D({}, stride = {}, name = '{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('scales')[0],
            IR_node.name)
        return code


    def emit_ConvTranspose(self, IR_node):
        return self.emit_Conv(IR_node)


    def emit_yolo(self, IR_node):
        self.used_layers.add(IR_node.type)
        code = "{:<15} = {}".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node)
        )
        # print(IR_node.layer)
        self.yolo_parameter = [IR_node.get_attr('anchors'),
            IR_node.get_attr('classes'),
            IR_node.get_attr("ignore_thresh"),
            IR_node.get_attr("jitter")]
        # assert False
        return code


    def emit_Crop(self, IR_node):
        self.used_layers.add(IR_node.type)
        output_shape = IR_node.get_attr('_output_shapes')[0]
        output_shape = shape_to_list(output_shape)[1:]
        code = "{:<15} = _crop({}, {}, {}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('border')[:2],
            output_shape,
            IR_node.real_name)
        return code


    def emit_Relu6(self, IR_node):
        codes = list()
        codes.append(self.emit_Relu(IR_node))
        codes.append("{:<15} = cntk.clip({}, 0, 6, name='{}_clip')".format(
            IR_node.variable_name + "_clip",
            IR_node.variable_name,
            IR_node.name
        ))
        IR_node.real_name = IR_node.name + '_clip'
        return codes


    def emit_DepthwiseConv(self, IR_node):
        return self.emit_Conv(IR_node)


    # def emit_Unstack(self, IR_node):
        # num_str = "{}.shape[{}]".format(self.parent_variable_name(IR_node), IR_node.get_attr('axis'))
        # axis = IR_node.get_attr('axis')
        # parent_variable_shape = "list({}.shape)".format(self.parent_variable_name(IR_node) 
        #         if self.IR_graph.get_parent(IR_node.name, [0]).type != 'Embedding' 
        #             else self.parent_variable_name(IR_node)+'.E')
        # if axis==1:
        #     shape_str = "tuple([{}[0]*{}[{}], 1].extend({}[{}+1:]))".format(
        #         parent_variable_shape,
        #         parent_variable_shape,
        #         str(axis),
        #         parent_variable_shape,
        #         str(axis))
        # else:
        #     shape_str = "tuple([{}[0]*{}[{}]].extend({}[1:{}]).append(1).extend({}[{}+1:]))".format(
        #         parent_variable_shape,
        #         parent_variable_shape,
        #         str(axis),
        #         parent_variable_shape,
        #         str(axis),
        #         parent_variable_shape,
        #         str(axis))
        # code = "{:<15} = cntk.reshape({}, {}, name='{}')".format(
        #     IR_node.variable_name, 
        #     self.parent_variable_name(IR_node), 
        #     shape_str,
        #     IR_node.variable_name)
        # code = "{: <15} = cntk.reshape({}, {}.shape, name='{}')".format(
        #     IR_node.variable_name,
        #     self.parent_variable_name(IR_node),
        #     self.parent_variable_name(IR_node),
        #     IR_node.name
        # )
        # return code


    def emit_Shape(self, IR_node):
        parent_node = self.IR_graph.get_parent(IR_node.name, [0])
        code = "{:<15} = {}.shape".format(
            IR_node.variable_name, 
            self.parent_variable_name(IR_node) if parent_node.type != 'Embedding' else self.parent_variable_name(IR_node)+".E")
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
        code = "{:<15} = {}[{}]".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            extra_str)
        return code


    def emit_Split(self, IR_node):
        self.used_layers.add(IR_node.type)
        axis = IR_node.get_attr('axis')
        split_num = IR_node.get_attr('split')
        code = "{:<15} = split(input={}, axis={}, split_num={})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            str(axis),
            str(split_num))
        
        return code


    # def emit_Fill(self, IR_node):
    #     code = "{:<15} = cntk.Constant({}, {}, name='{}')".format(
    #         IR_node.variable_name,
    #         IR_node.get_attr('value'),
    #         self.parent_variable_name(IR_node),
    #         IR_node.name)
    #     return code


    def emit_Unsqueeze(self, IR_node):
        code = "{:<15} = cntk.expand_dims({}, axis={}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('axes')[0],
            IR_node.name)
        return code


    def emit_Scope(self, IR_node):
        pattern = IR_node.pattern
        if pattern not in self.naive_scope_pattern and re.sub(r'(_\d+)*$', '', IR_node.pattern) not in self.naive_scope_pattern:
            func = getattr(self, "_emit_" + pattern)
            code = func(IR_node)
        else:
            code = "{:<15} = __{}({})".format(
                IR_node.real_variable_name,
                IR_node.pattern,
                ', '.join(self.parent_variable_name(IR_node, s) for s in IR_node.in_edges))
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
                        body_code += "    " + line + '\n'
                else:
                    print("CntkEmitter has not supported operator [%s]." % (node_type))
                    self.emit_UNKNOWN(node)

            # param_code does not need parameter slice.
            input_params = scope_node.input_params
            param_code = ', '.join(input_params)
            function_code = _scope_func(scope_node.pattern, param_code, body_code, scope_node.return_variables)

            self.layers_codes[scope_node.pattern] = function_code


    def _emit_h_zero(self, IR_node):
        code = "{:<15} = cntk.Constant({}, (1, {}))".format(
            IR_node.variable_name,
            IR_node.get_attr('fill_value'),
            IR_node.get_attr('fill_size'))
        return code


    def _layer_Crop(self):
        self.add_body(0, '''
def _crop(input, border, output_shape, **kwargs):
    dim = len(output_shape)
    output_shape = [output_shape[-1]] + output_shape[:-1]
    ref_tensor = np.zeros(shape=output_shape, dtype=np.float32)

    input = cntk.transpose(input, [dim - 1] + list(range(0, dim - 1)))
    layer = cntk.crop_manual(node_input=input, node_referent=ref_tensor, offset_x=border[0], offset_y=border[1])
    layer = cntk.transpose(layer, list(range(1, dim)) + [0])
    return layer
''')


    def _layer_LeakyRelu(self):
        self.add_body(0, '''
def _leaky_relu(x, leak, name):
    return cntk.param_relu(cntk.constant((np.ones(x.shape)*leak).astype(np.float32)), x, name = name)
''')


    def _layer_yolo(self):
        self.add_body(0, '''
def yolo_parameter():
    return {}
'''.format(self.yolo_parameter))


    def _layer_upsample(self):
        self.add_body(0, '''
def Upsampling2D(x, stride, name):
    assert stride == 2
    xr = cntk.reshape(x, (x.shape[0], 1, x.shape[1], 1, x.shape[2]))
    xx = cntk.splice(xr, xr, axis = -2)
    xy = cntk.splice(xx, xx, axis = -4)
    r = cntk.reshape(xy, (x.shape[0] * 2, x.shape[1] * 2, x.shape[2]), name = name)
    return r
''')


    def _layer_LRN(self):
        self.add_body(0, """
def lrn(input, **kwargs):
    dim = len(input.output.shape)
    input = cntk.transpose(input, [dim - 1] + list(range(0, dim - 1)))
    layer = BlockApiSetup.lrn(**kwargs)(input)
    layer = cntk.transpose(layer, list(range(1, dim)) + [0])
    return layer
""")


    def _layer_FullyConnected(self):
        self.add_body(0, """
def dense(input, name, **kwargs):
    w = _weights_dict[name]['weights']
    b = _weights_dict[name]['bias'] if 'bias' in _weights_dict[name] else None
    return BlockApiSetup.linear(output_shape=w.shape[1], input_shape=w.shape[0], scale_init=w, bias_init=b, name=name, **kwargs)(input)
""")


    def _layer_Conv(self):
        self.add_body(0, """
def convolution(input, is_transpose, name, **kwargs):
    dim = _weights_dict[name]['weights'].ndim

    if is_transpose:
        weight = np.transpose(_weights_dict[name]['weights'], [dim - 2, dim - 1] + list(range(0, dim - 2)))
        kwargs.pop('groups', None)
    else:
        weight = np.transpose(_weights_dict[name]['weights'], [dim - 1, dim - 2] + list(range(0, dim - 2)))
    w = cntk.Parameter(init=weight, name=name + '_weight')

    input = cntk.transpose(input, [dim - 2] + list(range(0, dim - 2)))

    if is_transpose:
        layer = ops.convolution_transpose(w, input, **kwargs)
    else:
        layer = ops.convolution(w, input, **kwargs)
    if 'bias' in _weights_dict[name]:
        bias = np.reshape(_weights_dict[name]['bias'], [-1] + [1] * (dim - 2))
        b = cntk.Parameter(init=bias, name=name + '_bias')
        layer = layer + b
    layer = cntk.transpose(layer, list(range(1, dim - 1)) + [0])
    return layer
""")


    def _layer_Pool(self):
        self.add_body(0, """
def pooling(input, **kwargs):
    dim = len(input.output.shape)
    input = cntk.transpose(input, [dim - 1] + list(range(0, dim - 1)))
    layer = ops.pooling(input, **kwargs)
    layer = cntk.transpose(layer, list(range(1, dim)) + [0])
    return layer
""")


    def _layer_GlobalPooling(self):
        self.add_body(0, """
def global_pooling(input, type, **kwargs):
    dim = len(input.output.shape)
    input = cntk.transpose(input, [dim - 1] + list(range(0, dim - 1)))
    layer = layers.GlobalMaxPooling(**kwargs)(input) if type == 'MAX' else layers.GlobalAveragePooling(**kwargs)(input)
    layer = cntk.transpose(layer, list(range(1, dim)) + [0])
    return layer
""")


    def _layer_BatchNorm(self):
        self.add_body(0, """
def batch_normalization(input, name, epsilon, **kwargs):
    mean = cntk.Parameter(init = _weights_dict[name]['mean'],
        name = name + "_mean")
    var = cntk.Parameter(init = _weights_dict[name]['var'],
        name = name + "_var")

    layer = (input - mean) / cntk.sqrt(var + epsilon)
    if 'scale' in _weights_dict[name]:
        scale = cntk.Parameter(init = _weights_dict[name]['scale'],
            name = name + "_scale")
        layer = scale * layer

    if 'bias' in _weights_dict[name]:
        bias = cntk.Parameter(init = _weights_dict[name]['bias'],
            name = name + "_bias")
        layer = layer + bias

    return layer
""")


    def _layer_Split(self):
        self.add_body(0, """
def split(input, axis, split_num):
        split_len = input.shape[axis]
        res = []
        st = 0
        for i in range(split_num):
            ed = st + split_len//split_num
            res.append(cntk.slice(input, axis, st, ed))
            st += split_len//split_num

        return res
        """)
