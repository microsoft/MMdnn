#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import division

import os
import sys
import math
import numpy as np

import caffe
from caffe import layers as L
from caffe import params as P
from mmdnn.conversion.common.IR.IR_graph import IRGraph, IRGraphNode
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.DataStructure.emitter import Emitter
from mmdnn.conversion.common.utils import *


class CaffeEmitter(Emitter):

    def __init__(self, model):
        from six import string_types as _string_types
        super(CaffeEmitter, self).__init__()
        if isinstance(model, _string_types):
            network_path = model
        else:
            network_path = model[0]
            self._load_weights(model[1])

        self.IR_graph = IRGraph(network_path)
        super(CaffeEmitter, self)._build()


    @property
    def header_code(self):
        return """from __future__ import print_function
import numpy as np
import sys, argparse
import caffe
from caffe import layers as L
from caffe import params as P
from caffe import to_proto
from six import text_type as _text_type


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
    n = caffe.NetSpec()
"""

    @property
    def end_code(self):
        return """    return n

def make_net(prototxt):
    n = KitModel()
    with open(prototxt, 'w') as fpb:
        print(n.to_proto(), file=fpb)

def gen_weight(weight_file, model, prototxt):
    global _weights_dict
    _weights_dict = load_weights(weight_file)

    net = caffe.Net(prototxt, caffe.TRAIN)

    for key in _weights_dict:
        if 'weights' in _weights_dict[key]:
            net.params[key][0].data.flat = _weights_dict[key]['weights']
        elif 'mean' in _weights_dict[key]:
            net.params[key][0].data.flat = _weights_dict[key]['mean']
            net.params[key][1].data.flat = _weights_dict[key]['var']
            if 'scale' in _weights_dict[key]:
                net.params[key][2].data.flat = _weights_dict[key]['scale']
        elif 'scale' in _weights_dict[key]:
            net.params[key][0].data.flat = _weights_dict[key]['scale']
        if 'bias' in _weights_dict[key]:
            net.params[key][1].data.flat = _weights_dict[key]['bias']
        if 'gamma' in _weights_dict[key]: # used for prelu, not sure if other layers use this too
            net.params[key][0].data.flat = _weights_dict[key]['gamma']
    net.save(model)
    return net



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate caffe model and prototxt')
    parser.add_argument('--weight_file', '-w', type=_text_type, default='IR weight file')
    parser.add_argument('--prototxt', '-p', type=_text_type, default='caffe_converted.prototxt')
    parser.add_argument('--model', '-m', type=_text_type, default='caffe_converted.caffemodel')
    args = parser.parse_args()
    # For some reason argparser gives us unicode, so we need to conver to str first
    make_net(str(args.prototxt))
    gen_weight(str(args.weight_file), str(args.model), str(args.prototxt))

"""

    def gen_code(self, phase = 'test'):
        self.phase = phase
        self.add_body(0, self.header_code)

        #for test
        # with open("graph.txt", 'w') as f:
        #     for layer in self.IR_graph.topological_sort:
        #         current_node = self.IR_graph.get_node(layer)
        #         print("========current_node=========\n{}".format(current_node.layer), file=f)
        #test end

        for layer in self.IR_graph.topological_sort:
            current_node = self.IR_graph.get_node(layer)
            node_type = current_node.type
            #print("========current_node={}".format(current_node.layer))

            if hasattr(self, "emit_" + node_type):
                func = getattr(self, "emit_" + node_type)
                func(current_node)
            else:
                print("CaffeEmitter has not supported operator [%s]." % (node_type))
                self.emit_UNKNOWN(current_node)

        self.add_body(0, "")
        self.add_body(0,self.end_code)

        return self.body_code


    def run(self, dstNetworkPath, dstWeightPath = None, phase = 'test'):
        super(CaffeEmitter, self).run(dstNetworkPath, dstWeightPath, phase)
        if self.weight_loaded:
            self.save_weights(self.weights_dict, dstWeightPath)


    @staticmethod
    def _shapeToStr(shapes):
        return [dim.size if dim.size > 0 else 1 for dim in shapes.dim]


    def _get_symmetric_padding(self, IR_node):
        stride_h = IR_node.get_attr('strides')[1]
        stride_w = IR_node.get_attr('strides')[2]

        # check if have pad layer
        IR_parent_node = self.IR_graph.get_parent(IR_node.name, [0])
        if IR_parent_node.type == 'Pad':
            pads = IR_parent_node.get_attr('pads')
        else:
            pads = IR_node.get_attr('pads')

        # Pad_h < kernel_h (vgg19 caffe2caffe)
        if IR_node.type == "Pool":
            if pads[1]:
                pad_h = pads[1] + (0 if pads[1] == pads[5] else stride_h)
            else:
                pad_h = 0
            if pads[2]:
                pad_w = pads[2] + (0 if pads[2] == pads[6] else stride_w)
            else:
                pad_w = 0
        elif IR_node.type == "Unpool":
            pad_h = 0
            pad_w = 0
        else:
            pad_h = pads[1] + (0 if pads[1] == pads[5] else stride_h)
            pad_w = pads[2] + (0 if pads[2] == pads[6] else stride_w)

        return pad_h, pad_w


    def check_if_need_transpose(self, IR_node):
        parent = self.IR_graph.get_parent(IR_node.name, [0])
        while parent.type == 'Flatten' or parent.type == 'Dropout' or parent.type == 'Reshape':
            parent = self.IR_graph.get_parent(parent.name, [0])
        dim = len(parent.layer.attr['_output_shapes'].list.shape[0].dim)
        if dim > 2:
            original_dims = self.weights_dict[IR_node.name]['weights'].shape
            dims = [i.size for i in parent.layer.attr['_output_shapes'].list.shape[0].dim[1:]] + [-1]
            self.weights_dict[IR_node.name]['weights'] = np.reshape(self.weights_dict[IR_node.name]['weights'], dims)
            self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], [dim - 2] + list(range(0, dim - 2)) + [dim - 1])
            self.weights_dict[IR_node.name]['weights'] = np.reshape(self.weights_dict[IR_node.name]['weights'], original_dims)


    def emit_Conv(self, IR_node):
        # implement asymmetric paddings by applying symmetric padding then cropping
        pad_h, pad_w = self._get_symmetric_padding(IR_node)

        num_output = IR_node.get_attr('kernel_shape')[-1]
        if IR_node.type == "DepthwiseConv":
            num_group = IR_node.get_attr("kernel_shape")[-2]
            # num_output = IR_node.get_attr('kernel_shape')[-2]
            num_output = IR_node.get_attr("_output_shapes")[0].dim[3].size
        else:
            num_group = IR_node.get_attr("group", 1)

        self.add_body(1, "n.{:<15} = L.Convolution(n.{}, kernel_h={}, kernel_w={}, stride={}, num_output={}, pad_h={}, pad_w={}, group={}, \
            bias_term={}, ntop=1)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('kernel_shape')[0],
            IR_node.get_attr('kernel_shape')[1],
            IR_node.get_attr('strides')[1],
            num_output,
            pad_h,
            pad_w,
            num_group,
            IR_node.get_attr('use_bias', False)))

        dim = len(IR_node.get_attr('strides')) - 2
        if self.weight_loaded:
            if IR_node.type == "DepthwiseConv":
                self.weights_dict[IR_node.name]['weights'] = np.swapaxes(self.weights_dict[IR_node.name]['weights'], -1, -2)
            self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], [dim + 1, dim] + list(range(0, dim)))
            self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)

        self.check_if_need_crop(IR_node)
        # keys = []
        # for key in self.weights_dict[IR_node.name].keys():
        #     keys.append(key)
        # print("=======Layer: {}, keys: {}".format(IR_node.name, keys))

    def compute_output_shape(self, IR_node, kernel_h, kernel_w):
        parent_node = self.IR_graph.get_parent(IR_node.name, [0])

        if parent_node.get_attr('_output_shapes'):
            shape = parent_node.get_attr('_output_shapes')[0]
            shape = shape_to_list(shape)
            h_i = shape[1]
            w_i = shape[2]
            pad_h, pad_w = self._get_symmetric_padding(IR_node)
            stride_h = IR_node.get_attr('strides')[1]
            stride_w = IR_node.get_attr('strides')[2]

            if IR_node.type == 'Pool':
                h_o = (h_i + 2 * pad_h - kernel_h + stride_h - 1) // stride_h + 1
                w_o = (w_i + 2 * pad_w - kernel_w + stride_w - 1) // stride_w + 1
            elif IR_node.type == 'Unpool':
                h_o = (h_i - 2 * pad_h - kernel_h + stride_h) * stride_h
                w_o = (w_i - 2 * pad_w - kernel_w + stride_w) * stride_w
            else:
                h_o = (h_i + 2 * pad_h - kernel_h) // stride_h + 1
                w_o = (w_i + 2 * pad_w - kernel_w) // stride_w + 1
            return h_o, w_o
        else:
            assert False


    def check_if_need_crop(self, IR_node):
        shape = IR_node.get_attr('_output_shapes')[0]
        shape = shape_to_list(shape)
        ir_ho = shape[1]
        ir_wo = shape[2]
        if ir_ho <0 or ir_wo<0:
            return
        if IR_node.type == 'Pool' or IR_node.type == 'Unpool':
            k_h = IR_node.get_attr('kernel_shape')[1]
            k_w = IR_node.get_attr('kernel_shape')[2]
        else:
            k_h = IR_node.get_attr('kernel_shape')[0]
            k_w = IR_node.get_attr('kernel_shape')[1]

        caffe_ho, caffe_wo = self.compute_output_shape(IR_node, k_h, k_w)

        # if asymmetric padding, set offset to 1
        pads = IR_node.get_attr('pads')
        offset = [0 if pads[1] == pads[5] else 1,
                  0 if pads[2] == pads[6] else 1]
        if caffe_ho > ir_ho or caffe_wo > ir_wo:
            crop_layer_variable_name = IR_node.variable_name + "_crop"
            self.add_body(1, "n.{:<15} = L.Crop(n.{}, L.DummyData(shape=[dict(dim=[1, {}, {}, {}])], \
                ntop=1), ntop=1, offset={})".format(
                crop_layer_variable_name,
                IR_node.variable_name,
                shape[3],
                ir_ho,
                ir_wo,
                offset
            ))
            # Change the layer name
            IR_node.real_name = IR_node.real_name + "_crop"


    def emit_Pool(self, IR_node):
        pooling_type = IR_node.get_attr('pooling_type')
        if pooling_type == 'MAX':
            pooling_type = P.Pooling.MAX
        elif pooling_type == 'AVG':
            pooling_type = P.Pooling.AVE
        elif pooling_type == 'STOCHASTIC':
            pooling_type = P.Pooling.STOCHASTIC
        else:
            raise ValueError()

        if IR_node.layer.attr['global_pooling'].b:
            self.add_body(1, "n.{:<15} = L.Pooling(n.{}, pool={}, stride={}, global_pooling=True, ntop=1)".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                pooling_type,
                IR_node.get_attr('strides')[1]))
        else:
            pad_h, pad_w = self._get_symmetric_padding(IR_node)
            pool_size = IR_node.get_attr('kernel_shape')[1:3]
            if pool_size[0] != pool_size[1]:
                self.add_body(1, "n.{:<15} = L.Pooling(n.{}, pool={}, kernel_h={}, kernel_w={}, pad_h={}, pad_w={}, stride={}, ntop=1)".format(
                    IR_node.variable_name,
                    self.parent_variable_name(IR_node),
                    pooling_type,
                    pool_size[0],
                    pool_size[1],
                    pad_h,
                    pad_w,
                    IR_node.get_attr('strides')[1]))
            else:
                self.add_body(1, "n.{:<15} = L.Pooling(n.{}, pool={}, kernel_size={}, pad_h={}, pad_w={}, stride={}, ntop=1)".format(
                    IR_node.variable_name,
                    self.parent_variable_name(IR_node),
                    pooling_type,
                    pool_size[0],
                    pad_h,
                    pad_w,
                    IR_node.get_attr('strides')[1]))

            # check if need crop output shape
            self.check_if_need_crop(IR_node)

    def emit_Unpool(self, IR_node):
        pad_h, pad_w = self._get_symmetric_padding(IR_node)
        pool_size = IR_node.get_attr('kernel_shape')[1:3]
        if pool_size[0] != pool_size[1]:
            self.add_body(1, "n.{:<15} = L.Unpooling(n.{}, kernel_h={}, kernel_w={}, stride={}, ntop=1)".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                pool_size[0],
                pool_size[1],
                IR_node.get_attr('strides')[1]))
        else:
            self.add_body(1, "n.{:<15} = L.Unpooling(n.{}, kernel_size={}, stride={}, ntop=1)".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                pool_size[0],
                IR_node.get_attr('strides')[1]))

        # check if need crop output shape
        self.check_if_need_crop(IR_node)

    def emit_ResizeBilinear(self, IR_node):
        shape = IR_node.get_attr("_output_shapes")[0]
        shape = shape_to_list(shape)
        self.add_body(1, "n.{:<15} = L.ResizeBilinear(n.{}, height={}, width={}, ntop=1)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            shape[1],
            shape[2]))

    def emit_UNKNOWN(self, IR_node):
        print(IR_node.IR_layer.name)


    def emit_DataInput(self, IR_node):
        shape = self._shapeToStr(IR_node.get_attr('shape'))
        shape = [shape[0], shape[-1]] + shape[1:-1]
        self.add_body(1, "n.{:<15} = L.Input(shape=[dict(dim={})], ntop=1)".format(
            IR_node.variable_name,
            shape))


    def emit_Dropout(self, IR_node):
        in_place = True
        self.add_body(1, "n.{:<15} = L.Dropout(n.{}, dropout_ratio={} , in_place={}, ntop=1)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            1 - IR_node.get_attr('keep_prob'),
            in_place))


    def emit_FullyConnected(self, IR_node):
        self.add_body(1, "n.{:<15} = L.InnerProduct(n.{}, num_output={}, bias_term={}, ntop=1)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.layer.attr["units"].i,
            IR_node.get_attr('use_bias', False)))
        if self.weight_loaded:
            self.check_if_need_transpose(IR_node)
            self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], (1, 0))
            self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)


    def emit_BatchNorm(self, IR_node):

        self.add_body(1, "n.{:<15} = L.BatchNorm(n.{}, eps={}, use_global_stats={}, ntop=1)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('epsilon'),
            self.phase == 'test'
        ))

        scale_layer_var_name = IR_node.variable_name + "_scale"
        self.add_body(1, "n.{:<15} = L.Scale(n.{}, bias_term={}, in_place=True, ntop=1)".format(
            scale_layer_var_name,
            IR_node.variable_name,
            IR_node.get_attr('bias', False)
        ))

        if self.weight_loaded:
            self.weights_dict[scale_layer_var_name] = dict()
            if 'scale' in self.weights_dict[IR_node.name]:
                self.weights_dict[scale_layer_var_name]['scale'] = self.weights_dict[IR_node.name]['scale']
            else:
                self.weights_dict[scale_layer_var_name]['scale'] = 1

            self.weights_dict[IR_node.name]['scale'] = 1

            if 'bias' in self.weights_dict[IR_node.name]:
                self.weights_dict[scale_layer_var_name]['bias'] = self.weights_dict[IR_node.name]['bias']
                self.weights_dict[IR_node.name].pop('bias', None)
                # change the key "name" to "variable_name", in case of the layer name has invalid characters

            self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)

        IR_node.real_name = IR_node.name + "_scale"


    def emit_Scale(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Scale(n.{}, bias_term={}, in_place=True, ntop=1)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('use_bias', False)
        ))
        if self.weight_loaded:
            self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)


    def emit_Constant(self, IR_node):
        if IR_node.get_attr('value'):
            value = IR_node.get_attr('value')
        else:
            value = self.weights_dict[IR_node.name]['value'][0]
        IR_node_after = self.IR_graph.get_son(IR_node.name, [0])
        shape = IR_node_after.get_attr("_output_shapes")[0]
        shape = shape_to_list(shape)
        if len(shape) == 4:
            shape[1], shape[3] = shape[3], shape[1]
            shape[0] = 1
        shape = list(map(lambda x:str(x), shape))

        self.add_body(1, "n.{:<15} = L.DummyData(shape=[dict(dim=[{}])], data_filler=dict(type='constant', value={}), ntop=1)".format(
            IR_node.variable_name,
            ', '.join(shape),
            value
        ))


    def emit_LRN(self, IR_node):
        output_name = IR_node.variable_name
        input_name = self.parent_variable_name(IR_node)
        size = IR_node.get_attr('size')
        alpha = IR_node.get_attr('alpha')
        beta = IR_node.get_attr('beta')
        bias = IR_node.get_attr('bias')

        self.add_body(1, "n.{:<15} = L.LRN(n.{}, local_size={}, alpha={}, beta={}, k={})".format(
            output_name,
            input_name,
            size,
            alpha,
            beta,
            bias
        ))


    def emit_Add(self, IR_node):
        input_layers = ', '.join(('n.' + self.IR_graph.get_parent(IR_node.name, [num]).real_variable_name) for num in range(0, len(IR_node.in_edges)))
        self.add_body(1, "n.{:<15} = L.Eltwise({}, operation=1, ntop=1)".format(
            IR_node.variable_name,
            input_layers,
        ))

    def emit_Flatten(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Flatten(n.{})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            ))


    def emit_Squeeze(self, IR_node):
        shape = IR_node.get_attr("_output_shapes")[0]
        shape = shape_to_list(shape)
        if shape:
            dim_str = "'dim': {}".format(shape)
            dim_str = " reshape_param={'shape': { " + dim_str + '} }'
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                dim_str
                ))
        else:
            IR_node.real_name = self.IR_graph.get_parent(IR_node.name, [0]).real_name


    def emit_Concat(self, IR_node):
        axis_array = (2, 3, 1, 0)
        axis = axis_array.index(IR_node.get_attr('axis'))
        input_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.in_edges)
        self.add_body(1, "n.{:<15} = L.Concat({}, axis={})".format(
            IR_node.variable_name,
            input_layers,
            axis
        ))

    def emit_Sigmoid(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Sigmoid(n.{}, ntop=1)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node)))


    def emit_Relu(self, IR_node):
        in_place = True
        self.add_body(1, "n.{:<15} = L.ReLU(n.{}, in_place={}, ntop=1)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            in_place))


    def emit_Elu(self, IR_node):
        in_place = True
        self.add_body(1, "n.{:<15} = L.ELU(n.{}, in_place={}, ntop=1)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            in_place))


    def emit_LeakyRelu(self, IR_node):
        in_place = True
        self.add_body(1, "n.{:<15} = L.ReLU(n.{}, in_place={}, negative_slope={}, ntop=1)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            in_place,
            IR_node.IR_layer.attr['alpha'].f))



    def emit_PRelu(self, IR_node):
        in_place = True
        self.add_body(1, "n.{:<15} = L.PReLU(n.{}, in_place={}, ntop=1)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            in_place))

    def emit_Tanh(self, IR_node):
        self.add_body(1, "n.{:<15} = L.TanH(n.{}, ntop=1)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node)))

    def emit_Softmax(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Softmax(n.{}, ntop=1)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node)))


    def emit_Pad(self, IR_node):
        IR_node.real_name = self.IR_graph.get_parent(IR_node.name, [0]).real_name

    def reduction(self, IR_node, op, axes):
        # Convert NHWC (IR) to NCHW (Caffe): [0,1,2,3]->[0,3,1,2]
        if len(axes) == 1:
            assert (axes[0] == 2)
        elif len(axes) == 2:
            assert ((axes[0] == 1) and (axes[1] == 2))

        self.add_body(1, "n.{:<15} = L.Reduction(n.{}, operation={} , axis={} ,ntop=1)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            op,
            len(axes)))

        if IR_node.get_attr('keepdims') == True:
            shape = IR_node.get_attr("_output_shapes")[0]
            shape = shape_to_list(shape)
            shape = [1] + [shape[-1]] + shape[1:-1]
            dim_str = "'dim': {}".format(shape)
            dim_str = "{'shape': { " + dim_str + '} }'
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, reshape_param={}) ".format(
                IR_node.variable_name + "_reshape",
                IR_node.real_variable_name,
                dim_str))
            IR_node.real_name = IR_node.real_name + '_reshape'


    def emit_ReduceMean(self, IR_node):
        self.reduction(IR_node, 4, IR_node.get_attr('axes'))

    def emit_ReduceSum(self, IR_node):
        self.reduction(IR_node, 1, IR_node.get_attr('axes'))

    def get_layer_list(self):
        try:
            from caffe.proto import caffe_pb2
            layer = caffe_pb2.LayerParameter()
            param_list = [f.name for f in layer.DESCRIPTOR.fields if f.name.endswith('_param')]
            layer_list = [type(getattr(layer, s)).__name__ for s in param_list]
            layer_list = [s[:-len('Parameter')] for s in layer_list]
            return layer_list
        except:
            return []

    def clip_exists(self):
        layer_list = self.get_layer_list()
        return 'Clip' in layer_list

    def emit_Relu6Clip(self, IR_node):
        in_place = True
        self.add_body(1, "n.{:<15} = L.Clip(n.{}, min=0, max=6, in_place={}, ntop=1)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            in_place))

    def emit_Relu6(self, IR_node):
        if self.clip_exists():
            self.emit_Relu6Clip(IR_node)
        else:
            self.emit_Relu(IR_node)

    def emit_DepthwiseConv(self, IR_node):
        self.emit_Conv(IR_node)

    def emit_Const(self, IR_node):
        pass
    def emit_Shape(self, IR_node):
        pass
    def emit_Reshape(self, IR_node):
        shape = IR_node.get_attr("_output_shapes")[0]
        shape = shape_to_list(shape)
        if shape:
            dim_str = "'dim': {}".format(shape)
            dim_str = " reshape_param={'shape': { " + dim_str + '} }'
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                dim_str
                ))
        else:
            IR_node.real_name = self.IR_graph.get_parent(IR_node.name, [0]).real_name

    def emit_Slice(self, IR_node):
        pass

    def emit_Pack(self, IR_node):
        pass

    def emit_Abs(self, IR_node):
        self.add_body(1, "n.{:<15} = L.AbsVal(n.{}, ntop=1)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node)))

    def emit_Sub(self, IR_node):
        input_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.in_edges)
        self.add_body(1, "n.{:<15} = L.Eltwise({}, coeff = [1, -1], ntop=1)".format(
            IR_node.variable_name,
            input_layers))

    def emit_Mul(self, IR_node):
        if len(IR_node.in_edges) == 2:
            input_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.in_edges)
            self.add_body(1, "n.{:<15} = L.Eltwise({}, operation=0, ntop=1)".format(
            IR_node.variable_name,
            input_layers))
        elif len(IR_node.in_edges) == 1:
            self.emit_Scale(IR_node)
        else:
            assert False

    def emit_UpSampling2D(self, IR_node):
        scales = IR_node.get_attr('scales')
        scale = tuple(scales)[0]

        shape = IR_node.get_attr('_output_shapes')[0]
        shape = shape_to_list(shape)

        self.add_body(1, "n.{:<15} = L.Deconvolution(n.{}, convolution_param=dict(kernel_size={}, stride={}, pad={}, num_output={}, group={}, bias_term={}), param=[dict(lr_mult=0)], ntop=1)".format(
            IR_node.variable_name,
            IR_node.in_edges[0],
            2 * scale - scale % 2,
            scale,
            int(math.ceil((scale - 1) / 2)),
            shape[-1],
            shape[-1],
            False))

    # def emit_Square(self, IR_node):
    #     input_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.in_edges)
    #     self.add_body(1, "n.{:<15} = L.Square({}, ntop=1)".format(
    #         IR_node.variable_name,
    #         input_layers))


    def emit_Elu(self, IR_node):
        in_place = True
        self.add_body(1, "n.{:<15} = L.ELU(n.{}, in_place={}, ntop=1)".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            in_place))

    def emit_SpaceToDepth(self, IR_node):
        self.add_body(1, "")
