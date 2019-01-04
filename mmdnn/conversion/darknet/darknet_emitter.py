#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import division

import os
import sys
import math
import numpy as np

from mmdnn.conversion.common.IR.IR_graph import IRGraph, IRGraphNode
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.DataStructure.emitter import Emitter
from mmdnn.conversion.common.utils import *


class DarknetEmitter(Emitter):

    def __init__(self, model):
        from six import string_types as _string_types
        super(DarknetEmitter, self).__init__()
        if isinstance(model, _string_types):
            network_path = model
        else:
            network_path = model[0]
            self._load_weights(model[1])

        self.IR_graph = IRGraph(network_path)
        super(DarknetEmitter, self)._build()

    @property
    def header_code(self):
        return """from __future__ import print_function
import numpy as np
import sys, argparse
from six import text_type as _text_type


__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file = None):
    n = []
"""

    @property
    def end_code(self):
        return """    return n
            
def save_cfg(blocks, cfg):
    with open(cfg, 'w') as fp:
        idx = -1
        for block in blocks:
            if 'name' in block:
                fp.write('# %d: %s\\n' % (idx, block['name']))
                pass
            fp.write('[%s]\\n' % (block['type']))
            for key,value in block.items():
                if value == None:
                    continue
                if key not in ('type', 'name'):
                    fp.write('%s=%s\\n' % (key, value))
            fp.write('\\n')

            idx = idx + 1

def gen_weight(weight_file, weights_output):
    global __weights_dict
    __weights_dict = load_weights(weight_file)

    weights_data = []

    model = KitModel()

    for layer in model:
        key = layer.get('name', None)
        if key == None or key not in __weights_dict:
            continue

        weights = __weights_dict[key]

        if 'bias' in weights:
            weights_data += list(weights['bias'].flat)
        if 'weights' in weights:
            weights_data += list(weights['weights'].flat)
        if 'gamma' in weights: # used for prelu, not sure if other layers use this too
            weights_data += list(weights['gamma'].flat)

    data = np.array(weights_data)
    wsize = data.size
    weights = np.zeros((wsize+4,), dtype=np.int32)
    ## write info
    weights[0] = 0
    weights[1] = 1
    weights[2] = 0      ## revision
    weights[3] = 0      ## net.seen
    weights.tofile(weights_output)
    weights = np.fromfile(weights_output, dtype=np.float32)
    weights[4:] = data
    weights.tofile(weights_output)
    
def make_net(cfg):
    save_cfg(KitModel(), cfg)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate darknet model')
    parser.add_argument('--weight_file', '-w', type=_text_type, default='IR weight file')
    parser.add_argument('--cfg', '-c', type=_text_type, default='darknet_converted.cfg')
    parser.add_argument('--weights', '-w', type=_text_type, default='darknet_converted.weights')
    args = parser.parse_args()

    # For some reason argparser gives us unicode, so we need to conver to str first
    make_net(str(args.cfg))
    gen_weight(str(args.weight_file), str(args.weights))

"""

    def fork(self, IR_node):
        inputs = [self.layer_id_by_name.get(self.IR_graph.get_node(edge).name, self.id - 1) for edge in IR_node.in_edges]
        if any((id != self.id - 1 for id in inputs)):
            input_layers = ','.join((str(id - self.id) for id in inputs))
            self.add_body(1, "n.append({{'name': '{}_route', 'type': 'route', 'layers': '{}'}})".format(
                IR_node.name,
                input_layers
            ))
            self.layer_id_by_name[IR_node.name] = self.id
            self.id += 1

    def gen_code(self, phase = 'test'):
        self.phase = phase
        self.add_body(0, self.header_code)

        self.id = 0
        self.layer_id_by_name = {}
        for layer in self.IR_graph.topological_sort:
            current_node = self.IR_graph.get_node(layer)
            node_type = current_node.type

            if hasattr(self, "emit_" + node_type):
                func = getattr(self, "emit_" + node_type)
                func(current_node)
            else:
                print("DarknetEmitter has not supported operator [%s]." % (node_type))
                self.emit_UNKNOWN(current_node)

        self.add_body(0, "")
        self.add_body(0,self.end_code)

        return self.body_code


    def run(self, dstNetworkPath, dstWeightPath = None, phase = 'test'):
        super(DarknetEmitter, self).run(dstNetworkPath, dstWeightPath, phase)
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
        else:
            pad_h = pads[1] + (0 if pads[1] == pads[5] else stride_h)
            pad_w = pads[2] + (0 if pads[2] == pads[6] else stride_w)

        return pad_h, pad_w

    def emit_Conv(self, IR_node):
        self.fork(IR_node)

        if IR_node.type == "DepthwiseConv":
            raise ValueError()

        # implement asymmetric paddings by applying symmetric padding then cropping
        pad_h, pad_w = self._get_symmetric_padding(IR_node)
        if pad_h != pad_w:
            raise ValueError()
        padding = pad_h

        kernel_h = IR_node.get_attr('kernel_shape')[0]
        kernel_w = IR_node.get_attr('kernel_shape')[1]
        if kernel_h != kernel_w:
            raise ValueError()
        size = kernel_h

        num_output = IR_node.get_attr('kernel_shape')[-1]

        outputs = [self.IR_graph.get_node(edge) for edge in IR_node.out_edges]
        activation = 'linear'
        activation_param = None
        for output in outputs:
            if output.type == 'Relu':
                activation = 'relu'
            elif output.type == 'LeakyRelu':
                activation = 'leaky'
                activation_param = output.IR_layer.attr['alpha'].f
            elif output.type == 'PRelu':
                activation = 'prelu'

        self.add_body(1, "n.append({{'name': '{}', 'type': 'convolutional', 'batch_normalize': 0, 'filters': {}, 'size': {}, 'stride': {}, 'padding': {}, 'activation': '{}', 'activation_param': {}}})".format(
            IR_node.name,
            num_output,
            size,
            IR_node.get_attr('strides')[1],
            padding,
            activation,
            activation_param
        ))

        dim = len(IR_node.get_attr('strides')) - 2
        if self.weight_loaded:
            self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], [dim + 1, dim] + list(range(0, dim)))
            self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)

        self.layer_id_by_name[IR_node.name] = self.id
        self.id += 1

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
            else:
                h_o = (h_i + 2 * pad_h - kernel_h) // stride_h + 1
                w_o = (w_i + 2 * pad_w - kernel_w) // stride_w + 1
            return h_o, w_o
        else:
            assert False


    def emit_Pool(self, IR_node):
        pooling_type = IR_node.get_attr('pooling_type')
        if pooling_type == 'MAX':
            pooling_type = 'maxpool'
        elif pooling_type == 'AVG':
            pooling_type = 'avgpool'
        else:
            raise ValueError()

        if IR_node.layer.attr['global_pooling'].b:
            raise ValueError()

        pad_h, pad_w = self._get_symmetric_padding(IR_node)
        if pad_h != pad_w:
            raise ValueError()
        padding = pad_h
        self.add_body(1, "n.append({{'name': '{}', 'type': '{}', 'size': {}, 'stride': {}, 'padding': {}}})".format(
            IR_node.name,
            pooling_type,
            IR_node.get_attr('kernel_shape')[1],
            IR_node.get_attr('strides')[1],
            padding,
        ))

        self.layer_id_by_name[IR_node.name] = self.id
        self.id += 1

    def emit_UNKNOWN(self, IR_node):
        print(IR_node.IR_layer.name)

    def emit_DataInput(self, IR_node):
        shape = self._shapeToStr(IR_node.get_attr('shape'))

        self.add_body(1, "n.append({{'type': 'net', 'width': {}, 'height': {}, 'batch': {}, 'channels': {}}})".format(
            shape[2],
            shape[1],
            shape[0],
            shape[3]))

    def emit_Concat(self, IR_node):
        input_layers = ','.join((str(self.layer_id_by_name[self.IR_graph.get_node(edge).name] - self.id) for edge in IR_node.in_edges))
        self.add_body(1, "n.append({{'name': '{}', 'type': 'route', 'layers': '{}'}})".format(
            IR_node.name,
            input_layers
        ))
        self.layer_id_by_name[IR_node.name] = self.id
        self.id += 1

    def emit_Softmax(self, IR_node):
        self.fork(IR_node)

        self.add_body(1, "n.append({{'name': '{}', 'type': 'softmax'}})".format(
            IR_node.name,
        ))

        self.layer_id_by_name[IR_node.name] = self.id
        self.id += 1

    def emit_Squeeze(self, IR_node):
        self.layer_id_by_name[IR_node.name] = self.layer_id_by_name[self.IR_graph.get_node(IR_node.in_edges[0]).name]

    def emit_Flatten(self, IR_node):
        self.add_body(1, "n.append({{'name': '{}', 'type': 'reorg', 'flatten': 1}})".format(
            IR_node.name
        ))

        self.layer_id_by_name[IR_node.name] = self.id
        self.id += 1

    def emit_Dropout(self, IR_node):
        self.add_body(1, "n.append({{'name': '{}', 'type': 'dropout', 'probability': {}}})".format(
            IR_node.name,
            1 - IR_node.get_attr('keep_prob'),
        ))

        self.layer_id_by_name[IR_node.name] = self.id
        self.id += 1

    def emit_FullyConnected(self, IR_node):
        self.fork(IR_node)

        outputs = [self.IR_graph.get_node(edge) for edge in IR_node.out_edges]
        activation = 'linear'
        activation_param = None
        for output in outputs:
            if output.type == 'Relu':
                activation = 'relu'
            elif output.type == 'LeakyRelu':
                activation = 'leaky'
                activation_param = output.IR_layer.attr['alpha'].f
            elif output.type == 'PRelu':
                activation = 'prelu'

        self.add_body(1, "n.append({{'name': '{}', 'type': 'connected', 'activation': '{}', 'activation_param': {}, 'output': {}}})".format(
            IR_node.name,
            activation,
            activation_param,
            IR_node.layer.attr["units"].i,
        ))

        self.layer_id_by_name[IR_node.name] = self.id
        self.id += 1

        if self.weight_loaded:
            self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], (1, 0))
            self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)

    def merge_with_last_layer(self, IR_node):
        parent_node = self.IR_graph.get_parent(IR_node.name, [0])
        if parent_node.type in ['Conv', 'FullyConnected']:
            self.layer_id_by_name[IR_node.name] = self.layer_id_by_name[parent_node.name]
            return True

        return False

    def emit_Relu(self, IR_node):
        if not self.merge_with_last_layer(IR_node):
            self.add_body(1, "n.append({{'name': '{}', 'type': 'activation', 'activation': 'relu'}})".format(
                IR_node.name
            ))

            self.layer_id_by_name[IR_node.name] = self.id
            self.id += 1

    def emit_LeakyRelu(self, IR_node):
        if not self.merge_with_last_layer(IR_node):
            self.add_body(1, "n.append({{'name': '{}', 'type': 'activation', 'activation': 'leaky'}})".format(
                IR_node.name
            ))

            self.layer_id_by_name[IR_node.name] = self.id
            self.id += 1

    def emit_PRelu(self, IR_node):
        if not self.merge_with_last_layer(IR_node):
            # TODO: Standalone activation layer
            raise ValueError()
