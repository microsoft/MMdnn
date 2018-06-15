#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import numpy as np
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType


class Parser(object):

    def __init__(self):
        self.IR_graph = GraphDef()
        self.weight_loaded = False

        # name --> (weight_name --> ndarray)
        self.weights = dict()


    def run(self, dest_path):
        self.gen_IR()
        self.save_to_json(dest_path + ".json")
        self.save_to_proto(dest_path + ".pb")
        self.save_weights(dest_path + ".npy")


    @property
    def src_graph(self):
        raise NotImplementedError


    def get_son(self, name, path, set_flag = False):
        return self.src_graph.get_son(name, path, set_flag)


    def get_parent(self, name, path, set_flag = False):
        return self.src_graph.get_parent(name, path, set_flag)


    def set_weight(self, layer_name, weight_name, data):
        if not layer_name in self.weights:
            self.weights[layer_name] = dict()
        layer = self.weights[layer_name]
        layer[weight_name] = data


    def save_to_json(self, filename):
        import google.protobuf.json_format as json_format
        json_str = json_format.MessageToJson(self.IR_graph, preserving_proto_field_name = True)

        with open(filename, "w") as of:
            of.write(json_str)

        print ("IR network structure is saved as [{}].".format(filename))

        return json_str


    def save_to_proto(self, filename):
        proto_str = self.IR_graph.SerializeToString()
        with open(filename, 'wb') as of:
            of.write(proto_str)

        print ("IR network structure is saved as [{}].".format(filename))

        return proto_str


    def save_weights(self, filename):
        if self.weight_loaded:
            with open(filename, 'wb') as of:
                np.save(of, self.weights)
            print ("IR weights are saved as [{}].".format(filename))

        else:
            print ("Warning: weights are not loaded.")


    def convert_inedge(self, source_node, IR_node, start_idx = 0, end_idx = None):
        if end_idx == None: end_idx = len(source_node.in_edges)
        for idx in range(start_idx, end_idx):
            IR_node.input.append(self.src_graph.get_node(source_node.in_edges[idx]).real_name)


    @staticmethod
    def channel_first_conv_kernel_to_IR(tensor):
        dim = tensor.ndim
        tensor = np.transpose(tensor, list(range(2, dim)) + [1, 0])
        return tensor


    @staticmethod
    def channel_first_shape_to_IR(shape):
        return [shape[0]] + list(shape[2:]) + [shape[1]]

    @staticmethod
    def channel_first_axis_to_IR(index):
        if index == 0:
            return 0
        elif index == 1:
            return -1
        else:
            return index - 1
