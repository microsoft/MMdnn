#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import os
import mxnet as mx
from mmdnn.conversion.common.DataStructure.graph import GraphNode, Graph

class MXNetGraphNode(GraphNode):



    def __init__(self, layer):
        super(MXNetGraphNode, self).__init__(layer)

        if "attr" in layer:
            self.attr = layer["attr"]
        elif "param" in layer:
            self.attr = layer["param"]
        elif "attrs" in layer:
            self.attr = layer["attrs"]
        else:
            self.attr = None


    @property
    def name(self):
        return self.layer["name"]


    @property
    def type(self):
        return self.layer["op"]


    @property
    def mx_layer(self):
        return self.layer


    def get_attr(self, name, default_value=None):
        assert self.attr
        return self.attr.get(name, default_value)


class MXNetGraph(Graph):

    def __init__(self, model):
        # sanity check non-sense always input module.Module
        # if not (type(model) == mx.module.Module
        #     or type(model) == mx.module.SequentialModule
        #     or type(model) == mx.model)
        #     raise TypeError("MXNet layer of type %s is not supported." % type(model))

        super(MXNetGraph, self).__init__(model)


    def build(self, json_data):

        self.input_layers = list()
        input_dict = dict() # dict{layer_num, layer_name}
        layer_num = -1

        import re

        for layer in json_data:

            layer_num += 1
            # if layer["op"] == "null":
            #     continue

            if re.search("_(weight|bias|var|mean|gamma|beta|label)", layer["name"]) and layer["op"] == "null":
                continue

            input_dict.update({layer_num: layer["name"]})
            self.layer_map[layer["name"]] = MXNetGraphNode(layer)
            self.layer_name_map[layer["name"]] = layer["name"]
            for input_layer in layer["inputs"]:
                assert isinstance(input_layer, list)
                if input_layer[0] in input_dict:
                    pred = input_dict.get(input_layer[0])

                    if pred not in self.layer_map:
                        new_node = dict({'op': 'NoOp', 'name': pred, 'inputs': list()})
                        self.layer_map[pred] = MXNetGraphNode(new_node)
                        self.layer_name_map[pred] = pred

                    self._make_connection(pred, layer["name"])

        super(MXNetGraph, self).build()

        # raise NotImplementedError("Cannot support multi-input")