#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
import os
import paddle.v2 as paddle
import paddle.trainer_config_helpers.layers as layers
from mmdnn.conversion.common.DataStructure.graph import GraphNode, Graph


class PaddleGraphNode(GraphNode):

    def __init__(self, layer):
        super(PaddleGraphNode, self).__init__(layer)


    @property
    def name(self):
        return self.layer.full_name


    @property
    def type(self):
        return self.layer.layer_type


    @property
    def paddle_layer(self):
        return self.layer



class PaddleGraph(Graph):

    def __init__(self, model):
        # sanity check.
        if not type(model) == layers.LayerOutput:
            raise TypeError("PaddlePaddle layer of type %s is not supported." % type(model))
        super(PaddleGraph, self).__init__(model)
        self.model = model


    def build(self):
        self.input_layers = list()
        nodes = list()
        outs = list()
        layer = self.model
        self.layer_map[layer.full_name] = PaddleGraphNode(layer)
        self.layer_name_map[layer.full_name] = layer.full_name
        nodes.append(layer.full_name)
        outs.append(layer)
        while outs:
            new_outs = list()
            for out in outs:
                for layer in out.parents:
                    if layer.full_name not in nodes:
                        self.layer_map[layer.full_name] = PaddleGraphNode(layer)
                        self.layer_name_map[layer.full_name] = layer.full_name
                        self._make_connection(layer.full_name, out.full_name)
                        new_outs.append(layer)
                    if not layer.parents:
                        self.input_layers.append(layer)
            outs = new_outs


        super(PaddleGraph, self).build()