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
        return self.layer.name


    @property
    def type(self):
        return self.layer.type


    @property
    def paddle_layer(self):
        return self.layer



class PaddleGraph(Graph):

    def __init__(self, model):
        from paddle.proto import ModelConfig_pb2
        # sanity check.
        if not isinstance(model, ModelConfig_pb2.ModelConfig):
            raise TypeError("PaddlePaddle layer of type %s is not supported." % type(model))
        super(PaddleGraph, self).__init__(model)
        self.model = model


    def build(self):
        self.input_layers = list()
        for layer in self.model.layers:
            self.layer_map[layer.name] = PaddleGraphNode(layer)
            self.layer_name_map[layer.name] = layer.name

            for input_layer in layer.inputs:
                self._make_connection(input_layer.input_layer_name, layer.name)

        super(PaddleGraph, self).build()