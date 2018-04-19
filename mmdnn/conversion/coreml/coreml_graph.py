#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
import os

import coremltools
from mmdnn.conversion.common.DataStructure.graph import GraphNode, Graph


class CoremlGraphNode(GraphNode):

    def __init__(self, layer):
        super(CoremlGraphNode, self).__init__(layer)


    @property
    def name(self):
        return self.layer.name


    @property
    def type(self):
        return self.layer.__class__.__name__


    @property
    def coreml_layer(self):
        return self.layer



class CoremlGraph(Graph):

    def __init__(self, model):
        from coremltools.proto import Model_pb2

        # sanity check.
        if not isinstance(model, Model_pb2.Model):
            raise TypeError("Coreml layer of type %s is not supported." % type(model))
        super(CoremlGraph, self).__init__(model)
        self.model = model


    def build(self):
        self.input_layers = list()

        # input layer

        for layer in self.model.description.input:
            self.layer_map[layer.name] = CoremlGraphNode(layer)
            self.layer_name_map[layer.name] = layer.name


        model_type = self.model.WhichOneof('Type')
        if model_type == 'neuralNetworkClassifier':
            # build each layer
            for layer in self.model.neuralNetworkClassifier.layers:
                self.layer_map[layer.name] = CoremlGraphNode(layer)
                self.layer_name_map[layer.name] = layer.name

            # if A.output == B.input, then make the connection: A -> B
            for layerA in self.model.neuralNetworkClassifier.layers:
                for layerB in self.model.neuralNetworkClassifier.layers:
                    for A in layerA.output:
                        for B in layerB.input:
                            if A == B :
                                # print('{0:20}->     {1:20}'.format(layerA.name, layerB.name))
                                self._make_connection(layerA.name, layerB.name)

            # if A.name == B.input, then make the connection: A -> B, here A is the input
            for layerA in self.model.description.input:
                for layerB in self.model.neuralNetworkClassifier.layers:
                    for B in layerB.input:
                        if layerA.name == B:
                            self._make_connection(layerA.name, layerB.name)
        elif model_type == 'neuralNetwork':
            # build each layer
            for layer in self.model.neuralNetwork.layers:
                self.layer_map[layer.name] = CoremlGraphNode(layer)
                self.layer_name_map[layer.name] = layer.name

            # if A.output == B.input, then make the connection: A -> B
            for layerA in self.model.neuralNetwork.layers:
                for layerB in self.model.neuralNetwork.layers:
                    for A in layerA.output:
                        for B in layerB.input:
                            if A == B :
                                # print('{0:20}->     {1:20}'.format(layerA.name, layerB.name))
                                self._make_connection(layerA.name, layerB.name)
            # if A.name == B.input, then make the connection: A -> B, here A is the input
            for layerA in self.model.description.input:
                for layerB in self.model.neuralNetwork.layers:
                    for B in layerB.input:
                        if layerA.name == B:
                            self._make_connection(layerA.name, layerB.name)
        elif model_type == 'neuralNetworkRegressor':
            # build each layer
            for layer in self.model.neuralNetworkRegressor.layers:
                self.layer_map[layer.name] = CoremlGraphNode(layer)
                self.layer_name_map[layer.name] = layer.name

            # if A.output == B.input, then make the connection: A -> B
            for layerA in self.model.neuralNetworkRegressor.layers:
                for layerB in self.model.neuralNetworkRegressor.layers:
                    for A in layerA.output:
                        for B in layerB.input:
                            if A == B :
                                # print('{0:20}->     {1:20}'.format(layerA.name, layerB.name))
                                self._make_connection(layerA.name, layerB.name)
            # if A.name == B.input, then make the connection: A -> B, here A is the input
            for layerA in self.model.description.input:
                for layerB in self.model.neuralNetworkRegressor.layers:
                    for B in layerB.input:
                        if layerA.name == B:
                            self._make_connection(layerA.name, layerB.name)
        else:
            assert False



            # The information of the layer
        super(CoremlGraph, self).build()



