# ----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

from mmdnn.conversion.common.DataStructure.graph import GraphNode, Graph
from onnx import onnx_pb2


class ONNXGraphNode(GraphNode):
    def __init__(self, layer):
        super(ONNXGraphNode, self).__init__(layer)
        self.weights = list()
        self.inputs = list()
        self.outputs = list()

    @property
    def name(self):
        return self.layer.name

    @property
    def type(self):
        return self.layer.op_type

    @property
    def onnx_layer(self):
        return self.layer


# node
#  input
#   edge(node a <-> node b)
#

class ONNXGraph(Graph):
    @staticmethod
    def _generate_name(layer):
        return ""

    def __init__(self, model):
        super(ONNXGraph, self).__init__(model)
        self._graph = model.graph
        # key is edge name, value is src/dst node name
        self._edge_src = dict()
        self._edge_dst = dict()
        # key is initializer name, value is TensorProto
        self._weights = dict()
        self._inputs = dict()
        self._outputs = dict()

    def build(self):
        for w in self._graph.initializer:
            self._weights[w.name] = w
        for s in self._graph.input:
            self._inputs[s.name] = s
        for s in self._graph.output:
            self._outputs[s.name] = s

        for i, layer in enumerate(self._graph.node):
            if not layer.name:
                layer.name = '{0}_{1}'.format(layer.op_type, i)
            name = layer.name
            # print(name)
            # print(layer.op_type)
            node = ONNXGraphNode(layer)
            self.layer_map[name] = node
            self.layer_name_map[name] = name
            for n in layer.input:
                if n in self._weights:
                    # n is input data
                    node.weights.append(n)
                if n in self._inputs:
                    node.inputs.append(n)
                else:
                    # n is input edge
                    self._edge_dst[n] = name
                    if n in self._edge_src:
                        self._make_connection(self._edge_src[n], name)
            for n in layer.output:
                if n in self._outputs:
                    node.outputs.append(n)
                else:
                    self._edge_src[n] = name
                    if n in self._edge_dst:
                        self._make_connection(name, self._edge_dst[n])

        super(ONNXGraph, self).build()
