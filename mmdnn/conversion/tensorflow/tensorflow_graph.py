#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from mmdnn.conversion.common.DataStructure.graph import GraphNode, Graph
from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow.core.framework import attr_value_pb2


class TensorflowGraphNode(GraphNode):

    def __init__(self, layer):
        super(TensorflowGraphNode, self).__init__(layer)


    @property
    def name(self):
        return self.layer.name


    @property
    def type(self):
        return self.layer.op


    @property
    def tf_layer(self):
        return self.layer


    def get_attr(self, name, default_value = None):
        if name in self.layer.attr:
            attr = self.layer.attr[name]
            field = attr.WhichOneof('value')
            val = getattr(attr, field) if field else default_value
            if isinstance(val, attr_value_pb2.AttrValue.ListValue):
                return list(val.ListFields()[0][1])
            else:
                return val.decode('utf-8') if isinstance(val, bytes) else val
        else:
            return default_value


class TensorflowGraph(Graph):

    def __init__(self, model):
        # sanity check.
        pass

        super(TensorflowGraph, self).__init__(model)
        self.model = model


    def build(self):
        for i, layer in enumerate(self.model.node):
            self.layer_map[layer.name] = TensorflowGraphNode(layer)

        for i, layer in enumerate(self.model.node):
            self.layer_map[layer.name] = TensorflowGraphNode(layer)
            self.layer_name_map[layer.name] = layer.name
            for pred in layer.input:
                if pred not in self.layer_map:
                    new_node = NodeDef()
                    new_node.name = pred
                    new_node.op = "NoOp"
                    self.layer_map[pred] = TensorflowGraphNode(new_node)
                    self.layer_name_map[pred] = pred

                self._make_connection(pred, layer.name)

        super(TensorflowGraph, self).build()