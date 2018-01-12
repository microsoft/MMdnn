#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from mmdnn.conversion.common.DataStructure.graph import GraphNode, Graph
from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow.core.framework import attr_value_pb2
import cntk as _cntk


class CntkGraphNode(GraphNode):

    def __init__(self, layer):
        super(CntkGraphNode, self).__init__(layer)


    @property
    def name(self):
        return self.layer.name


    @property
    def type(self):
        return self.layer.op_name


    @property
    def cntk_layer(self):
        return self.layer


    def get_attr(self, name, default_value=None):
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


class CntkGraph(Graph):

    def __init__(self, model):
        # sanity check.
        pass

        self.weights = dict()
        super(CntkGraph, self).__init__(model)


    def _traverse_graph(self, son_node):
        if not son_node.name in self.layer_map:
            self.layer_map[son_node.name] = CntkGraphNode(son_node)
            for input_node in son_node.inputs:
                print (type(input_node))
                if input_node.is_output:
                    print ("Kit", input_node.owner)
                else:
                    assert input_node.name in self.weights
                #     print ("CC", input_node.kind)
                #     print (input_node.kind.__class__.__name__)

            # print (dir(son_node))
            # for idx, layer in enumerate(self.model.inputs):
            #     print (idx, layer)

    def build(self):
        print (self.model.parameters)
        for param in self.model.parameters:
            print (param.name)
            self.weights[param.name] = param.asarray()
            # print (dir(param))
            # assert False

        # assert False
        for output in self.model.outputs:
            self._traverse_graph(output.owner)
        assert False
        # for i, layer in enumerate(self.model.node):
        #     self.layer_map[layer.name] = CntkGraphNode(layer)

        # for i, layer in enumerate(self.model.node):
        #     self.layer_map[layer.name] = CntkGraphNode(layer)
        #     self.layer_name_map[layer.name] = layer.name
        #     for pred in layer.input:
        #         if pred not in self.layer_map:
        #             new_node = NodeDef()
        #             new_node.name = pred
        #             new_node.op = "NoOp"
        #             self.layer_map[pred] = CntkGraphNode(new_node)
        #             self.layer_name_map[pred] = pred

        #         self._make_connection(pred, layer.name)

        super(CntkGraph, self).build()