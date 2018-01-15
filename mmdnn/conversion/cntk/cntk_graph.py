#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from mmdnn.conversion.common.DataStructure.graph import GraphNode, Graph


class CntkGraphNode(GraphNode):

    def __init__(self, layer):
        super(CntkGraphNode, self).__init__(layer)


    @property
    def name(self):
        return self.layer.uid


    @property
    def type(self):
        if hasattr(self.layer, 'op_name'):
            return self.layer.op_name
        elif self.layer.is_input:
            return "DataInput"
        else:
            raise NotImplementedError()


    @property
    def cntk_layer(self):
        return self.layer


    def get_attr(self, name, default_value=None):
        return self.layer.attributes[name]


class CntkGraph(Graph):

    def __init__(self, model):
        # sanity check.
        pass

        self.weights = dict()
        self.visited = set()
        super(CntkGraph, self).__init__(model)


    def _traverse_graph(self, son_node):
        if not son_node.uid in self.visited:
            self.visited.add(son_node.uid)

            if son_node.is_block:
                raise NotImplementedError("Cntk parser block node is not implemented.")

            for input_node in son_node.inputs:
                if input_node.is_output:
                    input_node = input_node.owner
                    if not input_node.uid in self.layer_map:
                        self.layer_map[input_node.uid] = CntkGraphNode(input_node)
                    self._make_connection(input_node.uid, son_node.uid)
                    self._traverse_graph(input_node)

                elif input_node.is_input:
                    if not input_node.uid in self.layer_map:
                        self.layer_map[input_node.uid] = CntkGraphNode(input_node)
                    self._make_connection(input_node.uid, son_node.uid)

                else:
                    if not input_node.name in self.weights:
                        # print ("Warning: node {} is not found.".format(input_node.name))
                        pass


    def build(self):
        for param in self.model.parameters:
            self.weights[param.name] = param.asarray()
            # print (param.name, self.weights[param.name].shape)

        for output in self.model.outputs:
            output = output.owner
            self.layer_map[output.uid] = CntkGraphNode(output)
            self._traverse_graph(output)

        super(CntkGraph, self).build()
