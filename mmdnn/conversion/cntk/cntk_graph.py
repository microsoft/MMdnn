#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import cntk as _cntk
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
        if self.layer.is_block:
            return self.layer.block_root.attributes[name]
        else:
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
                inputs = [input for _, input in son_node.block_arguments_mapping]

            else:
                inputs = son_node.inputs

            for input_node in inputs:
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

                elif input_node.is_placeholder:
                    raise NotImplementedError("PlaceHolder of placeholder is not supported.")


    def build(self):
        if len(self.model.outputs) > 1:
            for idx, output in enumerate(self.model.outputs):
                if len(output.shape) > 0:
                    eval_node = idx
                    break

            output = self.model[eval_node].owner
        else:
            output = self.model.outputs[0].owner

        self.layer_map[output.uid] = CntkGraphNode(output)
        self._traverse_graph(output)

        super(CntkGraph, self).build()

"""
    def __traverse_graph(self, node):
        if node.uid in self.visited:
            return

        self.visited.add(node.uid)

        if isinstance(node, _cntk.Function) and node.is_block:
            composite = node.block_root

            # BlockFunction node
            mapping = node.block_arguments_mapping

            # redirect the composite's inputs to the true inputs
            stack.extend([(actual_input, depth-1) for _, actual_input in mapping]) # traverse into actual composite inputs
            visited |= {comp_input.uid for comp_input, _ in mapping}    # don't traverse into the mapped-away inputs
            stack.append((composite, depth-1))
            # BlockFunctions are short-circuited, and not added to accum[]
        try:
            # Function node
            stack = list((i, depth) for i in node.root_function.inputs) + stack
        except AttributeError:
            # OutputVariable node
            try:
                if node.is_output:
                    stack.insert(0, (node.owner, depth))
                    visited.add(node.uid)
                    continue
            except AttributeError:
                pass

        if visitor(node):
            if isinstance(node, Variable):
                if node.is_parameter:
                    node = node.as_parameter()
                elif node.is_constant:
                    node = node.as_constant()

            accum.append(node)

        visited.add(node.uid)


    # def build(self):
    #     _traverse_graph(self, self.model.root_function)
    #     super(CntkGraph, self).build()
"""