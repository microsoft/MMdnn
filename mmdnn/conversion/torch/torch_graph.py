#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from mmdnn.conversion.common.DataStructure.graph import GraphNode, Graph
from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow.core.framework import attr_value_pb2
import torch


class TorchGraphNode(GraphNode):

    def __init__(self, layer, id):
        # self._type = layer.__class__.__name__.replace('Backward', '')
        # self._name = "{}_{}".format(self.type, id)
        # TODO
        super(PyTorchGraphNode, self).__init__(layer)

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type


    @property
    def torch_layer(self):
        return self.layer


class TorchGraph(Graph):

    def __init__(self, model):
        super(TorchGraph, self).__init__(model)
        self.model = model


    def build(self, shape):
        print (self.model)
        print (dir(self.model))

        output_shapes = self._infer_torch_output_shapes(
            self.model,
            shape
        )
        print (output_shapes)

        # """
        # build graph for pytorch 0.2.0
        # """
        # dummy_input = torch.autograd.Variable(torch.randn(shape))
        # output_node = self.model(dummy_input)

        # search_queue = [output_node.grad_fn]
        # tmp_node = PyTorchGraphNode(output_node.grad_fn, 0)
        # self.layer_map[tmp_node.name] = tmp_node
        # visited = {output_node.grad_fn : self.layer_map[tmp_node.name]}

        # idx = 0
        # node_count = 1
        # while (idx < len(search_queue)):
        #     current_node = search_queue[idx]
        #     current_type = visited[current_node].type
        #     if hasattr(current_node, 'next_functions'):
        #         for parent, _ in current_node.next_functions:
        #             parent_type = parent.__class__.__name__.replace('Backward', '')
        #             if parent_type != 'AccumulateGrad' and \
        #                (parent_type != 'Transpose' or current_type != 'Addmm'):
        #                 if not parent in visited:
        #                     tmp_node = PyTorchGraphNode(parent, node_count)
        #                     self.layer_map[tmp_node.name] = tmp_node
        #                     node_count += 1
        #                     visited[parent] = tmp_node
        #                     search_queue.append(parent)
        #                 self._make_connection(visited[parent].name, visited[current_node].name)
        #     idx += 1

        super(TorchGraph, self).build()


    @staticmethod
    def _infer_torch_output_shapes(torch_model, input_shapes):
        """
        Forward torch model to infer output shape
        """
        return TorchGraph._forward_torch_random_input(
                torch_model,
                input_shapes,
                is_batch=False)

        # try:
        #     return TorchGraph._forward_torch_random_input(
        #         torch_model,
        #         input_shapes,
        #         is_batch=False
        #     )
        # except:
        #     # try batch mode
        #     # return TorchGraph._forward_torch_random_input(
        #     #     torch_model,
        #     #     input_shapes,
        #     #     is_batch=True
        #     # )
        #     pass

    @staticmethod
    def _forward_torch_random_input(torch_model, input_shapes, is_batch=False):
        input_tensors = []
        for shape in input_shapes:
            if is_batch:
                tensor = torch.rand(1, *shape).float()
            else:
                tensor = torch.randn(shape)
                # tensor = torch.rand(*shape).float()
            input_tensors.append(tensor)

        print (input_tensors[0].shape)
        if len(input_tensors) == 1:
            result = torch_model.forward(input_tensors[0])
        else:
            result = torch_model.forward(input_tensors)

        print ("result", result)
        if isinstance(result, list):
            # multi output
            output_shapes = []
            for tensor in result:
                shape = tensor.numpy().shape
                if is_batch:
                    shape = shape[1:]
                output_shapes.append(shape)
            return output_shapes
        else:
            # single output
            output_shape = result.numpy().shape
            if is_batch:
                return [output_shape[1:]]
            else:
                return [output_shape]