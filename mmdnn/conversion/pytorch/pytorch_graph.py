#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from mmdnn.conversion.common.DataStructure.graph import GraphNode, Graph
import torch
import torch.autograd
import torch.serialization
import contextlib
from torch.jit import _unique_state_dict

class scope_name_workaround(object):
    def __init__(self):
        self.backup = None

    def __enter__(self):
        def _tracing_name(self_, tracing_state):
            if not tracing_state._traced_module_stack:
                return None
            module = tracing_state._traced_module_stack[-1]
            for name, child in module.named_children():
                if child is self_:
                    return name
            return None

        def _slow_forward(self_, *input, **kwargs):
            tracing_state = torch._C._get_tracing_state()
            if not tracing_state or isinstance(self_.forward, torch._C.ScriptMethod):
                return self_.forward(*input, **kwargs)
            if not hasattr(tracing_state, '_traced_module_stack'):
                tracing_state._traced_module_stack = []
            name = _tracing_name(self_, tracing_state)
            if name:
                tracing_state.push_scope('%s[%s]' % (self_._get_name(), name))
            else:
                tracing_state.push_scope(self_._get_name())
            tracing_state._traced_module_stack.append(self_)
            try:
                result = self_.forward(*input, **kwargs)
            finally:
                tracing_state.pop_scope()
                tracing_state._traced_module_stack.pop()
            return result

        self.backup = torch.nn.Module._slow_forward
        setattr(torch.nn.Module, '_slow_forward', _slow_forward)

    def __exit__(self, type, value, tb):
        setattr(torch.nn.Module, '_slow_forward', self.backup)

class PytorchGraphNode(GraphNode):

    def __init__(self, layer):
        self.version = torch.__version__
        import re
        self._kind = layer.kind()
        node_id = re.search(r"[\d]+", layer.__str__())
        self.id = node_id.group(0)
        super(PytorchGraphNode, self).__init__(layer)
        self.attrs = {k : layer[k] for k in layer.attributeNames()}



    @property
    def name(self):
        name = self._name + self.id
        # Scopes created in a nested scope may have initial characters
        # that are illegal as the initial character of an op name
        # (viz. '-', '\', '/', and '_').
        name = name.replace('-','n').replace('\\','n').replace('/','n').replace('_','n').replace('[','n').replace(']','n')
        return name

    @property
    def type(self):
        return self._kind

    @property
    def pytorch_layer(self):
        return self.layer


class PytorchGraphNode040(PytorchGraphNode):
    def __init__(self, layer):
        self._name = layer.scopeName()
        import re
        self.weights_name = '.'.join(
                re.findall(r'\[([\w\d.]+)\]', self._name)
            )
        super(PytorchGraphNode040, self).__init__(layer)


class PytorchGraphNode151(PytorchGraphNode):

    def __init__(self, layer):
        self._name = 'node'
        super(PytorchGraphNode151, self).__init__(layer)



class PytorchGraph(Graph):

    def __init__(self, model):
        # sanity check.
        super(PytorchGraph, self).__init__(model)
        self.model = model
        self.state_dict = _unique_state_dict(self.model)
        self.shape_dict = dict()
        self.layer_weight_map = dict()

    @staticmethod
    def get_node_id(node):
        import re
        node_id = re.search(r"[\d]+", node.__str__())
        return node_id.group(0)


    def build(self, shape):
        """
        build graph for pytorch
        """
        import re
        # construct graph
        dummy_input = torch.autograd.Variable(torch.randn(shape), requires_grad=False)

        graph, nodes = self.extractgraph(dummy_input)
        
        # build each layer
        for node in nodes:
            node_id = PytorchGraph.get_node_id(node)
            node_name = self.rename_nodes(node, node_id)
            output_str = node.__str__().split('=')[0]
            output_shape_str = re.findall(r'[^()!]+', output_str)
            if len(output_shape_str) > 1:
                output_shape = [int(x.replace('!', '')) for x in output_shape_str[1].split(',')]
            else:
                output_shape = None
            self.shape_dict[node_name] = output_shape
            self.layer_map[node_name] = self.CreateGraphNode(node)
            self.layer_name_map[node_name] = node_name
            # make connection
            self.node_connection(graph, node, node_name)

        super(PytorchGraph, self).build()


class PytorchGraph040(PytorchGraph):

    def __init__(self, model):
        super(PytorchGraph040, self).__init__(model)

    def extractgraph(self, dummy_input):
        with self.set_training(self.model, False):
            import torch.jit
            trace, output = torch.jit.get_trace_graph(self.model, (dummy_input, ))

        trace.set_graph(PytorchGraph._optimize_graph(trace.graph(), False))
        # nodes
        nodes = list(trace.graph().nodes())
        graph = trace.graph()
        return graph, nodes

    def rename_nodes(self, node, node_id):
        node_scope = node.scopeName()
        node_name = node_scope + node_id
        node_name = node_name.replace('-','n').replace('\\','n').replace('/','n').replace('_','n').replace('[','n').replace(']','n')
        return node_name

    def node_connection(self, graph, node, node_name):
        for node_input in list(node.inputs()):
            if PytorchGraph.get_node_id(node_input.node()) and node_input.node().scopeName():
                    node_input_name = node_input.node().scopeName() + PytorchGraph.get_node_id(node_input.node())
                    node_input_name = node_input_name.replace('-','n').replace('\\','n').replace('/','n').replace('_','n').replace('[','n').replace(']','n')
                    self._make_connection(node_input_name, node_name)
    
    def CreateGraphNode(self, node):
        return PytorchGraphNode040(node)

    @staticmethod
    def _optimize_graph(graph, aten, export_raw_ir=False):
        # run dce first to eliminate dead parts of the graph that might have been
        # left behind by things like symbolic_override

        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)

        torch._C._jit_pass_peephole(graph)
        torch._C._jit_pass_lint(graph)
        if not export_raw_ir:
            graph = torch._C._jit_pass_onnx(graph, aten)
            torch._C._jit_pass_lint(graph)
            torch._C._jit_pass_onnx_peephole(graph)
            torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)
        graph = torch._C._jit_pass_canonicalize(graph)
        torch._C._jit_pass_lint(graph)
        return graph
    
    @contextlib.contextmanager
    def set_training(self, model, mode):
        r"""
        A context manager to temporarily set the training mode of 'model'
        to 'mode', resetting it when we exit the with-block.  A no-op if
        mode is None.
        """
        if mode is None:
            yield
            return
        old_mode = model.training
        if old_mode != mode:
            model.train(mode)
        try:
            yield
        finally:
            if old_mode != mode:
                model.train(old_mode)

class PytorchGraph151(PytorchGraph):

    def __init__(self, model):
        super(PytorchGraph151, self).__init__(model)

    def extractgraph(self, dummy_input):
        import re
        from torch.onnx.utils import OperatorExportTypes
        from torch.onnx.utils import _trace

        self.model.eval()
        with scope_name_workaround():
            graph = _trace(self.model, dummy_input, OperatorExportTypes.ONNX)
        nodes = list(graph.nodes())
        
        for node in nodes:
            # print(node.__str__())
            node_id = PytorchGraph.get_node_id(node)
            node_name = 'node' + node_id
            self.layer_weight_map[node_name] = '.'.join(
                re.findall(r'\[([\w\d.]+)\]', node.scopeName())
            )
        return graph, nodes
    
    def rename_nodes(self, node, node_id):
        node_name = 'node' + node_id
        return node_name
    
    def node_connection(self, graph, node, node_name):
        for node_input in list(node.inputs()):
            if PytorchGraph.get_node_id(node_input.node()) and node_input.node() in graph.nodes():
                node_input_name = 'node' + PytorchGraph.get_node_id(node_input.node())
                self._make_connection(node_input_name, node_name)

    def CreateGraphNode(self, node):
        return PytorchGraphNode151(node)
    
    
