from collections import namedtuple
from functools import reduce
from google.protobuf import text_format
from copy import deepcopy
import numbers
import os
import tempfile

from mmdnn.conversion.caffe.mapper import get_handler_name
from mmdnn.conversion.caffe.resolver import get_caffe_resolver, has_pycaffe
from mmdnn.conversion.caffe.shape import *
from mmdnn.conversion.caffe.errors import print_stderr, ConversionError


layer_num_to_name = {
    0: 'None',
    1: 'Accuracy',
    2: 'BNLL',
    3: 'Concat',
    4: 'Convolution',
    5: 'Data',
    6: 'Dropout',
    7: 'EuclideanLoss',
    8: 'Flatten',
    9: 'HDF5Data',
    10: 'HDF5Output',
    11: 'Im2col',
    12: 'ImageData',
    13: 'InfogainLoss',
    14: 'InnerProduct',
    15: 'LRN',
    16: 'MultinomialLogisticLoss',
    17: 'Pooling',
    18: 'ReLU',
    19: 'Sigmoid',
    20: 'Softmax',
    21: 'SoftmaxWithLoss',
    22: 'Split',
    23: 'TanH',
    24: 'WindowData',
    25: 'Eltwise',
    26: 'Power',
    27: 'SigmoidCrossEntropyLoss',
    28: 'HingeLoss',
    29: 'MemoryData',
    30: 'ArgMax',
    31: 'Threshold',
    32: 'DummyData',
    33: 'Slice',
    34: 'MVN',
    35: 'AbsVal',
    36: 'Silence',
    37: 'ContrastiveLoss',
    38: 'Exp',
    39: 'Deconvolution',
    40: 'PReLU',
    41: 'ELU',
    }

LAYER_DESCRIPTORS = {
    # Caffe Types
    'AbsVal': shape_identity,
    'Accuracy': shape_scalar,
    'ArgMax': shape_not_implemented,
    'BatchNorm': shape_identity,
    'BNLL': shape_not_implemented,
    'Concat': shape_concat,
    'ContrastiveLoss': shape_scalar,
    'Convolution': shape_convolution,
    'Crop': shape_not_implemented,
    'Deconvolution': shape_deconvolution,
    'Data': shape_data,
    'Dropout': shape_identity,
    'DummyData': shape_data,
    'EuclideanLoss': shape_scalar,
    'Eltwise': shape_identity,
    'Exp': shape_identity,
    'Flatten': shape_flatten,
    'HDF5Data': shape_data,
    'HDF5Output': shape_identity,
    'HingeLoss': shape_scalar,
    'Im2col': shape_not_implemented,
    'ImageData': shape_data,
    'InfogainLoss': shape_scalar,
    'InnerProduct': shape_inner_product,
    'Input': shape_data,
    'LRN': shape_identity,
    'MemoryData': shape_mem_data,
    'MultinomialLogisticLoss': shape_scalar,
    'MVN': shape_not_implemented,
    'Pooling': shape_pool,
    'Unpooling': shape_unpool,
    'Power': shape_identity,
    'ReLU': shape_identity,
    'Scale': shape_identity,
    'Sigmoid': shape_identity,
    'SigmoidCrossEntropyLoss': shape_scalar,
    'Silence': shape_identity,
    'Softmax': shape_identity,
    'SoftmaxWithLoss': shape_scalar,
    'Split': shape_split,
    'Slice': shape_not_implemented,
    'TanH': shape_identity,
    'WindowData': shape_not_implemented,
    'Threshold': shape_identity,
    'Reshape' : shape_reshape,
    'ResizeBilinear': shape_reshape,
    'PReLU'   : shape_identity,
    'ELU' : shape_identity,
    }

LAYER_TYPES = LAYER_DESCRIPTORS.keys()

LayerType = type('LayerType', (), {t : t for t in LAYER_TYPES})

KernelParameters = namedtuple('KernelParameters', ['global_pooling', 'k_h', 'k_w', 's_h', 's_w', 'p_h', 'p_w'])

class NodeKind(LayerType):

    @staticmethod
    def map_raw_kind(node_kind):
        if isinstance(node_kind, int):
            node_kind = layer_num_to_name[node_kind]
        else:
            node_kind = str(node_kind)
        if node_kind in LAYER_TYPES:
            return node_kind
        return None

    @staticmethod
    def compute_output_shape(node):
        try:
            return LAYER_DESCRIPTORS[node.kind](node)
        except NotImplementedError:
            raise ConversionError('Output shape computation not implemented for type: %s' % node.kind)

LAYER_IN_TRAIN_PROTO = [NodeKind.ImageData, NodeKind.Data, NodeKind.HDF5Data, NodeKind.HDF5Output, NodeKind.WindowData, NodeKind.DummyData, NodeKind.MemoryData]

class CaffeNode(object):
    def __init__(self, name, kind, layer=None):
        self.name = name
        self.kind = kind
        self.layer = layer
        self.parents = []
        self.children = []
        self.data = None
        self.output = []
        self.output_shape = None
        self.metadata = {}

    def add_parent(self, parent_node, from_output, index=None):
        assert parent_node not in self.parents
        index = len(self.parents) if index is None else index
        self.parents.insert(index, (parent_node, from_output))
        if self not in parent_node.children:
            parent_node.children.append(self)

    def get_only_parent(self):
        if len(self.parents) != 1:
            raise ConversionError('Node (%s) expected to have 1 parent. Found %s.' % (self, len(self.parents)))
        return self.parents[0]

    @property
    def parameters(self):
        if self.layer is not None:
            params = get_handler_name(self.kind)
            if params == 'deconvolution':
                params = 'convolution'
            params = '_'.join((params, 'param'))
            try:
                return getattr(self.layer, params)
            except AttributeError:
                raise ConversionError('Caffe parameters not found for layer kind: %s' % (self.kind))
        return None

    @staticmethod
    def get_kernel_value(scalar, repeated, idx, default=None):
        if scalar:
            return scalar
        if repeated:
            if isinstance(repeated, numbers.Number):
                return repeated
            if len(repeated) == 1:
                # Same value applies to all spatial dimensions
                return int(repeated[0])
            assert idx < len(repeated)
            # Extract the value for the given spatial dimension
            return repeated[idx]
        if default is None:
            raise ValueError('Unable to determine kernel parameter!')
        return default

    @property
    def kernel_parameters(self):
        assert self.kind in (NodeKind.Convolution, NodeKind.Pooling, NodeKind.Unpooling, NodeKind.Deconvolution)
        params = self.parameters
        global_pooling = hasattr(params, 'global_pooling') and params.global_pooling
        if not global_pooling:
            k_h = self.get_kernel_value(params.kernel_h, params.kernel_size, 0)
            k_w = self.get_kernel_value(params.kernel_w, params.kernel_size, 1)
            s_h = self.get_kernel_value(params.stride_h, params.stride, 0, default=1)
            s_w = self.get_kernel_value(params.stride_w, params.stride, 1, default=1)
        else:
            k_h = k_w = 0
            s_h = s_w = 1
        p_h = self.get_kernel_value(params.pad_h, params.pad, 0, default=0)
        p_w = self.get_kernel_value(params.pad_w, params.pad, 1, default=0)
        return KernelParameters(global_pooling, k_h, k_w, s_h, s_w, p_h, p_w)

    def __str__(self):
        return '[%s] %s' % (self.kind, self.name)

    def __repr__(self):
        return '%s (0x%x)' %(self.name, id(self))


class CaffeGraph(object):

    def __init__(self, nodes=None, name=None):
        self.nodes = nodes or []
        self.node_lut = {node.name: node for node in self.nodes}
        self.name = name
        self.prototxt = None

    def add_node(self, node):
        self.nodes.append(node)
        self.node_lut[node.name] = node

    def get_node(self, name):
        try:
            return self.node_lut[name]
        except KeyError:
            raise ConversionError('Layer not found: %s' % name)

    def get_input_nodes(self):
        return [node for node in self.nodes if len(node.parents) == 0]

    def get_output_nodes(self):
        return [node for node in self.nodes if len(node.children) == 0]

    def topologically_sorted(self):
        visited = set()
        sorted_nodes = []
        def topo_sort_dfs(node, visited, sorted_nodes):
            if node in visited:
                return
            visited.add(node)
            for n, idx in node.parents:
                topo_sort_dfs(n, visited, sorted_nodes)
            sorted_nodes.append(node)
        for node in self.nodes:
            topo_sort_dfs(node, visited, sorted_nodes)
        return sorted_nodes

    def compute_output_shapes(self, model):
        sorted_nodes = self.topologically_sorted()
        (tmp_handle, tmp_prototxt) = tempfile.mkstemp(suffix=".prototxt")
        with open(tmp_prototxt, 'w') as f:
            f.write(text_format.MessageToString(model))
        self.prototxt = tmp_prototxt
        if has_pycaffe():
            caffe = get_caffe_resolver().caffe
            net = caffe.Net(tmp_prototxt, caffe.TEST)
            for key, value in net.blobs.items():
                try:
                    node = self.get_node(key)
                    dims = list(value.shape)
                    dims = dims + [1] * (4 - len(dims))
                    node.output_shape = TensorShape(*dims)
                except:
                    continue
            for node in sorted_nodes:
                if node.output_shape is None:
                    node.output_shape = TensorShape(*NodeKind.compute_output_shape(node))
            os.close(tmp_handle)
        else:
            for node in sorted_nodes:
                node.output_shape = TensorShape(*NodeKind.compute_output_shape(node))

    # consider rewrite this function to Network.py
    def replaced(self, new_nodes):
        return CaffeGraph(nodes=new_nodes, name=self.name)

    def transformed(self, transformers):
        graph = self
        for transformer in transformers:
            graph = transformer(graph)
            if graph is None:
                raise ConversionError('Transformer failed: {}'.format(transformer))
            assert isinstance(graph, CaffeGraph)
        return graph

    def __contains__(self, key):
        return key in self.node_lut

    def __str__(self):
        def get_max_shape(data):
            if isinstance(data, dict):
                max = 0
                val = None
                for k, v in data.items():
                    tmp = reduce(lambda x, y: x*y, v.shape)
                    if  tmp > max:
                        val = v.shape
                        max = tmp
                return val
            else:
                return data[0].shape
        hdr = '{:<20} {:<30} {:>20} {:>20}'.format('Type', 'Name', 'Param', 'Output')
        s = [hdr, '-' * 94]
        for node in self.topologically_sorted():
            data_shape = get_max_shape(node.data) if node.data else '--'
            out_shape = node.output_shape or '--'
            s.append('{:<20} {:<30} {!s:>20} {!s:>20}'.format(node.kind, node.name, data_shape, tuple(out_shape)))
        return '\n'.join(s)


class GraphBuilder(object):
    def __init__(self, model_path, input_shape=None, is_train_proto=False, phase='test'):
        self.model_path = model_path
        self.phase = phase
        self.is_train_proto = is_train_proto
        self.input_shape = input_shape
        self.load()

    def load(self):
        self.model = get_caffe_resolver().NetParameter()
        with open(self.model_path, 'r') as f:
            text_format.Merge(f.read(), self.model)
        if self.is_train_proto:
            self.process_train_proto()

    def process_train_proto(self):
        layers = self.model.layer or self.model.layers
        delete_layer = set()
        split_op_map = dict()
        loss_layers = [layer for layer in layers if NodeKind.map_raw_kind(layer.type) in (NodeKind.SoftmaxWithLoss, NodeKind.SigmoidCrossEntropyLoss)]
        a = [layers.remove(layer) for layer in layers[:] if layer in loss_layers[:-1] or NodeKind.map_raw_kind(layer.type) in LAYER_IN_TRAIN_PROTO]
        for layer in layers[:]:
            if 'label' in layer.bottom:
                if NodeKind.map_raw_kind(layer.type) in (NodeKind.SoftmaxWithLoss, NodeKind.SigmoidCrossEntropyLoss):
                    continue
                elif NodeKind.map_raw_kind(layer.type) == NodeKind.Split:
                    for item in layer.top:
                        delete_layer.add(item)
                layers.remove(layer)
            elif NodeKind.map_raw_kind(layer.type) == NodeKind.Split:
                for item in layer.top:
                    split_op_map[item] = layer.bottom[0]
                layers.remove(layer)

        for layer in layers[:]:
            for item in delete_layer:
                if item in layer.bottom:
                    layers.remove(layer)
                    break
            for key, value in split_op_map.items():
                if key in layer.bottom:
                    layer.bottom.remove(key)
                    layer.bottom.append(value)
        self.model.input.append('data')
        self.model.input_dim.extend(self.input_shape)
        last_layer = layers[-1]
        kind = NodeKind.map_raw_kind(last_layer.type)
        if kind in (NodeKind.SoftmaxWithLoss, NodeKind.SigmoidCrossEntropyLoss):
            pred = layers.add()
            pred.name = 'prob'
            pred.top.append('prob')
            pred.bottom.append(last_layer.bottom[0])
            if kind == NodeKind.SoftmaxWithLoss:
                pred.type = NodeKind.Softmax if self.model.layer else 20 # competiable with old version caffe proto
            elif kind == NodeKind.SigmoidCrossEntropyLoss:
                pred.type = NodeKind.Sigmoid if self.model.layer else 19
        layers.remove(last_layer)

    def filter_layers(self, layers):
        phase_map = {0: 'train', 1: 'test'}
        filtered_layer_names = set()
        filtered_layers = []
        for layer in layers:
            phase = self.phase
            if len(layer.include):
                phase = phase_map[layer.include[0].phase]
            if len(layer.exclude):
                phase = phase_map[1 - layer.include[0].phase]
            exclude = (phase != self.phase)
            # Dropout layers appear in a fair number of Caffe
            # test-time networks. These are just ignored. We'll
            # filter them out here.
            if (not exclude) and (phase == 'test'):
                exclude = (layer.type == LayerType.Dropout)
            if (not exclude):
                exclude = (layer.type == LayerType.Silence)
            if not exclude:
                if layer.name in filtered_layer_names:
                    for i in range(1, len(filtered_layer_names)):
                        new_name = layer.name + '_%s' % i
                        if new_name not in filtered_layer_names:
                            layer.name = new_name
                            break
                filtered_layer_names.add(layer.name)
                filtered_layers.append(layer)
        return filtered_layers

    def make_node(self, layer):
        kind = NodeKind.map_raw_kind(layer.type)
        if kind is None:
            # TODO: raise error
            pass
        node = CaffeNode(layer.name, kind, layer=layer)
        node.output.append(layer.name.replace('/', '_'))
        node.output.extend(layer.top[1:])
        return node

    def make_input_node(self):
        nodes = [CaffeNode(name, NodeKind.Data) for name in self.model.input]
        if len(nodes):
            input_dim = list(map(int, self.model.input_dim))
            if not input_dim:
                if len(self.model.input_shape) > 0:
                    input_dim = list(map(int, self.model.input_shape[0].dim))
                else:
                    # TODO: raise error
                    pass
            for node in nodes:
                node.output_shape = tuple(input_dim)
                node.output.append('data')
        return nodes

    def build(self):
        layers = self.model.layers or self.model.layer
        layers = self.filter_layers(layers)
        nodes = self.make_input_node()
        nodes += [self.make_node(layer) for layer in layers]
        graph = CaffeGraph(nodes=nodes, name=self.model.name)
        node_outputs = {}
        for idx, layer in enumerate(layers):
            node = graph.get_node(layer.name)
            for input_name in layer.bottom:
                assert input_name != layer.name
                parent_node = node_outputs.get(input_name)
                if (parent_node is None) or (parent_node==node):
                    parent_node = graph.get_node(input_name)
                if parent_node.layer:
                    for i, output in enumerate(parent_node.layer.top):
                        if input_name == output:
                            node.add_parent(parent_node, i)
                else:
                    node.add_parent(parent_node, 0)
            for output_name in layer.top:
                if output_name == layer.name:
                    continue
                node_outputs[output_name] = node
        graph.compute_output_shapes(self.model)
        return graph
