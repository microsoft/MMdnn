from __future__ import unicode_literals
from google.protobuf import text_format
import numpy as np
from mmdnn.conversion.caffe.graph import GraphBuilder, NodeKind, LAYER_IN_TRAIN_PROTO
from mmdnn.conversion.caffe.mapper import NodeMapper, get_handler_name
from mmdnn.conversion.caffe.resolver import get_caffe_resolver, has_pycaffe
from mmdnn.conversion.caffe.errors import print_stderr, ConversionError
from mmdnn.conversion.caffe.common_graph import Graph
from mmdnn.conversion.caffe.utils import get_lower_case, get_upper_case


class DataInjector(object):
    '''
    Associates parameters loaded from a .caffemodel file with their corresponding nodes.
    '''

    def __init__(self, def_path, data_path):
        # The .prototxt file defining the graph
        self.def_path = def_path
        # The .caffemodel file containing the learned parameters
        self.data_path = data_path
        # Set to true if the fallback protocol-buffer based backend was used
        self.did_use_pb = False
        # A list containing (layer name, parameters) tuples
        self.params = None
        # Load the parameters
        self.caffemodel = None
        if has_pycaffe() and self.def_path:
            self.load_using_caffe()
        else:
            self.load_using_pb()

    def load_using_caffe(self):
        caffe = get_caffe_resolver().caffe
        net = caffe.Net(str(self.def_path), str(self.data_path), caffe.TEST)
        data = lambda blob: blob.data
        self.params = [(k, list(map(data, v))) for k, v in net.params.items()]

    def load_using_pb(self):
        self.caffemodel = get_caffe_resolver().NetParameter()
        self.caffemodel.MergeFromString(open(self.data_path, 'rb').read())
        pair = lambda layer: (layer.name, self.normalize_pb_data(layer))
        layers = self.caffemodel.layers or self.caffemodel.layer
        self.params = [pair(layer) for layer in layers if layer.blobs]
        self.did_use_pb = True

    def normalize_pb_data(self, layer):
        transformed = []
        for blob in layer.blobs:
            if len(blob.shape.dim):
                dims = blob.shape.dim
                c_o, c_i, h, w = map(int, [1] * (4 - len(dims)) + list(dims))
            else:
                c_o = blob.num
                c_i = blob.channels
                h = blob.height
                w = blob.width
            data = np.array(blob.data, dtype=np.float32).reshape(c_o, c_i, h, w)
            transformed.append(data)
        return transformed

    def adjust_parameters(self, node, data):
        if not self.did_use_pb:
            return data
        # When using the protobuf-backend, each parameter initially has four dimensions.
        # In certain cases (like FC layers), we want to eliminate the singleton dimensions.
        # This implementation takes care of the common cases. However, it does leave the
        # potential for future issues.
        # The Caffe-backend does not suffer from this problem.
        data = list(data)
        squeeze_indices = [1]  # Squeeze biases.
        if node.kind == NodeKind.InnerProduct:
            squeeze_indices.append(0)  # Squeeze FC.
        if len(data)==1:
            squeeze_indices=[0]
        if node.kind == 'Convolution':
            return data
        for idx in squeeze_indices:
            data[idx] = np.squeeze(data[idx])
        return data

    def __call__(self, graph):
        for layer_name, data in self.params:
            if layer_name in graph:
                node = graph.get_node(layer_name)
                node.data = self.adjust_parameters(node, data)
            else:
                print_stderr('Ignoring parameters for non-existent layer: %s' % layer_name)
        return graph


class NodeRenamer(object):

    def __call__(self, graph):
        for node in graph.nodes:
            node.name = node.name.replace('/', '_')
        return graph


class DataReshaper(object):

    def __init__(self, mapping, replace=True):
        # A dictionary mapping NodeKind to the transposed order.
        self.mapping = mapping
        # The node kinds eligible for reshaping
        self.reshaped_node_types = self.mapping.keys()
        # If true, the reshaped data will replace the old one.
        # Otherwise, it's set to the reshaped_data attribute.
        self.replace = replace

    def has_spatial_parent(self, node):
        try:
            parent = node.get_only_parent()[0]
            s = parent.output_shape
            return s.height > 1 or s.width > 1
        except ConversionError:
            return False

    def map(self, node_kind):
        try:
            return self.mapping[node_kind]
        except KeyError:
            raise ConversionError('Ordering not found for node kind: {}'.format(node_kind))

    def _is_image_data(self, node):
        return len([child for child in node.children if child.kind in (NodeKind.Convolution, NodeKind.Pooling, NodeKind.Unpooling)])

    def __call__(self, graph):
        for node in graph.nodes:
            if node.data is None:
                continue
            if node.kind not in self.reshaped_node_types:
                # Check for 2+ dimensional data
                if any(len(tensor.shape) > 1 for tensor in node.data):
                    print_stderr('Warning: parameters not reshaped for node: {}'.format(node))
                continue
            transpose_order = self.map(node.kind)
            weights = node.data[0]
            if (node.kind == NodeKind.InnerProduct) and self.has_spatial_parent(node):
                # The FC layer connected to the spatial layer needs to be
                # re-wired to match the new spatial ordering.
                in_shape = node.get_only_parent()[0].output_shape
                fc_shape = weights.shape
                output_channels = fc_shape[0]
                weights = weights.reshape((output_channels, in_shape.channels, in_shape.height,
                                           in_shape.width))
                weights = weights.transpose(self.map(NodeKind.Convolution))
                node.reshaped_data = weights.reshape(fc_shape[transpose_order[0]],
                                                     fc_shape[transpose_order[1]])
            else:
                node.reshaped_data = weights.transpose(transpose_order)
            # node.reshaped_data = weights.transpose(transpose_order)
        if self.replace:
            for node in graph.nodes:
                if hasattr(node, 'reshaped_data'):
                    # Set the weights
                    node.data[0] = node.reshaped_data
                    del node.reshaped_data
        return graph


class SubNodeFuser(object):
    '''
    An abstract helper for merging a single-child with its single-parent.
    '''

    def __call__(self, graph):
        nodes = graph.nodes
        fused_nodes = []
        for node in nodes:
            if len(node.parents) != 1:
                # We're only fusing nodes with single parents
                continue
            parent, from_output = node.get_only_parent()
            if len(parent.children) != 1:
                # We can only fuse a node if its parent's
                # value isn't used by any other node.
                continue
            if not self.is_eligible_pair(parent, node):
                continue
            # Rewrite the fused node's children to its parent.
            for child in node.children:
                index = [n for n, (input, idx) in enumerate(child.parents) if input == node][0]
                child.parents.pop(index)
                child.add_parent(parent, from_output, index)
            # Disconnect the fused node from the graph.
            parent.children.remove(node)
            fused_nodes.append(node)
            # Let the sub-class merge the fused node in any arbitrary way.
            self.merge(parent, node)
        transformed_nodes = [node for node in nodes if node not in fused_nodes]
        return graph.replaced(transformed_nodes)

    def is_eligible_pair(self, parent, child):
        '''Returns true if this parent/child pair is eligible for fusion.'''
        raise NotImplementedError('Must be implemented by subclass.')

    def merge(self, parent, child):
        '''Merge the child node into the parent.'''
        raise NotImplementedError('Must be implemented by subclass')


class ReLUFuser(SubNodeFuser):
    '''
    Fuses rectified linear units with their parent nodes.
    '''

    def __init__(self, allowed_parent_types=None):
        # Fuse ReLUs when the parent node is one of the given types.
        # If None, all node types are eligible.
        self.allowed_parent_types = allowed_parent_types

    def is_eligible_pair(self, parent, child):
        return ((self.allowed_parent_types is None or parent.kind in self.allowed_parent_types) and
                child.kind == NodeKind.ReLU)

    def merge(self, parent, _):
        parent.metadata['relu'] = True


class BatchNormScaleBiasFuser(SubNodeFuser):
    '''
    The original batch normalization paper includes two learned
    parameters: a scaling factor \gamma and a bias \beta.
    Caffe's implementation does not include these two. However, it is commonly
    replicated by adding a scaling+bias layer immidiately after the batch norm.

    This fuser merges the scaling+bias layer with the batch norm.
    '''

    def is_eligible_pair(self, parent, child):
        return (parent.kind == NodeKind.BatchNorm and child.kind == NodeKind.Scale and
                child.parameters.axis == 1 and child.parameters.bias_term == True)

    def merge(self, parent, child):
        parent.scale_bias_node = child


class BatchNormPreprocessor(object):
    '''
    Prescale batch normalization parameters.
    Concatenate gamma (scale) and beta (bias) terms if set.
    '''

    def __call__(self, graph):
        for node in graph.nodes:
            if node.kind != NodeKind.BatchNorm:
                continue
            assert node.data is not None
            assert len(node.data) == 3
            mean, variance, scale = node.data

            # Prescale the stats
            scaling_factor = 1.0 / scale if scale != 0 else 0

            if len(np.squeeze(mean) == 1):
                mean = np.squeeze(mean)
                variance = np.squeeze(variance)
                scaling_factor = np.squeeze(scaling_factor)

            mean *= scaling_factor
            variance *= scaling_factor

            # Replace with the updated values
            node.data = [mean, variance]
            if hasattr(node, 'scale_bias_node'):
                # Include the scale and bias terms
                gamma, beta = node.scale_bias_node.data
                node.data += [gamma, beta]
        return graph


class ParameterNamer(object):
    '''
    Convert layer data arrays to a dictionary mapping parameter names to their values.
    '''

    def __call__(self, graph):
        for node in graph.nodes:
            if node.data is None:
                continue
            if node.kind in (NodeKind.Convolution, NodeKind.Deconvolution, NodeKind.InnerProduct):
                names = ('weights',)
                if node.parameters.bias_term:
                    names += ('bias',)
            elif node.kind == NodeKind.BatchNorm:
                names = ('mean', 'var')
                if len(node.data) == 4:
                    names += ('scale', 'bias')
            elif node.kind == NodeKind.PReLU:
                names = ('gamma',)
            elif node.kind == NodeKind.ELU:
                names = ('alpha',)
            else:
                print_stderr('WARNING: Unhandled parameters: {}'.format(node.kind))
                continue
            assert len(names) == len(node.data)
            node.data = dict(zip(names, node.data))
        return graph


class CaffeTransformer(object):

    def __init__(self, def_path, data_path, target_toolkit, input_shape=None, phase='test'):
        self.layer_name_map = {}
        self.data_injector = None
        self.is_train_proto = False
        self.input_shape = input_shape
        if def_path is None:
            if self.input_shape is None:
                raise ConversionError('if the graph prototxt is not provided, the input shape should be provided')
            self.input_shape = [1] + self.input_shape
            def_path, self.data_injector = self.gen_prototxt_from_caffemodel(data_path, self.input_shape)
            self.is_train_proto = True
        else:
            model = get_caffe_resolver().NetParameter()
            with open(def_path, 'r') as f:
                text_format.Merge(f.read(), model)
            layers = model.layers or model.layer
            if len([layer for layer in layers if NodeKind.map_raw_kind(layer.type) in LAYER_IN_TRAIN_PROTO]) > 0:
                if self.input_shape is None:
                    raise ConversionError('the train_val.prototxt should be provided with the input shape')
                self.input_shape = [1] + self.input_shape
                self.is_train_proto = True
        graph = GraphBuilder(def_path, self.input_shape, self.is_train_proto, phase).build()
        if self.is_train_proto:
            def_path = graph.prototxt
        if data_path is not None:
            graph = graph.transformed([
                self.data_injector if self.data_injector else DataInjector(def_path, data_path), # Load and associate learned parameters
                BatchNormScaleBiasFuser(),
                BatchNormPreprocessor() # Pre-process batch normalization data
            ])
            target_toolkit = target_toolkit.lower()
            if target_toolkit not in ('caffe', 'caffe2'):
                graph = graph.transformed([DataReshaper({ # Reshape the parameters to TensorFlow's ordering
                    NodeKind.Convolution: (2, 3, 1, 0), # (c_o, c_i, h, w) -> (h, w, c_i, c_o)
                    NodeKind.Deconvolution: (2, 3, 1, 0), # (c_o, c_i, h, w) -> (h, w, c_i, c_o)
                    NodeKind.InnerProduct: (1, 0) # (c_o, c_i) -> (c_i, c_o)
                }),
                    ParameterNamer() # Convert parameters to dictionaries
                ])
        self.graph = graph
        #  self.graph = NodeRenamer()(graph)
        print (self.graph)

    def gen_prototxt_from_caffemodel(self, data_path, input_shape):
        prototxt = 'deploy.prototxt'
        data_injector = DataInjector(None, data_path)
        caffemodel = data_injector.caffemodel
        layers = caffemodel.layers or caffemodel.layer
        for item in layers:
            item.ClearField('blobs')
        with open(prototxt ,'w') as f:
            f.write(str(caffemodel))
        return prototxt, data_injector

    def transform_data(self):
        return {self.layer_name_map[node.name]: node.data for node in self.graph.nodes if node.data}

    def transform_graph(self):
        for node in self.graph.nodes:
            self.layer_name_map[node.name] = node.name

        ret = []
        for node in self.graph.nodes:
            mapped_node = self.map_node(node)
            if isinstance(mapped_node, list):
                ret.extend([n for n in mapped_node])
            elif mapped_node:
                ret.append(mapped_node)


        name = get_upper_case(get_lower_case(self.graph.name))
        return Graph(name, ret)
        #return Graph(name, [self.map_node(node) for node in self.graph.nodes])

    def get_handler(self, node_kind, prefix):
        name = get_handler_name(node_kind)
        name = '_'.join((prefix, name))
        try:
            return getattr(NodeMapper, name)
        except AttributeError:
            raise ConversionError('No handler found for node kind: %s (expected: %s)' % (node_kind, name))

    def map_node(self, node):
        map_func = self.get_handler(node.kind, 'map')

        mapped_node = map_func(node)
        # assert mapped_node is not None

        if isinstance(mapped_node, list):
            ret = []
            for idx, cur_node in enumerate(mapped_node):
                cur_node.name = node.name + '_' + str(idx)
                if idx == 0:
                    cur_node.input.extend([self.layer_name_map[input.name] for input, idx in node.parents])
                else:
                    cur_node.input.extend([node.name + '_' + str(idx - 1)])

                if idx == len(mapped_node) - 1:
                    cur_node.output.extend(node.output)
                else:
                    cur_node.output.extend([node.name + '_' + str(idx + 1)])

                self.layer_name_map[node.name] = node.name + '_' + str(len(mapped_node) - 1)
                ret.append(cur_node)
            return ret

        # skip when mapped_node is None
        elif not mapped_node:
            input_of_next = node.get_only_parent()[0]
            next_node = node.children
            for next in next_node:
                next.parents[0] = tuple([input_of_next, next.parents[0][1]])

        else:
            mapped_node.name = node.name
            mapped_node.input.extend(['%s' % (self.layer_name_map[input.name]) for input, idx in node.parents])
            mapped_node.output.extend(node.output)
            return mapped_node

