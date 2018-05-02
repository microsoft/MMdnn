#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import six
from six import string_types as _string_types
from mmdnn.conversion.caffe.errors import ConversionError
from mmdnn.conversion.common.IR.graph_pb2 import GraphDef, NodeDef, TensorShape
from mmdnn.conversion.caffe.utils import get_real_name


def assign_attr_value(attr, val):
    '''Assign value to AttrValue proto according to data type.'''
    if isinstance(val, bool):
        attr.b = val
    elif isinstance(val, six.integer_types):
        attr.i = val
    elif isinstance(val, float):
        attr.f = val
    elif isinstance(val, str):
        attr.s = val.encode('utf-8')
    elif isinstance(val, TensorShape):
        attr.shape.MergeFromString(val.SerializeToString())
    elif isinstance(val, list):
        if len(val) == 0: return

        if isinstance(val[0], six.integer_types):
            attr.list.i.extend(val)
        elif isinstance(val[0], TensorShape):
            attr.list.shape.extend(val)
        else:
            raise NotImplementedError('AttrValue cannot be of %s %s' % (type(val), type(val[0])))
    else:
        raise NotImplementedError('AttrValue cannot be of %s' % type(val))


def fetch_attr_value(attr):
    '''Fetch valid value from AttrValue proto.'''
    field = attr.WhichOneof('value')
    val = getattr(attr, field) if field else None
    return val.decode('utf-8') if isinstance(val, bytes) else val


class Node(object):
    '''An intermediate representation for DL operations.'''

    def __init__(self, node_pb2):
        assert isinstance(node_pb2, NodeDef)
        self.node_pb2 = node_pb2
        self.output = []

    @staticmethod
    def create(op, **kwargs):
        node_pb2 = NodeDef()
        node_pb2.op = op
        for k, v in kwargs.items():
            assign_attr_value(node_pb2.attr[k], v)
        return Node(node_pb2)

    @property
    def op(self):
        return self.node_pb2.op

    @property
    def name(self):
        return self.node_pb2.name

    @name.setter
    def name(self, value):
        assert isinstance(value, _string_types)
        self.node_pb2.name = value

    @property
    def input(self):
        return self.node_pb2.input

    @property
    def attr(self):
        return self.node_pb2.attr.items()


class Graph(object):
    '''An intermediate representation for DL graph.'''

    def __init__(self, name, node_list, version=0):
        if node_list and len(node_list):
            assert isinstance(node_list[0], Node)
            self.node_dict = {node.name: node for node in node_list}
        else:
            self.node_dict = {}
        self.name = name
        self.version = version

    def topologically_sorted(self):
        visited = set()
        sorted_nodes = []
        def topo_sort_dfs(node, visited, sorted_nodes):
            if node in visited:
                return
            visited.add(node)
            for n in self.get_input(node):
                topo_sort_dfs(n, visited, sorted_nodes)
            sorted_nodes.append(node)
        for node in self.node_dict.values():
            topo_sort_dfs(node, visited, sorted_nodes)
        return sorted_nodes

    def get_node(self, name):
        return self.node_dict[name]

    def add_node(self, node):
        assert node.name not in self.node_dict
        self.node_dict[node.name] = node

    def remove_node(self, name):
        return self.node_dict.pop(name)

    def get_input(self, node):
        input_nodes = []
        for name in node.input:
            name = get_real_name(name)
            if name in self.node_dict:
                input_nodes.append(self.get_node(name))
        return input_nodes

    def as_graph_def(self):
        graph_pb2 = GraphDef()
        graph_pb2.version = self.version
        graph_pb2.node.extend([node.node_pb2 for node in self.node_dict.values()])
        return graph_pb2