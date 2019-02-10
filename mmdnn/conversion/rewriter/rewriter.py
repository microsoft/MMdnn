from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow
from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import attr_value_pb2

import sys

from mmdnn.conversion.tensorflow.tensorflow_graph import *
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.rewriter.graph_matcher import *

from mmdnn.conversion.common.DataStructure import *
from tensorflow.core.framework.node_def_pb2 import NodeDef
from mmdnn.conversion.rewriter.rnn_utils import *


class UnitRewriterBase(object):


    def __init__(self, graph, weights_dict):
        self._graph = graph
        self._weights_dict = weights_dict

    def _rewrite_graph_by_pattern(self, pattern_name, graph_type):
        pattern = rnn_patterns[graph_type][pattern_name]
        matcher = GraphMatcher(pattern)
        match_results = list(matcher.match_ops(self._graph.get_nodes()))
        scope_names_dict = dict() # name: No.

        for i in range(len(match_results)):
            result = match_results[i]
            top_pattern_name = pattern_name + '_' + str(i)

            top_pattern = result._name_to_pattern[pattern_name]
            self.create_scope(result, top_pattern, scope_names_dict)

            top_op = result._pattern_to_op[top_pattern]
            top_op.scope = top_op.scope + '/top'

            # self.store_const_to_top(result)
            # self.set_top_node_prop(result, pattern_name)
            self.process_match_result(result, pattern_name)


    def rewrite_graph(self, pattern_names, graph_type):
        from six import string_types as _string_types
        if isinstance(pattern_names, _string_types):
            pattern_names = [pattern_names]
        elif not isinstance(pattern_names, list):
            raise ValueError
        for pattern_name in pattern_names:
            self._rewrite_graph_by_pattern(pattern_name, graph_type)

    def run(self, pattern_names, graph_type):
        self.rewrite_graph(pattern_names, graph_type)
    
    def store_const_to_top(self, match_result):
        top_node = list(match_result._pattern_to_op.values())[0]
        kwargs = dict()
        for pattern, op in match_result._pattern_to_op.items():
            if pattern.name and pattern.type == 'Const':
                if tensor_util.MakeNdarray(op.get_attr('value')).shape == (1, ):
                    kwargs[pattern.name] = np.asscalar(tensor_util.MakeNdarray(op.get_attr('value')))
                else:
                    kwargs[pattern.name] = np.squeeze(tensor_util.MakeNdarray(op.get_attr('value')))
        if hasattr(top_node, 'kwargs'):
            top_node.kwargs.update(kwargs)
        else:
            top_node.kwargs = kwargs

    def create_scope(self, result, pattern, scope_names_dict, parent_scope_name=''):
        op = result._pattern_to_op[pattern]

        if pattern.name:
            # Do not include input op.
            if 'input' in pattern.name.split('/')[-1]:
                return
            else:
                no = scope_names_dict.get(pattern.name, 0)
                scope_names_dict[pattern.name] = no + 1
                if parent_scope_name:
                    current_scope_name = '/'.join([parent_scope_name, pattern.name]) + '_' + str(no)
                else:
                    current_scope_name = pattern.name + '_' + str(no)
        else:
            current_scope_name = parent_scope_name
        op.scope = current_scope_name
        for sub_pattern in pattern.inputs:
            self.create_scope(result, sub_pattern, scope_names_dict, current_scope_name)

    def set_top_node_prop(self, match_result):
        raise NotImplementedError


    def process_match_result(self, match_result, pattern_name):
        raise NotImplementedError
