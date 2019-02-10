from mmdnn.conversion.common.IR.IR_graph import *
import sys
import re
import numpy as np
import collections


class Folder(object):
    """A floder to fold a graph' nodes which has same scope into one node."""

    def __init__(self, graph, weights_dict, init_level=0, fold_level_num=0, scope_names=None):
        """
        Initializes a Folder.

        Args:
          graph: the graph to be folded.
          init_level: the start scope level to be folded.
          fold_level_num: the number of level from init_level to be folded. For example,
            there are three nodes, whose scope are A/B/X, A/B/Y, A/C/Z. If init_level=0 and fold_level_num=1,
            then the fold result is A/B and A/C.
        """
        self._graph = graph
        self._weights_dict = weights_dict
        self._init_level = init_level
        self._fold_level_num = fold_level_num
        self._scope_names = scope_names


    """fold the graph by compressing the nodes which have same scope into one scope node."""
    
    def fold(self):
        self.scope_level_name_map = self._get_scope_level_name_dict()  # level: scope_name set

        if self._scope_names:
            scope_names = self._scope_names
        else:
            if not self.scope_level_name_map:
                return
            scope_names = self.scope_level_name_map[0]

        for scope_name in scope_names:
            level = self._init_level
            sub_fold_level = self._fold_level_num
            while sub_fold_level >= 0:
                self._fold(self._graph.topological_sort,
                           scope_name, level, level + sub_fold_level)
                sub_fold_level -= 1

        # check the same pattern scope node whether have same inputs, outputs and weights. 
        # For those don't have, rename their scope names.
        self.check_scope()
        self.check_weights()

        # clear out scope node, typically input constant node.
        self._graph.clear_out_scope_node()


    '''
    fold the given node list by squash the nodes whose scope is given scope into one scope node.
    Args:
      scope_list: scope node topology list
      scope_name: the scope name need to be folded
      level: the scope's level
      fold_level: the scope's sub scope level need to be folded.
    '''
    
    def _fold(self, scope_list, scope_name, level, sub_level):
        top_node = None
        # get sub_scopes
        if not self.scope_level_name_map.get(sub_level, 0):
            raise ValueError("The fold level exceed maxium scope level.")
        sub_scopes = self.scope_level_name_map[sub_level]

        for sub_scope in sub_scopes:
            sub_scope_node_name_list = self._get_scope_name_dict(self._graph.topological_sort, top_level=level,
                                                                 top_scope=scope_name, sub_level=sub_level, sub_scope=sub_scope)
            for scope_list in sub_scope_node_name_list.values():
                top_node = self._graph.get_node(scope_list[-1])
                self._create_scope_node(
                    sub_scope, scope_list, top_node)


    '''get scope_level_name_dict.  {level: scope name list(without suffix number)}'''
    
    def _get_scope_level_name_dict(self):
        scope_level_name = collections.OrderedDict()
        for node in self._graph.get_nodes():
            if not node.get_attr('scope'):
                continue
            for i, scope_name in enumerate(node.get_attr('scope').split('/')):
                # decline the suffix number
                import re
                if re.search(r'\d', scope_name.split('_')[-1]):
                    scope_name = '_'.join(scope_name.split('_')[:-1])
                if scope_name == 'top':
                    continue
                if scope_level_name.get(i, None):
                    if scope_name not in scope_level_name[i]:
                        scope_level_name[i].append(scope_name)
                else:
                    scope_level_name[i] = list([scope_name]) # use list to keep sort.

        return scope_level_name


    '''
    get the dict from required node_list by appointed scope_name and level

    Args:
    node_list: current self.topology_sort
    scope_name_dict: scope_no: node_name, a dict like {scope_name_no: a set of scope node' name}
    '''
    def _get_scope_name_dict(self, node_list, top_level=0, top_scope=None, sub_level=2, sub_scope=None):
        scope_node_names = collections.OrderedDict()

        def _insert_scope_node_names_dict(scope_no, node_name):
            if scope_node_names.get(scope_no, None):
                scope_node_names[scope_no].append(node_name)
            else:
                scope_node_names[scope_no] = list([node_name])

        def _get_scope_name_dict_by_cond(cond_top, cond_sub):
            for node_name in node_list:
                node = self._graph.get_node(node_name)
                if not node.get_attr('scope'):
                    continue
                node_scope = node.get_attr('scope')
                if cond_top(top_scope, node_scope) and cond_sub(sub_scope, node_scope):
                    if 'True' in cond_top.__name__ and 'True' not in cond_sub.__name__:
                        scope_no = node_scope.split('/')[sub_level]
                    elif 'True' not in cond_top.__name__ and 'True' in cond_sub.__name__:
                        scope_no = node_scope.split('/')[top_level]
                    else:  # both not equal True
                        scope_no = node_scope.split(
                            '/')[top_level] + '_' + node_scope.split('/')[sub_level]
                    _insert_scope_node_names_dict(scope_no, node.name)

        def cond_x_in_y(x, y): return x in y

        def cond_True(x, y): return True

        # Obtain nodes where the scope name that satisfies top_level is top_scope and sub_level is sub_scope
        if top_scope and sub_scope:
            _get_scope_name_dict_by_cond(cond_x_in_y, cond_x_in_y)
        # Obtain nodes where the scope name that satisfies in sub_level is sub_scope
        elif not top_scope and sub_scope:
            _get_scope_name_dict_by_cond(cond_True, cond_x_in_y)
        # Obtain nodes where the scope name that satisfies in top_level is top_scope
        elif top_scope and not sub_scope:
            _get_scope_name_dict_by_cond(cond_x_in_y, cond_True)
        # Obtain all nodes grouped by sub_level sub_scope
        elif top_scope is None and sub_scope is None:
            top_scopes = self.scope_level_name_map[top_level]
            for top_scope in top_scopes:  # this top_scope will replace the input top_scope
                _get_scope_name_dict_by_cond(cond_x_in_y, cond_True)

        return scope_node_names


    '''get the node names' topology sort of scope nodes'''
    
    def _get_scope_nodes_topology_list(self, scope_node_name_set):

        temp_dict = {}
        for index, name in enumerate(scope_node_name_set):
            # cover the node
            self._graph.get_node(name).covered = True
            # store idx, node into a dict and sort it later to keep its topology sort.
            index = self._graph.topological_sort.index(name)
            temp_dict[name] = index

        temp_dict = sorted(
            temp_dict.items(), key=lambda item: item[1])

        return [x[0] for x in temp_dict]


    ''' rebuild the conncetion of the edge around this scope node.'''
    
    def _rebuild_scope_edges_and_get_ret_vars(self, scope_node):
        
        def _get_index(node ,name):
            for idx, in_edge in enumerate(node.in_edges):
                if in_edge.split(':')[0] == name:
                    return idx

        return_nodes = list()
        return_variable_names = list()

        for n_name in scope_node.topology_list:
            n = self._graph.get_node(n_name)
            for in_edge in n.in_edges:

                if not in_edge.split(':')[0] in scope_node.topology_list:
                    if not in_edge in scope_node.in_edges:
                        scope_node.in_edges.append(in_edge)

                    # in_node's out edges replace n_name with scope node name.
                    in_node = self._graph.get_node(in_edge)
                    if n_name in in_node.out_edges:
                        idx = in_node.out_edges.index(n_name)
                        in_node.out_edges.remove(n_name)
                        if scope_node.name not in in_node.out_edges:
                            in_node.out_edges.insert(idx, scope_node.name)

            for out_edge in n.out_edges:

                if not out_edge in scope_node.topology_list:
                    out_node = self._graph.get_node(out_edge)
                    parent_node_variable_name = self._graph.get_parent_variable_name(out_edge.split(
                        ':')[0], [_get_index(self._graph.get_node(out_edge), n_name)])

                    if parent_node_variable_name not in return_variable_names:
                        return_nodes.append(self._graph.get_node(n_name))
                        return_variable_names.append(parent_node_variable_name)
                    scope_node.out_edges.append(out_edge)

        # no out nodes means the last node in scope nodes should be returned
        if not return_nodes:
            return_nodes.append(self._graph.get_node(
                scope_node.topology_list[-1]))
            return_variable_names.append(self._graph.get_node(
                scope_node.topology_list[-1]).real_variable_name)

        ret_idx = 0
        for ret_node, ret_variable_name in zip(return_nodes, return_variable_names):

            subscript = '' if len(ret_variable_name.split(
                '[')) == 1 else ':'+ret_variable_name.split('[')[1].split(']')[0]

            for out_name in ret_node.out_edges:
                if not out_name in scope_node.topology_list:
                    out_node = self._graph.get_node(out_name)

                    ret_name = ret_node.name + subscript
                    if ret_name in out_node.in_edges:
                        insert_pos = out_node.in_edges.index(ret_name)
                        insert_name = scope_node.name + \
                            ':{}'.format(str(ret_idx)) if len(
                                return_variable_names) > 1 else scope_node.name
                        out_node.in_edges.remove(ret_name)
                        out_node.in_edges.insert(insert_pos, insert_name)

                        # if out_node is scope node, replace the scope node's inner topology list node.
                        if out_node.type == 'Scope':
                            for n in out_node.topology_list:
                                n = self._graph.get_node(n)
                                if ret_name in n.in_edges:
                                    idx = n.in_edges.index(ret_name)
                                    n.in_edges.remove(ret_name)
                                    n.in_edges.insert(idx, insert_name)
            ret_idx += 1

        return return_variable_names


    ''' if the input params include tensor which is multi-output type(e.g. unstack), then we need 
    to check whether this scope function body uses only one of the outputs or multi outputs. if it is 
    the former, feed the selected one(e.g. unstack[0]), otherwise feed all. '''
    
    def _check_and_get_input_params(self, scope_node):

        def wipe_in_egde_idx(in_name, node):
            for idx, in_edge in enumerate(node.in_edges):
                if in_name in in_edge:
                    node.in_edges[idx] = in_edge.split(':')[0]
            node.in_edges = sorted(set(node.in_edges), key=node.in_edges.index)

        input_params = list()
        in_name_dict = collections.OrderedDict()
        for in_name in scope_node.in_edges:

            if self._graph.get_node(in_name).variable_name not in input_params:
                input_params.append(self._graph.get_node(
                    in_name).variable_name)
            if ':' not in in_name:
                continue

            if in_name_dict.get(in_name.split(':')[0], None):
                in_name_dict[in_name.split(':')[0]].add(
                    in_name.split(':')[1])
            else:
                in_name_dict[in_name.split(':')[0]] = set(
                    [in_name.split(':')[1]])

        for in_name, subscript_set in in_name_dict.items():
            # the input parameter shoule be sliced when call func.
            if len(subscript_set) == 1:

                # modify the in_edges in scope inner nodes. decline the :idx.
                for n in scope_node.topology_list:
                    n = self._graph.get_node(n)
                    wipe_in_egde_idx(in_name, n)
            else:  # >2
                wipe_in_egde_idx(in_name, scope_node)

        return input_params


    '''
    create a scope node.

    Args:
    scope_name: the name of this scope, will be assigned to scope pattern.
    scope_node_names: node names involved by this scope. 
    top_node: the top node in these scope nodes.
    '''
    
    def _create_scope_node(self, scope_name, scope_node_names, top_node):
        # 1. initilize scope node
        scope_node = self._initialize_scope_node(top_node)

        # 2. get scope nodes' topology list.
        scope_node.topology_list = self._get_scope_nodes_topology_list(
            scope_node_names)
        scope_node.pattern = scope_name

        # 3. rebuild the edges connection after folding these scope nodes into one node and 
        # get this scope node's return variables.
        scope_node.return_variables = self._rebuild_scope_edges_and_get_ret_vars(
            scope_node)

        # 4. rebuild graph.
        self._graph.layer_map[scope_node.name] = scope_node
        self._graph.layer_name_map[scope_node.name] = scope_node.name
        self._graph.rebuild()


    '''initialize a scope node by copying source_node's attrs.'''
    
    def _initialize_scope_node(self, source_node):
        scope_node = self._graph.model.node.add()
        scope_node.name = source_node.name + '_scope'
        scope_node.op = 'Scope'
        scope_node = IRGraphNode(scope_node)

        kwargs = {}
        kwargs['scope'] = source_node.get_attr('scope')

        if 'data_format' in source_node.layer.attr:
            kwargs['data_format'] = source_node.get_attr('data_format')

        if '_output_shapes' in source_node.layer.attr:
            scope_node.layer.attr["_output_shapes"].MergeFromString(
                source_node.layer.attr['_output_shapes'].SerializeToString())
        if 'value' in source_node.layer.attr:
            kwargs['value'] = source_node.get_attr('value')
        # RNN-related attrs.
        if 'input_size' in source_node.layer.attr:
            kwargs['input_size'] = source_node.get_attr('input_size')
        if 'num_units' in source_node.layer.attr:
            kwargs['num_units'] = source_node.get_attr('num_units')
        if 'fill_size' in source_node.layer.attr:
            kwargs['fill_size'] = source_node.get_attr('fill_size')
        if 'fill_value' in source_node.layer.attr:
            kwargs['fill_value'] = source_node.get_attr('fill_value')

        assign_IRnode_values(scope_node.layer, kwargs)
        return scope_node


    '''
    check whether same pattern scope node has same inputs and outputs.
    For thoese do not have, rename it's pattern by adding serial number suffix.
    '''
    
    def check_scope(self):
        name_no_dict = collections.OrderedDict()
        name_inp_out_dict = collections.OrderedDict()

        for name, ir_node in self._graph.layer_map.items():
            if ir_node.type == 'Scope':
                #get input params
                ir_node.input_params = self._check_and_get_input_params(ir_node)
                origi_pattern = re.sub(r'(_\d+)*$', '', ir_node.pattern)
                if name_inp_out_dict.get(origi_pattern, None):
                    inps_and_outs = name_inp_out_dict[origi_pattern]
                    exist_Equal = False
                    for inp_out in inps_and_outs:
                        if len(ir_node.input_params) == len(inp_out[0]) and len(ir_node.return_variables)== len(inp_out[1]):
                            exist_Equal = True
                            if inp_out[2]:
                                ir_node.pattern = ir_node.pattern + '_' + str(inp_out[2])

                    if not exist_Equal:
                        name_inp_out_dict[origi_pattern].append([ir_node.input_params, ir_node.return_variables, name_no_dict.get(origi_pattern, 1)])
                        ir_node.pattern = ir_node.pattern + '_' + str(name_no_dict.get(origi_pattern, 1))
                        name_no_dict[origi_pattern] = name_no_dict.get(origi_pattern, 1) + 1

                else:
                    name_inp_out_dict[origi_pattern] = [[ir_node.input_params, ir_node.return_variables, 0]]
                    name_no_dict[ir_node.pattern] = name_no_dict.get(origi_pattern, 0) + 1
    

    '''
    check whether same pattern scope node has same weights.
    For thoese do not have, rename it's pattern by adding serial number suffix.
    '''
    
    def check_weights(self):
        weight_related_ops = ['FullyConnected']
        pattern_weight_op = collections.OrderedDict()
        name_no_dict = collections.OrderedDict()
        pattern_weights = collections.OrderedDict()

        for ir_node_name in self._graph.topological_sort:
            ir_node = self._graph.get_node(ir_node_name)
            if ir_node.type == 'Scope':
                for inner_name in ir_node.topology_list:
                    if self._graph.get_node(inner_name).type in weight_related_ops:
                        if pattern_weight_op.get(ir_node.pattern, None):
                            if self._weights_dict[inner_name]['weights'].any() and pattern_weights.get(ir_node.pattern, None):
                                inner_weights = self._weights_dict[inner_name]['weights']
                                isExist = False
                                for idx, it in enumerate(pattern_weights[ir_node.pattern]):                                        
                                    if np.array_equal(inner_weights, it['weights']):
                                        ir_node.pattern = ir_node.pattern + '_'+ str(idx)
                                        isExist = True
                                        break
                                if isExist:
                                    continue
                            pattern_weight_op[ir_node.pattern].add(inner_name)
                            pattern_weights[ir_node.pattern].append(self._weights_dict[inner_name])
                            ir_node.pattern = ir_node.pattern + '_'+ str(len(pattern_weights[ir_node.pattern])-1)

                        else:
                            pattern_weight_op[ir_node.pattern] = set([inner_name])
                            if self._weights_dict.get(inner_name, None):
                                pattern_weights[ir_node.pattern] = [self._weights_dict[inner_name]]
                                ir_node.pattern = ir_node.pattern + '_'+ str(name_no_dict.get(ir_node.pattern, 0))

