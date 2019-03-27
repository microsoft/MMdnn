from mmdnn.conversion.rewriter.rewriter import UnitRewriterBase
import numpy as np
import re

class GRURewriter(UnitRewriterBase):

    def __init__(self, graph, weights_dict):
        return super(GRURewriter, self).__init__(graph, weights_dict)
    
    def process_gru_cell(self, match_result):
        if 'gru_cell' not in match_result._pattern_to_op.keys():
            return
        kwargs = dict()
        top_node = match_result._pattern_to_op[match_result._name_to_pattern['gru_cell']]

        w_e = match_result.get_op("cell_kernel")
        w = self._weights_dict[w_e.name.replace('/read', '')]

        num_units = w.shape[1]//2
        input_size = w.shape[0] - num_units

        kwargs['num_units'] = num_units
        kwargs['input_size'] = input_size

        if hasattr(top_node, 'kwargs'):
            top_node.kwargs.update(kwargs)
        else:
            top_node.kwargs = kwargs


    def process_rnn_h_zero(self, match_result):
        if 'h_zero' not in match_result._name_to_pattern.keys():
            return
        kwargs = dict()
        top_node = match_result._pattern_to_op[match_result._name_to_pattern['h_zero']]

        fill_size = match_result.get_op('fill_size')
        fill_value = match_result.get_op('fill_value')

        kwargs['fill_size'] = fill_size.get_attr('value').int_val[0]
        kwargs['fill_value'] = fill_value.get_attr('value').float_val[0]

        if hasattr(top_node, 'kwargs'):
            top_node.kwargs.update(kwargs)
        else:
            top_node.kwargs = kwargs


    def process_match_result(self, match_result, pattern_name):
        if pattern_name == 'gru_cell':
            self.process_gru_cell(match_result)
        elif pattern_name == 'h_zero':
            if self.check_match_scope(match_result, 'GRUCellZeroState'):
                self.process_rnn_h_zero(match_result)

    '''For some short pattern, to avoid match other pattern, check it's scope'''
    def check_match_scope(self, match_result, scope_name):
        ops = match_result._pattern_to_op.values()

        for op in ops:
            op_name_splits = op.name.split('/')
            if len(op_name_splits) < 2:
                return False
            if re.sub(r'(_\d+)*$', '', op_name_splits[-2]) != scope_name:
                if len(op_name_splits) > 2:
                    if re.sub(r'(_\d+)*$', '', op_name_splits[-3]) != scope_name:
                        return False
                else:
                    return False
        return True


    def run(self):
        return super(GRURewriter, self).run(['gru_cell', 'h_zero'], 'tensorflow')