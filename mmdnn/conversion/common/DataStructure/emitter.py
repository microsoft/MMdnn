#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------          
from six import string_types as _string_types

import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType


class Emitter(object):
    
    def __init__(self):        
        self.body_code = str()
        self.weights_dict = dict()
        self.used_layers = set()
        self.weight_loaded = False


    def run(self, dstNetworkPath, dstWeightPath = None, phase = 'test'):
        self.save_code(dstNetworkPath, phase)


    # share functions
    def add_body(self, indent, codes):
        if isinstance(codes, _string_types):
            codes = [codes]
        for code in codes:
            self.body_code += ("    " * indent) + code + '\n'

    def _load_weights(self, file_name=None):
        import numpy as np
        self.weight_loaded = True
        try:
            self.weights_dict = np.load(file_name).item()
        except:
            self.weights_dict = np.load(file_name, encoding='bytes').item()


    def parent_variable_name(self, IR_node, path = [0]):
        return self.IR_graph.get_parent(IR_node.name, path).real_variable_name


    '''input_ids is a list of parent node's input index for current node and it's subscript, like
        [input_idx_1, subscript_1, input_idx_2, subscript_2]
    '''
    def _parent_output_idx(self, IR_node, parent_name_or_idx=None):
        if parent_name_or_idx is None:
            parent_idx = 0
        else:
            if isinstance(parent_name_or_idx, int):
                parent_idx = parent_name_or_idx
            elif isinstance(parent_name_or_idx, _string_types):
                parent_idx = IR_node.in_edges.index(parent_name_or_idx)
            else:
                raise ValueError("parent_name_or_idx should be int index or string name!")

        input_ids = IR_node.get_attr('input_ids')

        if input_ids is None:
            output_idx = -1
        elif parent_idx in input_ids[::2]:
            output_idx = input_ids[2 * input_ids[::2].index(parent_idx) + 1]
        else:
            output_idx = -1 #num of parent node's output is one.

        return output_idx


    def parent_variable_idx_name(self, IR_node, parent_name_or_ids=None):
        
        if parent_name_or_ids is None:
            parent_ids = [0]
        else:
            if isinstance(parent_name_or_ids, list):
                parent_ids = parent_name_or_ids
            elif isinstance(parent_name_or_ids, _string_types):
                parent_ids = [IR_node.in_edges.index(parent_name_or_ids)]
            else:
                raise ValueError("It should be list indies or string name!")
        
        parent_variable_name = self.parent_variable_name(IR_node, parent_ids)

        '''if len(parent_ids) is 1, parent_IR_node is IR_node'''
        parent_IR_node = self.IR_graph.get_parent(IR_node.name, parent_ids[:-1])
        parent_output_idx = self._parent_output_idx(parent_IR_node, parent_ids[-1])
        
        if parent_output_idx == -1:
            return parent_variable_name
        else:
            return parent_variable_name+'[{}]'.format(parent_output_idx)


    def _build(self):
        self.IR_graph.build()
    
    
    def gen_code(self, phase):
        raise NotImplementedError("do not use base emitter class.")


    def save_code(self, filepath, phase):
        code = self.gen_code(phase)
        with open(filepath, 'w') as fout:
            fout.write(code)
        print("Target network code snippet is saved as [{}].".format(filepath))


    @staticmethod
    def save_weights(weights, filename):
        import numpy as np
        with open(filename, 'wb') as of:
            np.save(of, weights)
        print("Target weights are saved as [{}].".format(filename))
        


    @staticmethod
    def _image_in_transpose_str(dim):
        dims = [dim]
        dims.extend(range(dim))
        return ','.join('%s' % id for id in dims)


    @staticmethod
    def _image_out_transpose_str(dim):
        dims = list(range(1, dim + 1))
        dims.append(0)
        return ','.join('%s' % id for id in dims)


    @staticmethod
    def _conv_kernel_transpose_str(dim):
        dims = [dim + 1, dim]
        dims.extend(range(dim))
        return ','.join('%s' % id for id in dims)