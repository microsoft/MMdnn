# ----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
from six import string_types as _string_types

import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType


class Emitter(object):

    def __init__(self):
        self.body_code = str()
        self.weights_dict = dict()
        self.used_layers = set()
        self.weight_loaded = False
        self.layers_codes = dict()

    def run(self, dstNetworkPath, dstWeightPath=None, phase='test'):
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
            self.weights_dict = np.load(file_name,
                                        allow_pickle=True).item()  # made default False in response to CVE-2019-6446
        except:
            self.weights_dict = np.load(file_name, encoding='bytes', allow_pickle=True).item()

    def parent_variable_name(self, IR_node, path_or_name=[0]):
        if isinstance(path_or_name, _string_types):
            path = [IR_node.in_edges.index(path_or_name)]
        elif isinstance(path_or_name, list):
            path = path_or_name
        else:
            raise ValueError
        return self.IR_graph.get_parent_variable_name(IR_node.name, path)

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
