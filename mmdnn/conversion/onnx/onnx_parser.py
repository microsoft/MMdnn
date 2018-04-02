# ----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

from mmdnn.conversion.common.DataStructure.parser import Parser
from mmdnn.conversion.onnx.onnx_graph import ONNXGraph


class ONNXParser(Parser):
    skip_type = set()

    @property
    def src_graph(self):
        return self.onnx_graph

    @staticmethod
    def _load_model(model_file):
        """Load a ONNX model file from disk

        Parameters
        ----------
        model_file: str
            Path where the model file path is (protobuf file)

        Returns
        -------
        model: A ONNX protobuf model
        """
        from onnx import onnx_pb2
        from mmdnn.conversion.common.IR.IR_graph import load_protobuf_from_file

        model = onnx_pb2.ModelProto()
        load_protobuf_from_file(model, model_file)

        print("ONNX model file [%s] loaded successfully." % model_file)
        return model

    def __init__(self, model_file):
        super(ONNXParser, self).__init__()

        model = ONNXParser._load_model(model_file)
        self.onnx_graph = ONNXGraph(model)
        self.onnx_graph.build()
        self.weight_loaded = True

    def rename_Conv(self, source_node):

    def rename_UNKNOWN(self, source_node):
        if source_node.type in self.skip_type:
            return
        print("ONNX has not supported operator [%s] with name [%s]."
              % (source_node.type, source_node.name))
        return

    def gen_IR(self):
        # if node len(in_edges), generate additional DataInput node

        # print
        for layer in self.src_graph.topological_sort:
            current_node = self.src_graph.get_node(layer)
            node_type = current_node.type
            if hasattr(self, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                func(current_node)
            else:
                self.rename_UNKNOWN(current_node)
