#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import numpy as np
import tensorflow
from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import attr_value_pb2
from mmdnn.conversion.tensorflow.tensorflow_graph import TensorflowGraph
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.utils import *
from mmdnn.conversion.common.DataStructure.parser import Parser


class TensorflowParser(Parser):

    skip_prefix = [
        "^",
        "train_op",
        "save",
        "gradients",
        "init",
        "global_step",
        "distort_image",
        "Adagrad",
    ]

    skip_scope = [
        "random_uniform",
        "Initializer",
        "optimizer",
        "weight_loss",
        "parallel_read",
        "case"
    ]

    skip_type = set([
        "L2Loss",
        "VariableV2",
        "Const",
        "Assign",
        "RandomUniform",
        "FIFOQueueV2"
    ])

    dtype_map = {
        0  : graph_pb2.DT_UNDEFINED,
        1  : graph_pb2.DT_FLOAT32,
        2  : graph_pb2.DT_FLOAT64,
        3  : graph_pb2.DT_INT32,
        4  : graph_pb2.DT_UINT8,
        5  : graph_pb2.DT_INT16,
        6  : graph_pb2.DT_INT8,
        7  : graph_pb2.DT_STRING,
        9  : graph_pb2.DT_INT64,
        10 : graph_pb2.DT_BOOL,
    }


    @property
    def src_graph(self):
        return self.tf_graph


    @staticmethod
    def _load_meta(model_network_path):
        """Load a tensorflow meta file from disk

        Parameters
        ----------
        model_network_path: str
            Path where the model network path is (protobuf meta file)

        Returns
        -------
        model: A tensorflow protobuf file
        """
        from tensorflow.core.protobuf import meta_graph_pb2
        from mmdnn.conversion.common.IR.IR_graph import load_protobuf_from_file

        meta_graph = meta_graph_pb2.MetaGraphDef()
        load_protobuf_from_file(meta_graph, model_network_path)
        graph = meta_graph.graph_def

        print ("Tensorflow model file [%s] loaded successfully." % model_network_path)
        return graph


    def _convert_identity_operation(self, source_node, start_edge_id = 0, in_edge_count = None, new_op = None):
        IR_node = self.IR_graph.node.add()
        TensorflowParser._copy_and_reop(source_node, IR_node, new_op)
        self._convert_inedge(source_node, IR_node, start_edge_id, in_edge_count)
        return IR_node


    @staticmethod
    def _get_scopes(layer_name):
        return layer_name.split('/')


    def __init__(self, meta_file, checkpoint_file, frozen_file, dest_nodes = None):
        super(TensorflowParser, self).__init__()

        # load model files into TensorFlow graph
        if meta_file:
            model = TensorflowParser._load_meta(meta_file)

        if checkpoint_file:
            self.ckpt_data = TensorflowParser._load_weights(checkpoint_file)
            self.weight_loaded = True

        if dest_nodes != None:
            from tensorflow.python.framework.graph_util import extract_sub_graph
            model = extract_sub_graph(model, dest_nodes.split(','))

        # Build network graph
        self.tf_graph = TensorflowGraph(model)
        self.tf_graph.build()


    @classmethod
    def _skip_node(cls, source_node):
        if source_node.covered:
            return True

        for prefix in cls.skip_prefix:
            if source_node.name.startswith(prefix):
                return True

        scopes = TensorflowParser._get_scopes(source_node.name)

        for s in scopes:
            if s in cls.skip_scope:
                return True

        return False


    def gen_IR(self):
        for layer in self.src_graph.topological_sort:
            current_node = self.src_graph.get_node(layer)

            if self._skip_node(current_node):
                continue

            node_type = current_node.type

            self.rename_UNKNOWN(current_node)


    @staticmethod
    def _copy_and_reop(source_node, IR_node, new_op = None):
        if new_op == None: new_op = source_node.type
        IR_node.name = source_node.name
        IR_node.op = new_op

        kwargs = {}
        if 'data_format' in source_node.layer.attr:
            kwargs['data_format'] = source_node.get_attr('data_format')

        if 'dtype' in source_node.layer.attr:
            assert source_node.layer.attr['dtype'].type in TensorflowParser.dtype_map, 'type [{}] is unknown.'.format(source_node.layer.attr['dtype'].type)
            IR_node.attr["dtype"].type = TensorflowParser.dtype_map[source_node.layer.attr['dtype'].type]

        if '_output_shapes' in source_node.layer.attr:
            IR_node.attr["_output_shapes"].MergeFromString(source_node.layer.attr['_output_shapes'].SerializeToString())

        assign_IRnode_values(IR_node, kwargs)


    def _convert_inedge(self, source_node, IR_node, start_idx = 0, end_idx = None):
        if end_idx == None: end_idx = len(source_node.in_edges) - start_idx
        for idx in range(start_idx, end_idx + start_idx):
            IR_node.input.append(self.src_graph.get_node(source_node.in_edges[idx]).real_name)


    @staticmethod
    def _copy_shape(source_node, IR_node):
        assert 'shape' in source_node.layer.attr
        IR_node.attr['shape'].shape.MergeFromString(source_node.layer.attr['shape'].shape.SerializeToString())


    def rename_UNKNOWN(self, source_node):
        if source_node.type in self.skip_type:
            return
        self._convert_identity_operation(source_node)
