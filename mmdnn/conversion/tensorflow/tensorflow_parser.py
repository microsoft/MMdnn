#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import tensorflow
import numpy as np
from mmdnn.conversion.tensorflow.tensorflow_graph import TensorflowGraph
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.DataStructure.parser import Parser
from tensorflow.python.framework import tensor_util


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
        0 : graph_pb2.DT_UNDEFINED,
        1 : graph_pb2.DT_FLOAT32,
        2 : graph_pb2.DT_FLOAT64,
        3 : graph_pb2.DT_INT32,
        4 : graph_pb2.DT_UINT8,
        5 : graph_pb2.DT_INT16,
        6 : graph_pb2.DT_INT8,
        7 : graph_pb2.DT_STRING,
        9 : graph_pb2.DT_INT64        
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


    @staticmethod
    def _load_weights(model_weight_path):
        """Load a tensorflow checkpoint file from disk

        Parameters
        ----------
        model_weight_path: str
            Path where the weight path is (checkpoint file)
        
        Returns
        -------
        model: tensor name --> ndarry
        """
        reader = tensorflow.train.NewCheckpointReader(model_weight_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        data = dict()
        for name in var_to_shape_map:
            tensor = reader.get_tensor(name)
            data[name] = tensor

        print ("Tensorflow checkpoint file [%s] loaded successfully. [%d] variables loaded." % (model_weight_path, len(data)))
        return data
        
    
    @staticmethod
    def _get_scopes(layer_name):
        return layer_name.split('/')

    
    def _convert_reduction_operators(self, source_node, new_op = None):
        IR_node = self._convert_identity_operation(source_node, 1, new_op)

        # keep dims
        IR_node.attr['keepdims'].b = source_node.layer.attr['keep_dims'].b

        # axes
        axes = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor
        axes = tensor_util.MakeNdarray(axes)
        IR_node.attr['axes'].list.i.extend(axes)
    
    
    def _convert_padding(self, source_node, kernel_shape):
        pad = source_node.layer.attr["padding"].s
        if pad == "SAME"：
            input_node = self.tf_graph.get_parent(source_node.name, [0])
            input_shape = [dim.size for dim in input_node.layer.attr["_output_shapes"].list.shape[0].dim]
            cnt = 0
            if source_node.layer.attr["data_format"].s == 'channels_last':
                input_shape = input_shape[1:-1]
                cnt = 1
            else:
                input_shape = input_shape[2:]
                cnt = 2
            pad_along = list()
            for dims in input_shape:
                if (dims % source_node.layer.attr["strides"].list.i[cnt] == 0):
                    pad_along.append(max(kernel_shape[cnt] - source_node.layer.attr["strides"].list.i[cnt], 0))
                else:
                    pad_along.append(max(kernel_shape[cnt] - (dims % source_node.layer.attr["strides"].list.i[cnt]), 0))
                cnt += 1
            pad_along_first = list()
            pad_along_second = list()
            for i in range(len(pad_along)):
                pad_along_first[i] = pad_along[i] // 2
                pad_along_second[i] = pad_along[i] - pad_along_second[i]
            return list([0, 0] + pad_along_first + pad_along_second + [0, 0])
        elif pad == "VALID":
            return list([0, 0] * len(source_node.layer.attr["strides"].list.i))
        else:
            raise ValueError("Padding format of [{}] is not supported".format(pad))


    def _convert_layers_batchnorm(self, source_node):
        # name, op
        IR_node = self.IR_graph.node.add()
        TensorflowParser._copy_and_reop(source_node, IR_node, 'BatchNorm')
        
        # epsilon
        epsilon = self.get_parent(source_node.name, [1])
        IR_node.attr['epsilon'].f = epsilon.layer.attr['value'].tensor.float_val[0]

        # moving variance (var)
        moving_variance = self.get_parent(source_node.name, [0, 0])
        if self.weight_loaded:
            self.set_weight(source_node.name, 'var', self.ckpt_data[moving_variance.name])

        # gamma (scale)
        gamma = self.get_son(source_node.name, [0, 0], True)
        gamma = self.get_parent(gamma.name, [1, 0], True)        
        if gamma is None or not gamma.type.startswith('Variable'):
            IR_node.attr['scale'].b = False
            output_node = self.get_son(source_node.name, [0, 0, 0], True)
        else:
            IR_node.attr['scale'].b = True            
            if self.weight_loaded:
                self.set_weight(source_node.name, 'scale', self.ckpt_data[gamma.name])
            output_node = self.get_son(source_node.name, [0, 0, 0, 0], True)

        # mean        
        mean = self.get_parent(output_node.name, [1, 1, 0, 0], True)
        if self.weight_loaded:
            self.set_weight(source_node.name, 'mean', self.ckpt_data[mean.name])

        # bias
        bias = self.get_parent(output_node.name, [1, 0, 0], True)
        if bias is None or not bias.type.startswith('Variable'):
            IR_node.attr['bias'].b = False
        else:
            IR_node.attr['bias'].b = True
            if self.weight_loaded:
                self.set_weight(source_node.name, 'bias', self.ckpt_data[bias.name])

        # input node
        assert output_node.type == 'Add'
        input_node = self.get_parent(output_node.name, [0, 0])
        IR_node.input.append(input_node.real_name)

        # output node        
        output_node.real_name = source_node.name


    def __init__(self, input_args, dest_nodes = None):
        super(TensorflowParser, self).__init__()

        # load model files into Keras graph
        from six import string_types as _string_types
        if isinstance(input_args, _string_types):
            model = TensorflowParser._load_meta(input_args)
        elif isinstance(input_args, tuple):
            model = TensorflowParser._load_meta(input_args[0])
            self.ckpt_data = TensorflowParser._load_weights(input_args[1])
            self.weight_loaded = True

        if dest_nodes != None:
            from tensorflow.python.framework.graph_util import extract_sub_graph
            model = extract_sub_graph(model, dest_nodes.split(','))
        
        # Build network graph
        self.tf_graph =  TensorflowGraph(model)
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

    def _convert_pooling(self, source_node, pool_type):        
        IR_node = self._convert_identity_operation(source_node, new_op = 'Pool')
                
        # strides
        IR_node.attr['strides'].list.i[:] = source_node.layer.attr['strides'].list.i[:]

        # data_format
        IR_node.attr['data_format'].s = source_node.layer.attr['data_format'].s

        # window_shape
        IR_node.attr['window_shape'].list.i[:] = source_node.layer.attr['ksize'].list.i[:]
        
        # padding
        IR_node.attr['pads'].list.i[:] = self._convert_padding(source_node, IR_node.attr["window_shape"].list.i)

        # pool type
        IR_node.attr['pooling_type'].s = pool_type
    
    def gen_IR(self):        
        for layer in self.src_graph.topological_sort:
            current_node = self.src_graph.get_node(layer)

            if self._skip_node(current_node):
                continue

            node_type = current_node.type

            if hasattr(self, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                func(current_node)
            else:                
                self.rename_UNKNOWN(current_node)
            

    @staticmethod
    def _copy_and_reop(source_node, IR_node, new_op = None):        
        if new_op == None: new_op = source_node.type
        IR_node.name = source_node.name
        IR_node.op = new_op
        if 'dtype' in source_node.layer.attr:            
            assert source_node.layer.attr['dtype'].type in TensorflowParser.dtype_map, 'type [{}] is unknown.'.format(source_node.layer.attr['dtype'].type)
            IR_node.attr["dtype"].type = TensorflowParser.dtype_map[source_node.layer.attr['dtype'].type]

        if '_output_shapes' in source_node.layer.attr:
            IR_node.attr["_output_shapes"].MergeFromString(source_node.layer.attr['_output_shapes'].SerializeToString())

        
    def _convert_inedge(self, source_node, IR_node, start_idx = 0, end_idx = None):
        if end_idx == None: end_idx = len(source_node.in_edges) 
        for idx in range(start_idx, end_idx):
            IR_node.input.append(self.src_graph.get_node(source_node.in_edges[idx]).real_name)

        
    def _get_bias(self, source_node, IR_node):
        if len(source_node.out_edges) < 1:
            return
        
        add_node = self.tf_graph.get_node(source_node.out_edges[0])        
        if add_node.type != "Add" and add_node.type != "BiasAdd":
            return

        variable = self.tf_graph.get_node(add_node.in_edges[1])
        variable = self.tf_graph.get_node(variable.in_edges[0])
        
        assert variable.layer.attr['shape'].shape.dim[0].size == IR_node.attr['filter'].list.i[-1]        

        if self.weight_loaded:
            assert variable.name in self.ckpt_data
            current_layer = self.weights[source_node.name]
            current_layer['bias'] = self.ckpt_data[variable.name]

        add_node.real_name = IR_node.name
        add_node.covered = True
        IR_node.attr['use_bias'].b = True
        
    
    @staticmethod
    def _copy_shape(source_node, IR_node):
        assert 'shape' in source_node.layer.attr        
        IR_node.attr['shape'].shape.MergeFromString(source_node.layer.attr['shape'].shape.SerializeToString())


    def rename_UNKNOWN(self, source_node):        
        if source_node.type in self.skip_type:
            return
        print("Tensorflow has not supported operator [%s] with name [%s]." % (source_node.type, source_node.name))
        return

    
    def rename_Placeholder(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'DataInput')

        # shape
        TensorflowParser._copy_shape(source_node, IR_node)

    
    def rename_Conv2D(self, source_node):
        """
        weights: name_weights, name_bias
        """
        IR_node = self._convert_identity_operation(source_node, 1, 'Convolution')
        
        # strides
        IR_node.attr['strides'].list.i[:] = source_node.layer.attr['strides'].list.i[:]

        # data_format
        IR_node.attr['data_format'].s = source_node.layer.attr['data_format'].s

        # input[1] : W
        # filter
        W = self.tf_graph.get_node(source_node.layer.input[1])
        W = self.tf_graph.get_node(W.layer.input[0]).layer
        for e in W.attr['shape'].shape.dim:
            IR_node.attr['filter'].list.i.append(e.size)

        # padding
        IR_node.attr['pads'].list.i[:] = self._convert_padding(source_node, IR_node.attr["filter"].list.i)
                
        if self.weight_loaded:            
            self.set_weight(source_node.name, 'weights', self.ckpt_data[W.name])

        # output[0] : B
        self._get_bias(source_node, IR_node)

    
    def _convert_identity_operation(self, source_node, in_edge_count = None, new_op = None):
        IR_node = self.IR_graph.node.add()
        TensorflowParser._copy_and_reop(source_node, IR_node, new_op)
        self._convert_inedge(source_node, IR_node, 0, in_edge_count)
        return IR_node
    
    
    def rename_Relu(self, source_node):
        self._convert_identity_operation(source_node)

    
    def rename_Relu6(self, source_node):
        self._convert_identity_operation(source_node)
        

    def rename_Add(self, source_node):
        if not source_node.covered:
            scopes = self._get_scopes(source_node.name)
            if scopes[-2] == 'dropout':
                # converted [dropout]
                pass

            if scopes[-2] == 'batchnorm':
                # convert [tf.contrib.layers.batch_norm]
                self._convert_layers_batchnorm(source_node)

            else:
                # normal Add
                self._convert_identity_operation(source_node)
    

    def rename_Sub(self, source_node):
        self._convert_identity_operation(source_node)
            

    def rename_Reshape(self, source_node):
        IR_node = self._convert_identity_operation(source_node, 1)

        # for target shape      
        IR_node.attr["shape"].shape.MergeFromString(source_node.layer.attr['_output_shapes'].list.shape[0].SerializeToString())


    def rename_MatMul(self, source_node):
        """
        weights: name_weights, name_bias
        """
        IR_node = self._convert_identity_operation(source_node, 1)
        
        # units
        units = source_node.layer.attr['_output_shapes'].list.shape[-1].dim[-1].size
        IR_node.attr['units'].i = units

        # Weights
        W = self.tf_graph.get_node(self.tf_graph.get_node(source_node.in_edges[1]).in_edges[0])        
        if self.weight_loaded:
            self.set_weight(source_node.name, 'weights', self.ckpt_data[W.name])

        add_node = self.tf_graph.get_node(source_node.out_edges[0])
        if add_node.type == 'Add':
            add_node.covered = True
            add_node.real_name = source_node.real_name
            # FullyConnected Layer
            # name, op
            TensorflowParser._copy_and_reop(source_node, IR_node, 'FullyConnected')            

            # get Bias
            B = self.tf_graph.get_node(self.tf_graph.get_node(source_node.out_edges[0]).in_edges[1]).in_edges[0]
            if self.weight_loaded:
                self.set_weight(source_node.name, 'bias', self.ckpt_data[B])
            IR_node.attr['use_bias'].b = True
        else:
            assert False


    def rename_RealDiv(self, source_node):
        scopes = self._get_scopes(source_node.name)
        
        # Deal Dropout
        if scopes[-2] == 'dropout':
            IR_node = self._convert_identity_operation(source_node, 1, 'Dropout')

            # keep prob
            if 'value' in self.tf_graph.get_node(source_node.layer.input[1]).layer.attr:
                IR_node.attr['keep_prob'].f = self.tf_graph.get_node(source_node.layer.input[1]).layer.attr['value'].tensor.float_val[0]
            else:
                IR_node.attr['keep_prob'].f = 1.0

            # Remove nodes
            # Mul
            mul_node = self.tf_graph.get_node(source_node.out_edges[0])
            assert mul_node.type == "Mul"
            mul_node.covered = True
            mul_node.real_name = source_node.name

            # Floor
            floor_node = self.tf_graph.get_node(mul_node.in_edges[1])
            assert floor_node.type == "Floor"
            floor_node.covered = True
        else:        
            assert False


    def rename_Floor(self, source_node):
        scopes = self._get_scopes(source_node.name)
        assert scopes[-2] == 'dropout'


    def rename_MaxPool(self, source_node):
        self._convert_pooling(source_node, b'MAX')


    def rename_AvgPool(self, source_node):
        self._convert_pooling(source_node, b'AVG')


    def rename_Identity(self, source_node):
        source_node.real_name =  self.src_graph.get_node(source_node.in_edges[0]).real_name        


    def rename_Squeeze(self, source_node):
        IR_node = self._convert_identity_operation(source_node)
        IR_node.attr['axes'].MergeFromString(source_node.layer.attr['squeeze_dims'].SerializeToString())


    def rename_QueueDequeueManyV2(self, source_node):
        IR_node = self._convert_identity_operation(source_node, 0, 'DataInput')
        IR_node.attr['shape'].shape.MergeFromString(source_node.layer.attr['_output_shapes'].list.shape[0].SerializeToString())
        IR_node.attr['shape'].shape.dim[0].size = -1
        IR_node.attr['dtype'].type = self.dtype_map[source_node.layer.attr['component_types'].list.type[0]]


    def rename_Pad(self, source_node):        
        IR_node = self._convert_identity_operation(source_node, 1, 'Pad')
        IR_node.attr['mode'].s = b'CONSTANT'
        IR_node.attr['constant_values'].f = 0.0
        
        # paddings
        padding = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor

        shapes = tensor_util.MakeNdarray(padding)
        for i in shapes:
            for j in i:
                IR_node.attr['paddings'].list.i.append(j)


    def rename_Mean(self, source_node):
        self._convert_reduction_operators(source_node, new_op = 'ReduceMean')

    
    def rename_ConcatV2(self, source_node):
        n = len(source_node.in_edges) - 1
        IR_node = self._convert_identity_operation(source_node, n, 'Concat')
        axis = self.tf_graph.get_parent(source_node.name, [n])
        IR_node.attr['axis'].i = axis.layer.attr['value'].tensor.int_val[0]


    def rename_DepthwiseConv2dNative(self, source_node):
        IR_node = self._convert_identity_operation(source_node, 1, 'DepthwiseConv')
        IR_node.attr['strides'].list.i[:] = source_node.layer.attr['strides'].list.i[:]
        IR_node.attr['data_format'].s = source_node.layer.attr['data_format'].s

        input_node = self.src_graph.get_parent(source_node.name, [1])
        IR_node.attr['filter'].list.i.extend([dim.size for dim in input_node.layer.attr['_output_shapes'].list.shape[0].dim])
        IR_node.attr['pads'].list.i[:] = self._convert_padding(source_node, IR_node.attr["filter"].list.i)

        if self.weight_loaded:
            weight = self.src_graph.get_parent(source_node.name, [1, 0])
            self.set_weight(source_node.name, 'weights', self.ckpt_data[weight.name])