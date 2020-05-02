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
from tensorflow.tools.graph_transforms import TransformGraph
from mmdnn.conversion.rewriter.utils import *
import tempfile
import os
import shutil


class TensorflowParser(Parser):

    skip_prefix = [
        "^",
        "train_op",
        "save",
        "gradients",
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
        19 : graph_pb2.DT_FLOAT16
    }


    @property
    def src_graph(self):
        return self.tf_graph


    @staticmethod
    def _shapeToStr(shapes):
        return [dim.size if dim.size > 0 else 1 for dim in shapes.dim]


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
            name_seg = name.split("/")
            if name_seg[-1] == "ExponentialMovingAverage":
                name = "/".join(name_seg[:-1])
            data[name] = tensor

        print ("Tensorflow checkpoint file [%s] loaded successfully. [%d] variables loaded." % (model_weight_path, len(data)))
        return data


    @staticmethod
    def _get_scopes(layer_name):
        return layer_name.split('/')


    def check_const(self, node):
        while node:
            if node.type == "Const":
                return node
            elif node.type == "NoOp":
                return None
            else:
                node =  self.get_parent(node.name, [0])

    def _add_constant_node(self, source_node):
        parent_ids=range(len(source_node.in_edges))
        for idx in parent_ids:
            parent_node = self.tf_graph.get_node(source_node.in_edges[idx])
            if parent_node.type == 'Const':
                self._rename_Const(parent_node)
    
    def _rename_Const(self, source_node):
        IR_node = self._convert_identity_operation(source_node, in_edge_count=0, new_op='Constant') # Constant
        value = source_node.get_attr('value')
        if value.float_val:
            shape = tuple(self.tensor_shape_to_list(value.tensor_shape))
            value = np.full(shape, value.float_val[0])
        elif value.int_val:
            shape = tuple(self.tensor_shape_to_list(value.tensor_shape))
            value = np.full(shape, value.int_val[0])
        else:
            value = np.array(tensor_util.MakeNdarray(value).tolist())
        
        if value.ndim > 1:
            self.set_weight(source_node.name, 'value', value)
        else:
            kwargs = {'value': value}
            assign_IRnode_values(IR_node, kwargs)


    def _convert_reduction_operators(self, source_node, new_op = None):
        IR_node = self._convert_identity_operation(source_node, 0, 1, new_op)

        # keep dims
        IR_node.attr['keepdims'].b = source_node.layer.attr['keep_dims'].b

        # axes
        axes = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor
        axes = tensor_util.MakeNdarray(axes)
        IR_node.attr['axes'].list.i.extend(axes)


    def _convert_layers_batchnorm(self, source_node):
        # name, op
        IR_node = self.IR_graph.node.add()
        TensorflowParser._copy_and_reop(source_node, IR_node, 'BatchNorm')

        # epsilon
        epsilon = self.get_parent(source_node.name, [1])
        IR_node.attr['epsilon'].f = epsilon.layer.attr['value'].tensor.float_val[0]

        # moving variance (var)
        moving_variance = self.get_parent(source_node.name, [0, 0])
        # print(moving_variance.name)
        if self.weight_loaded and moving_variance.name in self.ckpt_data.keys():
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
            if output_node.type == 'Sub':
                output_node = self.get_son(source_node.name, [0, 0, 1, 0], True)

        # mean
        mean = self.get_parent(output_node.name, [1, 1, 0, 0], True)
        if self.weight_loaded and mean.name in self.ckpt_data.keys():
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


    def __init__(self, meta_file, checkpoint_file, dest_nodes, inputShape = None, in_nodes = None):
        super(TensorflowParser, self).__init__()

        # load model files into TensorFlow graph
        if meta_file:
            model = TensorflowParser._load_meta(meta_file)

        if checkpoint_file:
            self.ckpt_data = TensorflowParser._load_weights(checkpoint_file)
            self.weight_loaded = True

        # extract subgraph using in_nodes and dest_nodes
        if in_nodes != None and inputShape != None:
            from tensorflow.python.tools import strip_unused_lib
            from tensorflow.python.framework import dtypes
            from tensorflow.python.platform import gfile
            model = strip_unused_lib.strip_unused(
                    input_graph_def = model,
                    input_node_names = in_nodes,
                    output_node_names = dest_nodes,
                    placeholder_type_enum = dtypes.float32.as_datatype_enum)

            input_list = [None]
            for i in range(len(inputShape)):
                input_list.append(tensorflow.Dimension(inputShape[i]))
            tensor_input = tensorflow.TensorShape(input_list)
            # Build network graph
            self.tf_graph = TensorflowGraph(model)
            for node in self.tf_graph.model.node:
                if node.name in in_nodes:
                    node.attr['shape'].shape.CopyFrom(tensor_input.as_proto())
                    node.attr['_output_shapes'].list.shape.pop()  #unknown_rank pop
                    node.attr['_output_shapes'].list.shape.extend([tensor_input.as_proto()])

        # extract subgraph using dest_nodes
        elif dest_nodes != None:
            from tensorflow.python.framework.graph_util import extract_sub_graph
            model = extract_sub_graph(model, dest_nodes)

        #  Get input node name
        if not in_nodes:
            in_nodes = []
            for node in model.node:
                if node.op == 'Placeholder':
                    in_nodes.append(node.name)

        # Graph Transform
        transforms = ["fold_constants(ignore_errors=true)"]
        transformed_graph_def = TransformGraph(model, in_nodes,
                                                dest_nodes, transforms)
        in_type_list = {}
        in_shape_list = {}

        for n in transformed_graph_def.node:
            if n.name in in_nodes:
                in_type_list[n.name] = n.attr['dtype'].type
                in_node_shape = n.attr['shape'].shape
                in_node_shape_str = self._shapeToStr(in_node_shape)
                in_shape_list[n.name] = in_node_shape_str

        dtype = tensorflow.float32
        with tensorflow.Graph().as_default() as g:
            input_map = {}
            for in_node in in_nodes:
                if in_type_list[in_node] == 1 or in_type_list[in_node] == 0:
                    dtype = tensorflow.float32

                elif in_type_list[in_node] == 3:
                    dtype = tensorflow.int32

                elif in_type_list[in_node] == 10:
                    dtype = tensorflow.bool
                
                x = tensorflow.placeholder(dtype, shape = in_shape_list[in_node])
                input_map[in_node] = x

            tensorflow.import_graph_def(transformed_graph_def, name='', input_map=input_map)

        with tensorflow.Session(graph = g) as sess:
            tempdir = tempfile.mkdtemp()
            meta_graph_def = tensorflow.train.export_meta_graph(filename=os.path.join(tempdir, 'my-model.meta'))
            model = meta_graph_def.graph_def
            shutil.rmtree(tempdir)

        self.tf_graph = TensorflowGraph(model)
        self.tf_graph.build()

        process_graph(self.tf_graph, self.ckpt_data)

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

    @staticmethod
    def tensor_shape_to_list(shapes):
        if isinstance(shapes, attr_value_pb2.AttrValue):
            return [dim.size for dim in shapes.shape.dim]
        elif isinstance(shapes, attr_value_pb2.tensorflow_dot_core_dot_framework_dot_tensor__shape__pb2.TensorShapeProto):
            return [dim.size for dim in shapes.dim]
        else:
            ret = []
            for shape in shapes:
                this_one = [dim.size for dim in shape.dim]
                ret.append(this_one)
            return ret

    '''
    check current source_node wether has input weights. If it has, set the weights into weight dict and remove the input edge.
    return edges' index which do not include edge connecting weights 
    '''
    def _check_weights(self, source_node, start_edge_id = 0, in_edge_count = None):
        if in_edge_count == None: in_edge_count = len(source_node.in_edges) - start_edge_id
        valid_pre_ids = []

        for pre_idx in range(start_edge_id, start_edge_id + in_edge_count):
            pre_node = self.get_parent(source_node.name, [pre_idx])
            if pre_node.type == 'Identity' and pre_node.name.split('/')[-1] == 'read':
                weight_node = self.get_parent(pre_node.name, [0])
                assert 'Variable' in weight_node.type
                self.set_weight(source_node.name, 'weights', self.ckpt_data[weight_node.name])
                source_node.feed_weights = True
            else:
                valid_pre_ids.append(pre_idx)

        return valid_pre_ids

    def _convert_padding(self, source_node, IR_node, kernel_size):
        # TODO: Fused conv and pool with padding is different from defused operators
        input_node = self.get_parent(source_node.name, [0])
        input_shape = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        if source_node.get_attr('padding') == 'VALID':
            dims = len(input_shape)
            assign_IRnode_values(IR_node, {'auto_pad' : "VALID", 'pads' : [0, 0] * dims})

        elif source_node.get_attr('padding') == 'SAME':
            padding = compute_tf_same_padding(
                input_shape,
                kernel_size,
                source_node.get_attr('strides'))
            assign_IRnode_values(IR_node, {'auto_pad' : "SAME_UPPER", 'pads' : padding})

        else:
            assert False


    def _convert_pooling(self, source_node, pool_type):
        IR_node = self._convert_identity_operation(source_node, new_op='Pool')
        kwargs = {}

        # strides
        kwargs['strides'] = source_node.get_attr('strides')

        # window_shape
        kwargs['kernel_shape'] = source_node.get_attr('ksize')

        # pool type
        kwargs['pooling_type'] = pool_type

        # padding
        self._convert_padding(source_node, IR_node, kwargs['kernel_shape'][1:-1])

        assign_IRnode_values(IR_node, kwargs)


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

        kwargs = {}
        if 'data_format' in source_node.layer.attr:
            kwargs["data_format"] = source_node.get_attr('data_format')

        if 'dtype' in source_node.layer.attr:
            assert source_node.layer.attr['dtype'].type in TensorflowParser.dtype_map, 'type [{}] is unknown.'.format(source_node.layer.attr['dtype'].type)
            IR_node.attr["dtype"].type = TensorflowParser.dtype_map[source_node.layer.attr['dtype'].type]

        if '_output_shapes' in source_node.layer.attr:
            IR_node.attr["_output_shapes"].MergeFromString(source_node.layer.attr['_output_shapes'].SerializeToString())

        if hasattr(source_node, 'feed_weights'):
            kwargs["feed_weights"] = True

        if hasattr(source_node, 'kwargs'):
            kwargs.update(source_node.kwargs)

        kwargs['scope'] = source_node.scope

        assign_IRnode_values(IR_node, kwargs)


    def _convert_inedge(self, source_node, IR_node, start_idx = 0, end_idx = None, in_ids=None):
        if end_idx == None: end_idx = len(source_node.in_edges) - start_idx
        if not in_ids:
            in_ids = range(start_idx, end_idx + start_idx)

        for idx in in_ids:
            if ':' in source_node.in_edges[idx]:
                input_tensor = self.src_graph.get_node(source_node.in_edges[idx]).real_name + ':' + source_node.in_edges[idx].split(':')[1]
            else:
                input_tensor = self.src_graph.get_node(source_node.in_edges[idx]).real_name

            IR_node.input.append(input_tensor)




    def _get_bias(self, source_node, IR_node):
        if not source_node.out_edges:
            return

        add_node = self.tf_graph.get_node(source_node.out_edges[0])
        if add_node.type != "Add" and add_node.type != "BiasAdd":
            return

        variable = self.tf_graph.get_node(add_node.in_edges[1])
        if variable.type != "Identity":
            return
        variable = self.tf_graph.get_node(variable.in_edges[0])

        assert variable.layer.attr['shape'].shape.dim[0].size == IR_node.attr['kernel_shape'].list.i[-1]

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
        if source_node.layer.attr['shape'].list.shape:
            IR_node.attr['shape'].shape.MergeFromString(source_node.layer.attr['shape'].list.shape[0].SerializeToString())
        else:
            IR_node.attr['shape'].shape.MergeFromString(source_node.layer.attr['shape'].shape.SerializeToString())


    def rename_UNKNOWN(self, source_node):
        if source_node.type in self.skip_type:
            return
        print("TensorflowEmitter has not supported operator [%s] with name [%s]."
              % (source_node.type, source_node.name))
        return


    def rename_Placeholder(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='DataInput')
        # shape
        TensorflowParser._copy_shape(source_node, IR_node)
        if len(IR_node.attr['shape'].shape.dim)>0 and len(IR_node.attr['_output_shapes'].list.shape)>0 and len(IR_node.attr['_output_shapes'].list.shape[0].dim)>0:
            IR_node.attr['shape'].shape.dim[0].size = -1
            IR_node.attr['_output_shapes'].list.shape[0].dim[0].size = -1


    def rename_Conv2D(self, source_node):
        """
        weights: name_weights, name_bias
        """
        IR_node = self._convert_identity_operation(source_node, 0, 1, 'Conv')

        kwargs = {}

        # strides
        kwargs['strides'] = source_node.get_attr('strides')

        # input[1] : W
        # filter
        W = self.tf_graph.get_node(source_node.layer.input[1])
        if W.type == 'Const':
            kwargs['kernel_shape'] = tensor_shape = self.tensor_shape_to_list(W.layer.attr['value'].tensor.tensor_shape)
        else:
            W = self.tf_graph.get_node(W.layer.input[0]).layer
            kwargs['kernel_shape'] = self.tensor_shape_to_list(W.attr['shape'])

        # padding
        self._convert_padding(source_node, IR_node, kwargs['kernel_shape'][:-2])

        if self.weight_loaded:
             self.set_weight(source_node.name, 'weights', self.ckpt_data[W.name])

        assign_IRnode_values(IR_node, kwargs)
        # output[0] : B
        self._get_bias(source_node, IR_node)


    def _convert_identity_operation(self, source_node, start_edge_id = 0, in_edge_count = None, new_op = None):
        IR_node = self.IR_graph.node.add()
        in_ids = self._check_weights(source_node, start_edge_id, in_edge_count)
        TensorflowParser._copy_and_reop(source_node, IR_node, new_op)
        self._convert_inedge(source_node, IR_node, start_edge_id, in_edge_count, in_ids)
        return IR_node


    def rename_Relu(self, source_node):
        self._convert_identity_operation(source_node)


    def rename_Softmax(self, source_node):
        self._convert_identity_operation(source_node)


    def rename_Relu6(self, source_node):
        self._convert_identity_operation(source_node)


    def rename_Add(self, source_node):
        if not source_node.covered:
            scopes = self._get_scopes(source_node.name)
            if len(scopes) < 3:
                self._convert_identity_operation(source_node,new_op='Add')

            elif scopes[-2] == 'dropout':
                # converted [dropout]
                pass

            elif scopes[-2] == 'batchnorm':
                # convert [tf.contrib.layers.batch_norm]
                self._convert_layers_batchnorm(source_node)

            else:
                # normal Add
                self._add_constant_node(source_node)
                self._convert_identity_operation(source_node,new_op='Add')

    def rename_AddV2(self, source_node):
        self.rename_Add(source_node)

    def rename_Sub(self, source_node):
        self._add_constant_node(source_node)
        self._convert_identity_operation(source_node)


    def rename_Reshape(self, source_node):
        IR_node = self._convert_identity_operation(source_node, in_edge_count = 1)
        kwargs = {'shape' : self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]}
        assign_IRnode_values(IR_node, kwargs)


    def rename_Abs(self, source_node):
        IR_node = self._convert_identity_operation(source_node, in_edge_count = 1, new_op = 'Abs')


    def rename_Square(self, source_node):
        IR_node = self._convert_identity_operation(source_node, in_edge_count = 1, new_op = 'Square')


    def rename_MatMul(self, source_node):

        W = self.tf_graph.get_node(self.tf_graph.get_node(source_node.in_edges[1]).in_edges[0])

        if 'Variable' in W.type:

            """
            weights: name_weights, name_bias
            """
            IR_node = self._convert_identity_operation(source_node, in_edge_count = 1)

            # units
            units = source_node.layer.attr['_output_shapes'].list.shape[-1].dim[-1].size
            IR_node.attr['units'].i = units

            # Weights
            W = self.tf_graph.get_node(self.tf_graph.get_node(source_node.in_edges[1]).in_edges[0])
            if self.weight_loaded:
                self.set_weight(source_node.name, 'weights', self.ckpt_data[W.name])

            if source_node.out_edges and (self.tf_graph.get_node(source_node.out_edges[0]).type == 'Add' or self.tf_graph.get_node(source_node.out_edges[0]).type == 'BiasAdd'):
                add_node = self.tf_graph.get_node(source_node.out_edges[0])
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
                # Matmul Layer
                TensorflowParser._copy_and_reop(source_node, IR_node, 'FullyConnected')
                assign_IRnode_values(IR_node, {'use_bias' : False})
        else:
            self._convert_identity_operation(source_node, new_op='MatMul')


    def rename_RealDiv(self, source_node):
        scopes = self._get_scopes(source_node.name)

        # Deal Dropout
        if len(scopes) > 1 and scopes[-2][:7] == 'dropout':
            IR_node = self._convert_identity_operation(source_node, in_edge_count = 1, new_op = 'Dropout')

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
            # print (source_node)
            # print (source_node.layer)
            # assert False
            self._convert_identity_operation(source_node, new_op='Div')


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

    def rename_QueueDequeueUpToV2(self, source_node):
        IR_node = self._convert_identity_operation(source_node, in_edge_count = 0, new_op = 'DataInput')
        IR_node.attr['shape'].shape.MergeFromString(source_node.layer.attr['_output_shapes'].list.shape[0].SerializeToString())
        IR_node.attr['shape'].shape.dim[0].size = -1
        IR_node.attr['dtype'].type = self.dtype_map[source_node.layer.attr['component_types'].list.type[0]]



    def rename_QueueDequeueManyV2(self, source_node):
        IR_node = self._convert_identity_operation(source_node, in_edge_count = 0, new_op = 'DataInput')
        IR_node.attr['shape'].shape.MergeFromString(source_node.layer.attr['_output_shapes'].list.shape[0].SerializeToString())
        IR_node.attr['shape'].shape.dim[0].size = -1
        IR_node.attr['dtype'].type = self.dtype_map[source_node.layer.attr['component_types'].list.type[0]]

    # def rename_RandomShuffleQueueV2(self, source_node):
    #     # print(source_node.layer)
    #     IR_node = self._convert_identity_operation(source_node, in_edge_count = 0, new_op = 'DataInput')
    #     # IR_node.attr['shape'].shape.MergeFromString(source_node.layer.attr['_output_shapes'].list.shape[0].SerializeToString())
    #     # IR_node.attr['shape'].shape.dim[0].size = -1
    #     IR_node.attr['dtype'].type = self.dtype_map[source_node.layer.attr['component_types'].list.type[0]]


    def rename_Pad(self, source_node):
        IR_node = self._convert_identity_operation(source_node, in_edge_count = 1, new_op = 'Pad')
        kwargs = {}
        kwargs['mode'] = 'constant'
        kwargs['constant_values'] = 0.0

        # paddings
        padding = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor
        shapes = tensor_util.MakeNdarray(padding)
        kwargs['pads'] = convert_tf_pad_to_onnx(shapes)

        assign_IRnode_values(IR_node, kwargs)


    def rename_Mean(self, source_node):
        self._convert_reduction_operators(source_node, new_op = 'ReduceMean')


    def rename_ConcatV2(self, source_node):
        n = len(source_node.in_edges) - 1
        self._add_constant_node(source_node)
        IR_node = self._convert_identity_operation(source_node, in_edge_count = n, new_op = 'Concat')
        axis = self.tf_graph.get_parent(source_node.name, [n])
        IR_node.attr['axis'].i = axis.layer.attr['value'].tensor.int_val[0]


    def rename_DepthwiseConv2dNative(self, source_node):
        IR_node = self._convert_identity_operation(source_node, in_edge_count=1, new_op='DepthwiseConv')
        kwargs = {}
        kwargs['strides'] = source_node.get_attr('strides')

        input_node = self.src_graph.get_parent(source_node.name, [1])
        kwargs['kernel_shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        self._convert_padding(source_node, IR_node, kwargs['kernel_shape'][:-2])

        if self.weight_loaded:
            weight = self.src_graph.get_parent(source_node.name, [1, 0])
            self.set_weight(source_node.name, 'weights', self.ckpt_data[weight.name])

        assign_IRnode_values(IR_node, kwargs)
        self._get_bias(source_node, IR_node)


    def rename_FusedBatchNorm(self, source_node):
        IR_node = self._convert_identity_operation(source_node, in_edge_count=1, new_op='BatchNorm')
        IR_node.attr['epsilon'].f = source_node.get_attr('epsilon', 0)

        # gamma (scale)
        scale = self.get_parent(source_node.name, [1], True)

        if scale.type == 'Const':
            value = scale.get_attr('value')
            shape = value.tensor_shape
            assert len(shape.dim) == 1
            shape = shape.dim[0].size

            assert len(value.float_val) == 1
            value = value.float_val[0]

            if np.isclose(value, 1.0):
                IR_node.attr['scale'].b = False
            else:
                IR_node.attr['scale'].b = True
                if self.weight_loaded:
                    self.set_weight(source_node.name, 'scale', np.array([value] * shape))

        else:
            scale = self.get_parent(scale.name, [0], True)
            if self.weight_loaded:
                self.set_weight(source_node.name, 'scale', self.ckpt_data[scale.name])
            IR_node.attr['scale'].b = True

        # bias
        bias = self.get_parent(source_node.name, [2, 0], True)
        IR_node.attr['bias'].b = True

        # Mean
        mean = self.get_parent(source_node.name, [3, 0], True)

        # Var
        var = self.get_parent(source_node.name, [4, 0], True)

        if self.weight_loaded:
            self.set_weight(source_node.name, 'bias', self.ckpt_data[bias.name])
            self.set_weight(source_node.name, 'mean', self.ckpt_data[mean.name])
            self.set_weight(source_node.name, 'var', self.ckpt_data[var.name])

    def rename_FusedBatchNormV3(self, source_node):
        self.rename_FusedBatchNorm(source_node)

    def rename_Shape(self, source_node):
        IR_node = self._convert_identity_operation(source_node, in_edge_count=1, new_op='Shape')

    def rename_Pack(self, source_node):
        N = len(source_node.layer.input)
        for i in range(N):
            this_node = self.get_parent(source_node.name, [i])
            if this_node.type == 'Const':

                IR_node = self.IR_graph.node.add()
                TensorflowParser._copy_and_reop(this_node, IR_node, 'Constant')
                kwargs = {
                    'value' : this_node.layer.attr['value'].tensor.int_val[0],
                }
                assign_IRnode_values(IR_node, kwargs)

        IR_node = self._convert_identity_operation(source_node, new_op='Pack')
        kwargs = {
            'axis' : source_node.layer.attr['axis'].i,
            'N'    : source_node.layer.attr['N'].i
        }
        assign_IRnode_values(IR_node, kwargs)

    def rename_Gather(self, source_node):

        W = self.src_graph.get_parent(source_node.name, [0])
        W = self.src_graph.get_parent(W.name, [0])

        if 'Variable' in W.type:
            IR_node = self._convert_identity_operation(source_node, new_op='Embedding')

            self.set_weight(source_node.name, "weights", self.ckpt_data[W.name])

            kwargs = {
                'input_dim' : self.ckpt_data[W.name].shape[0],
                'output_dim' : self.ckpt_data[W.name].shape[1],
                'mask_zero' : False
            }
            kwargs['axis'] = 0  # add default
            assign_IRnode_values(IR_node, kwargs)
        else:
            IR_node = self._convert_identity_operation(source_node, new_op='Gather')

        return IR_node

    def rename_GatherV2(self, source_node):
        
        IR_node = self.rename_Gather(source_node)

        kwargs = {}
        kwargs['axis'] = source_node.layer.attr['axis'].i
        assign_IRnode_values(IR_node, kwargs)


    def rename_Transpose(self, source_node):
        IR_node = self._convert_identity_operation(source_node)



    def rename_Sigmoid(self, source_node):
        self._convert_identity_operation(source_node)


    def rename_Mul(self, source_node):
        scale1 = self.get_parent(source_node.name, [1], True)
        scale2 = self.get_parent(source_node.name, [0], True)

        if scale1.type == 'Const' or scale2.type == 'Const':
            self._add_constant_node(source_node)
            self._convert_identity_operation(source_node)

        elif scale2.type == 'Identity':
            scale2 = self.get_parent(scale2.name, [0], True)
            assert scale2.type == "VariableV2"
            self.set_weight(source_node.name, 'alpha', self.ckpt_data[scale2.name])
            self._convert_identity_operation(source_node)

        else:
            self._convert_identity_operation(source_node)


    '''
    tf.unpack has been deprecated with replaced tf.unstack
    '''
    def rename_Unpack(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='Unstack')
        kwargs = {
            'axis' : source_node.get_attr('axis'),
            'num'  : source_node.get_attr('num')
        }
        assign_IRnode_values(IR_node, kwargs)


    def rename_Split(self, source_node):
        if source_node.get_attr('num_split') == 1:            
            for n in source_node.out_nodes:
                for idx, e in enumerate(n.in_edges):
                    if source_node.name in e:
                        n.in_edges[idx] = e.split(':')[0]

            source_node.real_name = self.get_parent(source_node.name, [1]).real_name

        else:
            IR_node = self._convert_identity_operation(source_node, 1, 1)
            kwargs = {
                'axis' : self.get_parent(source_node.name, [0]).layer.attr['value'].tensor.int_val[0],
                'split' : source_node.get_attr('num_split')
            }
            assign_IRnode_values(IR_node, kwargs)


    def rename_StridedSlice(self, source_node):
        # TODO: Current it is only for slice

        if self.get_parent(source_node.name, [1]).type != 'Const':
            self._add_constant_node(source_node)
            IR_node = self._convert_identity_operation(source_node, new_op='Slice')
            return

        IR_node = self._convert_identity_operation(source_node, in_edge_count=1, new_op='Slice')
        kwargs = {
            'begin_mask' : source_node.get_attr('begin_mask'),
            'end_mask'   : source_node.get_attr('end_mask'),
            'shrink_axis_mask': source_node.get_attr('shrink_axis_mask'),
            'new_axis_mask' :source_node.get_attr('new_axis_mask')
        }

        starts = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor
        starts = tensor_util.MakeNdarray(starts).tolist()
        kwargs['starts'] = starts

        ends = self.get_parent(source_node.name, [2]).layer.attr['value'].tensor
        ends = tensor_util.MakeNdarray(ends).tolist()
        kwargs['ends'] = ends

        if self.get_parent(source_node.name, [3]) != None:
            strides = self.get_parent(source_node.name, [3]).layer.attr['value'].tensor
            strides = tensor_util.MakeNdarray(strides).tolist()
            kwargs['strides'] = strides

        assign_IRnode_values(IR_node, kwargs)


    def rename_Slice(self, source_node):
        input_node_begin = self.get_parent(source_node.name, [1])
        input_node_size = self.get_parent(source_node.name, [2])

        begin = tensor_util.MakeNdarray(input_node_begin.layer.attr['value'].tensor)
        size = tensor_util.MakeNdarray(input_node_size.layer.attr['value'].tensor)

        IR_node = self._convert_identity_operation(source_node, in_edge_count=1, new_op='Slice')

        # TODO:  axis
        end = size + begin
        kwargs = {
            'starts' : begin,
            'ends' : end
        }

        assign_IRnode_values(IR_node, kwargs)


    def rename_LRN(self, source_node):
        IR_node = self._convert_identity_operation(source_node)
        size = source_node.get_attr('depth_radius') * 2 + 1
        alpha = source_node.get_attr('alpha') * size
        beta = source_node.get_attr('beta')
        bias = source_node.get_attr('bias')

        kwargs = {
            "alpha" : alpha,
            "beta" : beta,
            "bias" : bias,
            'size' : size,
        }
        assign_IRnode_values(IR_node, kwargs)


    def rename_Tanh(self, source_node):
        self._convert_identity_operation(source_node)


    def rename_ExpandDims(self, source_node):
        IR_node = self._convert_identity_operation(source_node, 0, 1, new_op='Unsqueeze')
        
        ax_node = self.get_parent(source_node.name, [1])
        kwargs = {
            'axes': [ax_node.layer.attr['value'].tensor.int_val[0]]
        }
        assign_IRnode_values(IR_node, kwargs)


    def rename_Fill(self, source_node):
        IR_node = self._convert_identity_operation(source_node, 0, 1, new_op='Fill')

        value_node = self.get_parent(source_node.name, [1])
        if value_node.layer.attr['value'].tensor.float_val:
            IR_node.attr['value'].f = value_node.layer.attr['value'].tensor.float_val[0]
        elif value_node.layer.attr['value'].tensor.int_val:
            IR_node.attr['value'].i = value_node.layer.attr['value'].tensor.int_val[0]
        else:
            raise NotImplementedError()


    def rename_Conv2DBackpropInput(self, source_node):
        """
        weights: name_weights, name_bias
        """
        IR_node = self._convert_identity_operation(source_node, new_op = 'ConvTranspose')

        kwargs = {}

        # strides
        kwargs['strides'] = source_node.get_attr('strides')

        # input[1] : W
        # filter
        W = self.tf_graph.get_node(source_node.layer.input[1])
        W = self.tf_graph.get_node(W.layer.input[0]).layer
        kwargs['kernel_shape'] = self.tensor_shape_to_list(W.attr['shape'])

        # padding
        self._convert_padding(source_node, IR_node, kwargs['kernel_shape'][:-2])

        if self.weight_loaded:
            self.set_weight(source_node.name, 'weights', self.ckpt_data[W.name])

        assign_IRnode_values(IR_node, kwargs)
        # output[0] : B
        self._get_bias(source_node, IR_node)
    
    def rename_Minimum(self, source_node):
        self._add_constant_node(source_node)
        self._convert_identity_operation(source_node)

    def rename_Maximum(self, source_node):
        self._add_constant_node(source_node)
        self._convert_identity_operation(source_node)

    def rename_Cast(self, source_node):
        IR_node = self._convert_identity_operation(source_node)
        dst = source_node.get_attr('DstT')
        if dst == 1:
            dst = 'float'
        elif dst == 3:
            dst = 'int'
        else:
            raise NotImplementedError

        kwargs = {'dstType' : dst}
        assign_IRnode_values(IR_node, kwargs)
