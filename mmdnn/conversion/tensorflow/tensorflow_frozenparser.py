import numpy as np
import tensorflow
from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import attr_value_pb2
from mmdnn.conversion.tensorflow.tensorflow_graph import TensorflowGraph
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.utils import *
from mmdnn.conversion.common.DataStructure.parser import Parser

class TensorflowParser2(Parser):

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

    q_type = set([
        "Dequantize",
        "QuantizeV2",
        "QuantizedConv2D",
        "QuantizedReshape",
        "RequantizationRange"
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

    def __init__(self, frozen_file, inputshape, dest_nodes):
        super(TensorflowParser2, self).__init__()

        self.weight_loaded = True
        # load model files into TensorFlow graph
        with open(frozen_file, 'rb') as f:
            serialized = f.read()
        tensorflow.reset_default_graph()
        original_gdef = tensorflow.GraphDef()

        original_gdef.ParseFromString(serialized)
        model = original_gdef

        if dest_nodes != None:
            from tensorflow.python.framework.graph_util import extract_sub_graph
            model = extract_sub_graph(model, dest_nodes.split(','))

        output_shape_map = dict()
        input_shape_map = dict()
        with tensorflow.Graph().as_default() as g:
            tensorflow.import_graph_def(original_gdef, name='')
            ops = g.get_operations()
            N = len(ops)
            p = 0
            for i in range(N):
                for x in ops[i].inputs:
                    input_shape_map[x.name] = x.get_shape()
                for x in ops[i].outputs:
                    output_shape_map[x.name] = x.get_shape()
        tensor_input = tensorflow.TensorShape([None, tensorflow.Dimension(inputshape[0]), tensorflow.Dimension(inputshape[1]), tensorflow.Dimension(inputshape[2])])
        output_shape_map['input:0'] = tensor_input
        self.tf_graph = TensorflowGraph(model)
        for node in self.tf_graph.model.node:
            if node.name == 'input' and node.op == 'Placeholder':
                node.attr['shape'].list.shape.extend([output_shape_map[node.name + ':0'].as_proto()])

            if (node.name + ':0') in output_shape_map:
                node.attr['_output_shapes'].list.shape.extend([output_shape_map[node.name + ':0'].as_proto()])

            if  node.op == 'MirrorPad':
                node.attr['paddings'].list.shape.extend([input_shape_map[node.name + '/paddings:0'].as_proto()])

            if  node.op == 'QuantizeV2':
                node.attr['shape'].list.shape.extend([input_shape_map[node.name + ':0'].as_proto()])

            if  node.op == 'RequantizationRange':
                map_key = node.name.split('eightbit')[0] + "eightbit_quantized_conv:0"
                node.attr['shape'].list.shape.extend([input_shape_map[map_key].as_proto()])

            if  node.op == 'Requantize':
                map_key = node.name.replace("requantize", "quantized_conv")+":0"
                node.attr['shape'].list.shape.extend([input_shape_map[map_key].as_proto()])
        self.tf_graph.build()

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


    def _convert_layers_batchnorm(self, source_node):
        # name, op
        IR_node = self.IR_graph.node.add()
        TensorflowParser2._copy_and_reop(source_node, IR_node, 'BatchNorm')

        # epsilon
        epsilon = self.get_parent(source_node.name, [1])
        IR_node.attr['epsilon'].f = epsilon.layer.attr['value'].tensor.float_val[0]

        # moving variance (var) /read
        moving_variance = self.get_parent(source_node.name, [0])

        if moving_variance.type == 'Identity':
            moving_variance_read = self.src_graph.get_parent(moving_variance.name, [0])
            tensor_content = moving_variance_read.get_attr('value')
            moving_variance_content = tensor_util.MakeNdarray(tensor_content)
            self.set_weight(source_node.name, 'var', moving_variance_content)

        else:
            print(moving_variance)
            assert False

        # gamma (scale)
        Rsqrt = self.get_son(source_node.name, [0], True)
        if len(Rsqrt.out_edges) == 2:
            IR_node.attr['scale'].b = False
            output_node = self.get_son(source_node.name, [0, 0, 0], True)
        else:
            IR_node.attr['scale'].b = True
            assert False
            # self.set_weight(source_node.name, 'scale', )
            output_node = self.get_son(source_node.name, [0, 0, 0, 0], True)

        # beta  (bias)
        Sub = self.get_son(source_node.name, [0, 1, 0], True)
        beta = self.get_parent(Sub.name, [0, 0]).get_attr('value')
        bias = tensor_util.MakeNdarray(beta)  #(96,)
        IR_node.attr['bias'].b = True
        self.set_weight(source_node.name, 'bias', bias)

        # moving mean (mean)
        son2 = self.get_son(source_node.name, [0, 1], True)
        moving_mean = self.get_parent(son2.name, [0, 0]).get_attr('value')
        mean = tensor_util.MakeNdarray(moving_mean)
        self.set_weight(source_node.name, 'mean', mean)

        # input node
        assert output_node.type == 'Add'
        input_node = self.get_parent(output_node.name, [0, 0])
        IR_node.input.append(input_node.real_name)

        # output node
        output_node.real_name = source_node.name

    @classmethod
    def _skip_node(cls, source_node):
        if source_node.covered:
            return True

        for prefix in cls.skip_prefix:
            if source_node.name.startswith(prefix):
                return True

        scopes = TensorflowParser2._get_scopes(source_node.name)

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
            if hasattr(self, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                func(current_node)

            else:
                self.rename_UNKNOWN(current_node)

    @staticmethod
    def tensor_shape_to_list(shapes):
        if isinstance(shapes, attr_value_pb2.AttrValue):
            return [dim.size for dim in shapes.shape.dim]

        else:
            ret = []
            for shape in shapes:
                this_one = [dim.size for dim in shape.dim]
                ret.append(this_one)
            return ret

    @staticmethod
    def _copy_and_reop(source_node, IR_node, new_op = None):
        if new_op == None: new_op = source_node.type
        IR_node.name = source_node.name
        IR_node.op = new_op

        kwargs = {}
        if 'data_format' in source_node.layer.attr:
            kwargs['data_format'] = source_node.get_attr('data_format')

        if 'T' in source_node.layer.attr:
            if source_node.type not in TensorflowParser2.q_type:
                assert source_node.layer.attr['T'].type in TensorflowParser2.dtype_map, 'type [{}] is unknown.'.format(source_node.layer.attr['dtype'].type)
                IR_node.attr["dtype"].type = TensorflowParser2.dtype_map[source_node.layer.attr['T'].type]
            else:
                 IR_node.attr["dtype"].type = TensorflowParser2.dtype_map[6]

        if '_output_shapes' in source_node.layer.attr:
            IR_node.attr["_output_shapes"].MergeFromString(source_node.layer.attr['_output_shapes'].SerializeToString())

        if 'paddings' in source_node.layer.attr:
            IR_node.attr["paddings"].MergeFromString(source_node.layer.attr['paddings'].SerializeToString())

        assign_IRnode_values(IR_node, kwargs)

    def _convert_inedge(self, source_node, IR_node, start_idx = 0, end_idx = None):
        if end_idx == None: end_idx = len(source_node.in_edges)
        for idx in range(start_idx, end_idx):
            IR_node.input.append(self.src_graph.get_node(source_node.in_edges[idx]).real_name)

    @staticmethod
    def _copy_shape(source_node, IR_node):
        assert 'shape' in source_node.layer.attr

        IR_node.attr['shape'].shape.MergeFromString(source_node.layer.attr['shape'].list.shape[0].SerializeToString())

    def rename_UNKNOWN(self, source_node):
        if source_node.type in self.skip_type:
            return
        print("Tensorflow has not supported operator [%s] with name [%s]."
              % (source_node.type, source_node.name))
        assert False

    def rename_NoOp(self, source_node):
        return

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
            assign_IRnode_values(IR_node, {'auto_pad' : "SAME_LOWER", 'pads' : padding})

        else:
            assert False


    def _get_bias(self, source_node, IR_node):
        if not source_node.out_edges:
            return

        add_node = self.tf_graph.get_node(source_node.out_edges[0])
        if add_node.type != "Add" and add_node.type != "BiasAdd":
            return

        variable = self.tf_graph.get_node(add_node.in_edges[1]) #add_bias node
        if variable.type == 'Identity':
            variable = self.tf_graph.get_node(variable.in_edges[0])

        bias_value = variable.get_attr('value')
        bias = tensor_util.MakeNdarray(bias_value)

        # assert variable.get_attr('_output_shapes')[0].dim[0].size == IR_node.attr['kernel_shape'].list.i[-1]


        add_node.real_name = IR_node.name
        add_node.covered = True
        IR_node.attr['use_bias'].b = True
        current_layer = self.weights[source_node.name]
        current_layer['bias'] = bias


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



    def _convert_identity_operation(self, source_node, start_idx = 0, end_idx = None, new_op = None):
        IR_node = self.IR_graph.node.add()
        TensorflowParser2._copy_and_reop(source_node, IR_node, new_op)
        self._convert_inedge(source_node, IR_node, start_idx, end_idx)
        return IR_node


    def rename_Placeholder(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='DataInput')
        TensorflowParser2._copy_shape(source_node, IR_node)


    def rename_Reshape(self, source_node):
        IR_node = self._convert_identity_operation(source_node, end_idx = 1)
        kwargs = {'shape' : self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]}
        assign_IRnode_values(IR_node, kwargs)

    def rename_MirrorPad(self, source_node):
        IR_node = self._convert_identity_operation(source_node, 'MirrorPad')
        kwargs = {}
        kwargs['mode'] = source_node.get_attr('mode')

        assign_IRnode_values(IR_node, kwargs)

    def rename_Min(self, source_node):
        IR_node = self._convert_identity_operation(source_node, 'Min')
        kwargs = {}
        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs['shape_0'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        input_node = self.src_graph.get_parent(source_node.name, [1])
        kwargs['shape_1'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
        assign_IRnode_values(IR_node, kwargs)

    def rename_Max(self, source_node):
        IR_node = self._convert_identity_operation(source_node, 'Max')
        kwargs = {}
        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs['shape_0'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        input_node = self.src_graph.get_parent(source_node.name, [1])
        kwargs['shape_1'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
        assign_IRnode_values(IR_node, kwargs)

    def rename_Mul(self, source_node):
        input_node_0 = self.src_graph.get_parent(source_node.name, [0])

        # mean/read
        if input_node_0.type == 'Identity':
            input_node_0_read = self.src_graph.get_parent(input_node_0.name, [0])
            tensor_content = input_node_0_read.get_attr('value')
            tensor_content = tensor_util.MakeNdarray(tensor_content)
            self.set_weight(source_node.name, 'weights', tensor_content)
            IR_node = self._convert_identity_operation(source_node, start_idx = 1)

        else:
            IR_node = self._convert_identity_operation(source_node)

    def rename_Add(self, source_node):
        scopes = self._get_scopes(source_node.name)
        if scopes[-2] == 'batchnorm':
            self._convert_layers_batchnorm(source_node)

    def rename_Sub(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = "Sub")

        kwargs = {}
        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        assign_IRnode_values(IR_node, kwargs)

    def rename_Sum(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'Sum')
        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs = {}
        kwargs['cal_shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        input_node_indices = self.src_graph.get_parent(source_node.name, [1])
        indice_value = input_node_indices.get_attr('value')
        if indice_value.tensor_content:
            shapes = tensor_util.MakeNdarray(indice_value)
            c = shapes.tolist()
            kwargs['sum_indices'] = c
        else:
            kwargs['sum_indices'] = input_node_indices.get_attr('value').int_val[0]
        assign_IRnode_values(IR_node, kwargs)

    def rename_Rsqrt(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = "Rsqrt")

        kwargs = {}
        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        assign_IRnode_values(IR_node, kwargs)

    def rename_Square(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'Square')
        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs = {}
        kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        assign_IRnode_values(IR_node, kwargs)

    def rename_Sigmoid(self, source_node):
        IR_node = self._convert_identity_operation(source_node)

        kwargs = {}
        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        assign_IRnode_values(IR_node, kwargs)

    def rename_Reciprocal(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'Reciprocal')


    def rename_Cast(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'Cast')
        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs = {}
        kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        assign_IRnode_values(IR_node, kwargs)


    def rename_Prod(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'Prod')

        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs = {}
        kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        input_node_const = self.src_graph.get_parent(source_node.name, [1])

        kwargs['const'] = input_node_const.get_attr('value').int_val[0]
        assign_IRnode_values(IR_node, kwargs)


    def rename_Shape(self, source_node):

        IR_node = self._convert_identity_operation(source_node, new_op = 'Shape')
        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs = {}
        kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        assign_IRnode_values(IR_node, kwargs)


    def rename_Squeeze(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'Squeeze')


    def rename_Pack(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'Pack')


    def rename_Gather(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'Gather')
        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs = {}
        kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        input_node_indices = self.src_graph.get_parent(source_node.name, [1])
        indice_value = input_node_indices.get_attr('value')
        shapes = tensor_util.MakeNdarray(indice_value)
        c = shapes.tolist()
        kwargs['gather_indices'] = c

        assign_IRnode_values(IR_node, kwargs)


    def rename_StridedSlice(self, source_node):

        IR_node = self._convert_identity_operation(source_node, new_op = 'StridedSlice')

        kwargs = {}
        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        input_node_const0 = self.src_graph.get_parent(source_node.name, [1])
        input_node_const1 = self.src_graph.get_parent(source_node.name, [2])
        input_node_const2 = self.src_graph.get_parent(source_node.name, [3])

        if input_node_const0.get_attr('value').int_val:
            kwargs['const0'] = input_node_const0.get_attr('value').int_val[0]
        if input_node_const1.get_attr('value').int_val:
            kwargs['const1'] = input_node_const1.get_attr('value').int_val[0]
        if input_node_const2.get_attr('value').int_val:
            kwargs['const2'] = input_node_const2.get_attr('value').int_val[0]
        assign_IRnode_values(IR_node, kwargs)


    def rename_ExpandDims(self, source_node):

        IR_node = self._convert_identity_operation(source_node, new_op = 'ExpandDims')
        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs = {}
        kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        input_node_indices = self.src_graph.get_parent(source_node.name, [1])

        kwargs['exp_dim'] = input_node_indices.get_attr('value').int_val[0]
        assign_IRnode_values(IR_node, kwargs)


    def rename_ResizeNearestNeighbor(self, source_node):
        IR_node = self._convert_identity_operation(source_node)
        kwargs = {}
        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        input_node_size = self.src_graph.get_parent(source_node.name, [1])
        kwargs['size'] = self.tensor_shape_to_list(input_node_size.get_attr('_output_shapes'))[0]

        assign_IRnode_values(IR_node, kwargs)



    def rename_Conv2D(self, source_node):
        IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op = 'Conv')
        kwargs = {}
        kwargs['strides'] = source_node.get_attr('strides')
        kwargs['padding'] = source_node.get_attr('padding')

        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        # weights
        input_node_weight = self.src_graph.get_parent(source_node.name, [1])
        if input_node_weight.type == 'Const':
            tensor_content = input_node_weight.get_attr('value')
        else:
            input_node_weight_read = self.src_graph.get_parent(input_node_weight.name, [0])
            tensor_content = input_node_weight_read.get_attr('value')
        W = tensor_util.MakeNdarray(tensor_content)

        kwargs['kernel_shape'] = self.tensor_shape_to_list(input_node_weight.get_attr('_output_shapes'))[0]

        self.set_weight(source_node.name, 'weights', W)

        self._convert_padding(source_node, IR_node, kwargs['kernel_shape'][:-2])

        assign_IRnode_values(IR_node, kwargs)
        self._get_bias(source_node, IR_node)


    def rename_Relu(self, source_node):
        IR_node = self._convert_identity_operation(source_node)
        kwargs = {'shape' : self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]}
        assign_IRnode_values(IR_node, kwargs)


    def rename_MaxPool(self, source_node):
        # print(source_node.layer.attr)
        self._convert_pooling(source_node, b'MAX')


    def rename_AvgPool(self, source_node):
        self._convert_pooling(source_node, b'AVG')


    def rename_LRN(self, source_node):
        IR_node = self._convert_identity_operation(source_node)

        # alpha
        IR_node.attr["alpha"].f = float(source_node.get_attr("alpha", "0.0001"))
        # beta
        IR_node.attr["beta"].f = float(source_node.get_attr("beta", "0.75"))
        IR_node.attr["size"].i = source_node.get_attr("depth_radius")
        IR_node.attr["bias"].f = float(source_node.get_attr("bias"))


    def rename_Concat(self, source_node):
        n = len(source_node.in_edges)
        IR_node = self._convert_identity_operation(source_node, start_idx=1, end_idx=n, new_op='Concat')
        axis = self.tf_graph.get_parent(source_node.name, [0])
        IR_node.attr["axis"].i = axis.get_attr('value').int_val[0]


    def rename_ConcatV2(self, source_node):
        n = len(source_node.in_edges)
        IR_node = self._convert_identity_operation(source_node, start_idx=0, end_idx=n-1, new_op='Concat')
        axis = self.tf_graph.get_parent(source_node.name, [n-1])
        IR_node.attr["axis"].i = axis.get_attr('value').int_val[0]


    def rename_MatMul(self, source_node):
        IR_node = self._convert_identity_operation(source_node, end_idx = 1)

        input_weight_node = self.src_graph.get_parent(source_node.name, [1])
        weight_value = input_weight_node.get_attr('value')
        weight = tensor_util.MakeNdarray(weight_value)
        self.set_weight(source_node.name, 'weights', weight)

        units = source_node.layer.attr['_output_shapes'].list.shape[-1].dim[-1].size
        IR_node.attr['units'].i = units

        if source_node.out_edges and self.tf_graph.get_node(source_node.out_edges[0]).type == 'BiasAdd':
            add_node = self.tf_graph.get_node(source_node.out_edges[0])
            add_node.covered = True
            add_node.real_name = source_node.real_name

            TensorflowParser2._copy_and_reop(source_node, IR_node, 'FullyConnected')
            variable = self.tf_graph.get_node(add_node.in_edges[1]) #add_bias node
            bias_value = variable.get_attr('value')
            bias = tensor_util.MakeNdarray(bias_value)
            self.set_weight(source_node.name, 'bias', bias)
            IR_node.attr['use_bias'].b = True


    def rename_Softmax(self, source_node):
        IR_node = self._convert_identity_operation(source_node)
        kwargs = {'shape' : self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]}
        assign_IRnode_values(IR_node, kwargs)


    def rename_Identity(self, source_node):
        source_node.real_name =  self.src_graph.get_node(source_node.in_edges[0]).real_name


    def rename_QuantizeV2(self, source_node):
        IR_node = self._convert_identity_operation(source_node, 'QuantizeV2')
        TensorflowParser2._copy_shape(source_node, IR_node)

    def rename_QuantizedRelu(self, source_node):
        IR_node = self._convert_identity_operation(source_node, "QuantizedRelu")
        kwargs = {'shape' : self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]}
        assign_IRnode_values(IR_node, kwargs)

    def rename_QuantizedReshape(self, source_node):
        IR_node = self._convert_identity_operation(source_node)
        kwargs = {'shape' : self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]}
        assign_IRnode_values(IR_node, kwargs)

    def rename_QuantizedConv2D(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'QConv')
        kwargs = {}
        kwargs['strides'] = source_node.get_attr('strides')
        kwargs['padding'] = source_node.get_attr('padding')


        # weights
        input_node = self.src_graph.get_parent(source_node.name, [1])
        tensor_content = input_node.get_attr('value')
        W = tensor_util.MakeNdarray(tensor_content)
        W = W.astype(np.uint8)

        kwargs['kernel_shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]


        input_node_minw = self.src_graph.get_parent(source_node.name, [4])
        min_W = input_node_minw.get_attr('value').float_val[0]

        input_node_maxw = self.src_graph.get_parent(source_node.name, [5])
        max_W = input_node_maxw.get_attr('value').float_val[0]

        if source_node.get_attr('Tfilter') == tensorflow.quint8:
            W = ((max_W - min_W)/255.0) * W + min_W

        else:
            assert False, ('Only uint8 weights handled currently by the converter')

        self.set_weight(source_node.name, 'kernel_weights', W)

        assign_IRnode_values(IR_node, kwargs)


    def rename_Dequantize(self, source_node):
        IR_node = self._convert_identity_operation(source_node)
        kwargs = {}
        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        assign_IRnode_values(IR_node, kwargs)

    def rename_Requantize(self, source_node):
        IR_node = self._convert_identity_operation(source_node, 'Requantize')
        TensorflowParser2._copy_shape(source_node, IR_node)

    def rename_RequantizationRange(self, source_node):
        IR_node = self._convert_identity_operation(source_node, 'RequantizationRange')
        TensorflowParser2._copy_shape(source_node, IR_node)