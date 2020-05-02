import numpy as np
import tensorflow
from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import attr_value_pb2
from mmdnn.conversion.tensorflow.tensorflow_graph import TensorflowGraph
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.utils import *
from mmdnn.conversion.common.DataStructure.parser import Parser
from distutils.version import LooseVersion
import tempfile
import os
import shutil


class TensorflowParser2(Parser):

    # skip_prefix = [
    #     "^",
    #     "train_op",
    #     "save",
    #     "gradients",
    #     "init",
    #     "global_step",
    #     "distort_image",
    #     "Adagrad",
    # ]
    skip_prefix = [
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
        "FIFOQueueV2",
        "Assert",
        "Unpack",
        "NextIteration",
        "TensorArrayV3",
        "Range",
        "TensorArrayScatterV3",
        "TensorArrayReadV3",
        "TensorArrayWriteV3",
        # "Switch"
        "Dequantize",
        # "RequantizationRange",
        # "Requantize",
        "ExpandDims",
        # "Identity",
        # "Mean",
        # "Cast"
        "Pack",
        "CheckNumerics",
        "Where"
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
        9 : graph_pb2.DT_INT64,
        10: graph_pb2.DT_BOOL
    }

    tf_dtype_map = {
        0: tensorflow.float32,  # set default to be float
        1: tensorflow.float32,
        2: tensorflow.float64,
        3: tensorflow.int32,
        4: tensorflow.uint8,
        5: tensorflow.int16,
        6: tensorflow.int8,
        7: tensorflow.string,
        9: tensorflow.int64,
        10: tensorflow.bool
    }


    @property
    def src_graph(self):
        return self.tf_graph

    def __init__(self, frozen_file, inputshape, in_nodes, dest_nodes):
        if LooseVersion(tensorflow.__version__) < LooseVersion('1.8.0'):
            raise ImportError(
                'Your TensorFlow version %s is outdated. '
                'MMdnn requires tensorflow>=1.8.0' % tensorflow.__version__)

        super(TensorflowParser2, self).__init__()

        self.weight_loaded = True
        # load model files into TensorFlow graph
        with open(frozen_file, 'rb') as f:
            serialized = f.read()
        tensorflow.reset_default_graph()
        original_gdef = tensorflow.GraphDef()

        original_gdef.ParseFromString(serialized)

        in_type_list = {}
        for n in original_gdef.node:
            if n.name in in_nodes:
                in_type_list[n.name] = n.attr['dtype'].type

        from tensorflow.python.tools import strip_unused_lib
        from tensorflow.python.framework import dtypes
        from tensorflow.python.platform import gfile
        original_gdef = strip_unused_lib.strip_unused(
                input_graph_def = original_gdef,
                input_node_names = in_nodes,
                output_node_names = dest_nodes,
                placeholder_type_enum = dtypes.float32.as_datatype_enum)
        # Save it to an output file
        tempdir = tempfile.mkdtemp()
        frozen_model_file = os.path.join(tempdir, 'frozen.pb')
        with gfile.GFile(frozen_model_file, "wb") as f:
            f.write(original_gdef.SerializeToString())
        with open(frozen_model_file, 'rb') as f:
            serialized = f.read()
        shutil.rmtree(tempdir)

        tensorflow.reset_default_graph()
        model = tensorflow.GraphDef()
        model.ParseFromString(serialized)

        output_shape_map = dict()
        input_shape_map = dict()
        dtype = tensorflow.float32

        with tensorflow.Graph().as_default() as g:
            input_map = {}
            for i in range(len(inputshape)):
                dtype = TensorflowParser2.tf_dtype_map[in_type_list[in_nodes[i]]]
                if in_type_list[in_nodes[i]] in (0, 1, 2):
                    x = tensorflow.placeholder(dtype, shape=[None] + inputshape[i])
                elif in_type_list[in_nodes[i]] in (3, 4, 5, 6, 7):
                    x = tensorflow.placeholder(dtype, shape=inputshape[i])
                elif in_type_list[in_nodes[i]] == 10:
                    x = tensorflow.placeholder(dtype)
                else:
                    raise NotImplementedError

                input_map[in_nodes[i] + ':0'] = x

            tensorflow.import_graph_def(model, name='', input_map=input_map)

        with tensorflow.Session(graph = g) as sess:

            tempdir = tempfile.mkdtemp()
            meta_graph_def = tensorflow.train.export_meta_graph(filename=os.path.join(tempdir, 'my-model.meta'))
            model = meta_graph_def.graph_def
            shutil.rmtree((tempdir))

        self.tf_graph = TensorflowGraph(model)
        self.tf_graph.build()


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


    def _convert_reduction_operators(self, source_node, new_op = None):
        IR_node = self._convert_identity_operation(source_node, 1, new_op)

        # keep dims
        IR_node.attr['keepdims'].b = source_node.layer.attr['keep_dims'].b

        # axes
        axes = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor
        axes = tensor_util.MakeNdarray(axes)
        IR_node.attr['axes'].list.i.extend(axes)


    def _convert_layers_batchnorm(self, source_node):
        IR_node = self.IR_graph.node.add()
        TensorflowParser2._copy_and_reop(source_node, IR_node, 'BatchNorm')

        is_transformed = False
        test = self.get_parent(source_node.name, [0])

        if test.type == 'Mul':
            is_transformed = True

        # ssd model is transformed
        if is_transformed:
            # Ax - (Au - b)

            # A
            input_mul_A = self.get_parent(source_node.name, [0, 1])
            tensor_content = input_mul_A.get_attr('value')
            A_content = tensor_util.MakeNdarray(tensor_content)
            self.set_weight(source_node.name, 'A', A_content)

            # b
            input_sub = self.get_parent(source_node.name, [1])
            tensor_content = input_sub.get_attr('value')
            sub_content = tensor_util.MakeNdarray(tensor_content)
            # print(sub_content)
            self.set_weight(source_node.name, 'b', sub_content)

            input_node = self.get_parent(source_node.name, [0])
            IR_node.input.append(input_node.real_name)
            IR_node.attr["_output_shapes"].list.shape.pop()
            IR_node.attr["_output_shapes"].MergeFromString(input_node.layer.attr['_output_shapes'].SerializeToString())

        else:
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
                print(moving_variance.layer)
                assert False

            # gamma (scale)
            Rsqrt = self.get_son(source_node.name, [0], True)
            # print(Rsqrt.out_edges)

            if len(Rsqrt.out_edges) == 2:
                IR_node.attr['scale'].b = False
                output_node = self.get_son(Rsqrt.name, [0, 0], True)
                if output_node.type == 'Sub':
                    output_node = self.get_son(Rsqrt.name, [1, 0], True)
                    Mul = self.get_son(Rsqrt.name, [0], True)
                else:
                    Mul = self.get_son(Rsqrt.name, [1], True)
            else:
                IR_node.attr['scale'].b = True
                son = self.get_son(Rsqrt.name, [0, 0], True)
                gamma_from = self.get_parent(son.name, [1, 1], True)
                gamma = self.check_const(gamma_from)
                gamma_tensor = gamma.get_attr('value')
                scale = tensor_util.MakeNdarray(gamma_tensor)
                self.set_weight(source_node.name, 'scale', scale)
                output_node = self.get_son(source_node.name, [0, 0, 0, 0], True)
                if output_node.type == 'Sub':
                    output_node = self.get_son(source_node.name, [0, 0, 0, 0, 0], True)
                    Mul = self.get_son(Rsqrt.name, [0, 0], True)
                else:
                    Mul = self.get_son(Rsqrt.name, [0, 1], True)

            # beta  (bias)
            beta = self.get_parent(output_node.name, [1, 0, 0], True).get_attr('value')
            bias = tensor_util.MakeNdarray(beta)
            IR_node.attr['bias'].b = True
            self.set_weight(source_node.name, 'bias', bias)

            # moving mean (mean)
            moving_mean = self.get_parent(Mul.name, [0, 0]).get_attr('value')
            mean = tensor_util.MakeNdarray(moving_mean)
            self.set_weight(source_node.name, 'mean', mean)

            # input node
            assert output_node.type == 'Add'
            input_node = self.get_parent(output_node.name, [0, 0])
            IR_node.input.append(input_node.real_name)
            IR_node.attr["_output_shapes"].list.shape.pop()
            IR_node.attr["_output_shapes"].MergeFromString(input_node.layer.attr['_output_shapes'].SerializeToString())
            output_node.real_name = source_node.name


    def _convert_layers_instancenorm(self, source_node):
        IR_node = self.IR_graph.node.add()
        TensorflowParser2._copy_and_reop(source_node, IR_node, 'InstanceNorm')

        # epsilon
        epsilon = self.get_parent(source_node.name, [1])
        epsilon_value = epsilon.get_attr('value').float_val[0]
        IR_node.attr['epsilon'].f = epsilon_value

        # beta
        output_node = self.get_son(source_node.name, [0, 0, 0, 0], True)
        beta = self.get_parent(output_node.name, [1, 0, 0, 0, 0, 1], True)
        beta_tensor = beta.get_attr('value')
        beta = tensor_util.MakeNdarray(beta_tensor)
        self.set_weight(source_node.name, 'bias', beta)


        # gamma (scale)
        IR_node.attr['scale'].b = True
        son = self.get_son(source_node.name, [0, 0, 0], True)
        gamma = self.get_parent(son.name, [1, 1, 0, 0, 0, 1], True)
        gamma_tensor = gamma.get_attr('value')
        scale = tensor_util.MakeNdarray(gamma_tensor)
        self.set_weight(source_node.name, 'scale', scale)
        # output_node = self.get_son(source_node.name, [0, 0, 0, 0], True)

        assert output_node.type == 'Add'
        input_node = self.get_parent(output_node.name, [0, 0])
        IR_node.input.append(input_node.real_name)

        output_node.real_name = source_node.name

        # assert False


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


    def _add_constant_node(self, source_node):
        parent_ids=range(len(source_node.in_edges))
        for idx in parent_ids:
            s = source_node.in_edges[idx]
            parent_node = self.tf_graph.get_node(s)
            if parent_node.type == 'Const':
                self._rename_Const(parent_node)

    def _rename_Const(self, source_node):
        IR_node = self._convert_identity_operation(source_node, end_idx=0, new_op='Constant') # Constant
        value = source_node.get_attr('value')
        if value.float_val:
            value = value.float_val[0]
        elif value.int_val:
            value = value.int_val[0]
        else:
            value = tensor_util.MakeNdarray(value).tolist()
        kwargs = {'value': value}
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
                if source_node.type == 'Enter':
                    IR_node.attr["dtype"].type = TensorflowParser2.dtype_map[6]
                else:
                    assert source_node.layer.attr['T'].type in TensorflowParser2.dtype_map, 'type [{}] is unknown.'.format(source_node.layer.attr['dtype'].type)
                    IR_node.attr["dtype"].type = TensorflowParser2.dtype_map[source_node.layer.attr['T'].type]
            else:
                # Quantized model type
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
        if source_node.layer.attr['shape'].list.shape:
            IR_node.attr['shape'].shape.MergeFromString(source_node.layer.attr['shape'].list.shape[0].SerializeToString())
        else:
            IR_node.attr['shape'].shape.MergeFromString(source_node.layer.attr['shape'].shape.SerializeToString())

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

        variable = self.check_const(self.tf_graph.get_node(add_node.in_edges[1])) #add_bias node
        if not variable or variable.type != 'Const':
            return


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


    def rename_Relu6(self, source_node):
        self._convert_identity_operation(source_node, new_op = 'Relu6')


    def rename_Merge(self, source_node):
        # In facenet or other newtwork using slim.batch_norm,
        # There are two BN(train, test) skip switch and merge.
        source_node.real_name =  self.src_graph.get_node(source_node.in_edges[0]).real_name


    def rename_DepthwiseConv2dNative(self, source_node):
        IR_node = self._convert_identity_operation(source_node, end_idx = 1, new_op = 'DepthwiseConv')
        kwargs = {}
        kwargs['strides'] = source_node.get_attr('strides')
        input_node = self.src_graph.get_parent(source_node.name, [1])
        kwargs['kernel_shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        self._convert_padding(source_node, IR_node, kwargs['kernel_shape'][:-2])


        weight_node = self.src_graph.get_parent(source_node.name, [1])
        weight = self.check_const(weight_node).get_attr('value')
        weight_content = tensor_util.MakeNdarray(weight)
        self.set_weight(source_node.name, 'weights', weight_content)
        assign_IRnode_values(IR_node, kwargs)
        self._get_bias(source_node, IR_node)


    def rename_BatchNormWithGlobalNormalization(self, source_node):

        IR_node = self._convert_identity_operation(source_node, start_idx=0, end_idx=1, new_op='BatchNorm')
        # epsilon
        IR_node.attr['epsilon'].f = source_node.get_attr('variance_epsilon')

        # moving variance (var) /read
        moving_variance = self.get_parent(source_node.name, [2])
        tensor_variance = moving_variance.get_attr('value')
        moving_variance_content = tensor_util.MakeNdarray(tensor_variance)
        self.set_weight(source_node.name, 'var', moving_variance_content)

        # gamma (scale)
        gamma = self.get_parent(source_node.name, [4])
        gamma_value = gamma.get_attr('value')
        gamma = tensor_util.MakeNdarray(gamma_value)
        self.set_weight(source_node.name, 'scale', gamma)
        IR_node.attr['scale'].b = True

        # beta  (bias)
        beta = self.get_parent(source_node.name, [3])
        beta_value = beta.get_attr('value')
        beta = tensor_util.MakeNdarray(beta_value)
        self.set_weight(source_node.name, 'bias', beta)
        IR_node.attr['use_bias'].b = True

        # moving mean (mean)
        mean = self.get_parent(source_node.name, [1])
        mean_value = mean.get_attr('value')
        mean = tensor_util.MakeNdarray(mean_value)
        self.set_weight(source_node.name, 'mean', mean)

    def rename_Placeholder(self, source_node):
        if source_node.layer.attr["shape"].shape.unknown_rank == True:
            return
        IR_node = self._convert_identity_operation(source_node, new_op='DataInput')
        TensorflowParser2._copy_shape(source_node, IR_node)
        IR_node.attr['shape'].shape.dim[0].size = -1
        IR_node.attr['_output_shapes'].list.shape[0].dim[0].size = -1


    def rename_Mean(self, source_node):
        # ReduceMean
        IR_node = self._convert_identity_operation(source_node, start_idx = 0, end_idx = 1, new_op='ReduceMean')
        # keep dims
        IR_node.attr['keepdims'].b = source_node.layer.attr['keep_dims'].b

        # axes
        axes = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor
        axes = tensor_util.MakeNdarray(axes)
        IR_node.attr['axes'].list.i.extend(axes)


    def rename_Reshape(self, source_node):
        IR_node = self._convert_identity_operation(source_node, end_idx = 1)
        kwargs = {'shape' : self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]}
        assign_IRnode_values(IR_node, kwargs)


    def rename_MirrorPad(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'MirrorPad')
        input_node = self.src_graph.get_parent(source_node.name, [1])

        tensor_content = tensor_util.MakeNdarray(input_node.get_attr('value')).reshape(-1)
        kwargs = {}
        kwargs['mode'] = source_node.get_attr('mode')
        kwargs['pads'] = tensor_content.tolist()

        assign_IRnode_values(IR_node, kwargs)


    def rename_Min(self, source_node):
        IR_node = self._convert_identity_operation(source_node, start_idx=0, end_idx=1, new_op = 'Min')
        kwargs = {}
        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs['shape_0'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        input_node = self.src_graph.get_parent(source_node.name, [1])
        kwargs['shape_1'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
        assign_IRnode_values(IR_node, kwargs)


    def rename_Max(self, source_node):
        IR_node = self._convert_identity_operation(source_node, start_idx=0, end_idx=1, new_op = 'Max')
        kwargs = {}
        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs['shape_0'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        input_node = self.src_graph.get_parent(source_node.name, [1])
        kwargs['shape_1'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
        assign_IRnode_values(IR_node, kwargs)


    def rename_Mul(self, source_node):
        scopes = self._get_scopes(source_node.name)

        if len(scopes) >= 2:
            if scopes[-2] == "batchnorm" or scopes[-2].startswith("Assign"):
                return
        self._add_constant_node(source_node)
        self._convert_identity_operation(source_node)


    def rename_Add(self, source_node):
        scopes = self._get_scopes(source_node.name)
        if len(scopes) > 2:
            if scopes[-2] == 'batchnorm':
                if scopes[-3] == 'BatchNorm' or scopes[-3] == 'batch_normalization':
                    self._convert_layers_batchnorm(source_node)
                elif scopes[-3] == 'InstanceNorm':
                    self._convert_layers_instancenorm(source_node)
            else:
                IR_node = self._convert_identity_operation(source_node, new_op = "Add")
        else:
            IR_node = self._convert_identity_operation(source_node, new_op = "Add")

    def rename_AddV2(self, source_node):
        self.rename_Add(source_node)

    def rename_Fill(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op="Fill")


    def rename_Sub(self, source_node):
        scopes = self._get_scopes(source_node.name)
        if len(scopes) > 2:
            if scopes[-2].startswith('Assign') or scopes[-1].startswith('Assign'):
                return
        IR_node = self._convert_identity_operation(source_node, end_idx=2, new_op = "Sub")


    def rename_Sum(self, source_node):
        IR_node = self._convert_identity_operation(source_node, start_idx=0, end_idx=1, new_op = 'Sum')
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
        IR_node = self._convert_identity_operation(source_node, start_idx=0, end_idx=1, new_op = 'Reciprocal')


    def rename_Minimum(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'Minimum')

    def rename_Maximum(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'Maximum')

    def rename_RealDiv(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'RealDiv')

    def rename_Enter(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'Enter')

    def rename_Switch(self, source_node):
        # Skip the node as merge
        source_node.real_name =  self.src_graph.get_node(source_node.in_edges[0]).real_name


    def rename_Identity(self, source_node):
        source_node.real_name =  self.src_graph.get_node(source_node.in_edges[0]).real_name


    def rename_Exp(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'Exp')

    def rename_ResizeBilinear(self, source_node):
        IR_node = self._convert_identity_operation(source_node,end_idx=1, new_op = 'ResizeBilinear')


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


    def rename_Gather(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='Embedding')

        W = self.src_graph.get_parent(source_node.name, [0])
        W = self.src_graph.get_parent(W.name, [0])

        self.set_weight(source_node.name, "weights", self.ckpt_data[W.name])

        kwargs = {
            'input_dim' : self.ckpt_data[W.name].shape[0],
            'output_dim' : self.ckpt_data[W.name].shape[1],
            'mask_zero' : False
        }
        kwargs['axis'] = 0  # add default
        assign_IRnode_values(IR_node, kwargs)

        return IR_node

    def rename_GatherV2(self, source_node):

        IR_node = self.rename_Gather(source_node)

        kwargs = {}
        kwargs['axis'] = source_node.layer.attr['axis'].i
        assign_IRnode_values(IR_node, kwargs)


    def rename_StridedSlice(self, source_node):
        IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op = 'Slice')
        kwargs = {}
        kwargs = {
            'begin_mask' : source_node.get_attr('begin_mask'),
            'end_mask'   : source_node.get_attr('end_mask'),
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
        tensor_content = self.check_const(input_node_weight).get_attr('value')
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
        self._convert_pooling(source_node, b'MAX')


    def rename_AvgPool(self, source_node):
        self._convert_pooling(source_node, b'AVG')


    def rename_LRN(self, source_node):
        IR_node = self._convert_identity_operation(source_node)
        size = source_node.get_attr('depth_radius') * 2 + 1
        alpha = source_node.get_attr('alpha') * size
        beta = source_node.get_attr('beta')
        bias = source_node.get_attr('bias')

        IR_node.attr["alpha"].f = alpha
        IR_node.attr["beta"].f = beta
        IR_node.attr["size"].i = size
        IR_node.attr["bias"].f = bias


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
        IR_node = self._convert_identity_operation(source_node, end_idx=1)
        input_weight_node = self.src_graph.get_parent(source_node.name, [1])
        weightnode = self.check_const(input_weight_node)
        weight_value = weightnode.get_attr('value')

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
            biasnode = self.check_const(variable)
            bias_value = biasnode.get_attr('value')
            bias = tensor_util.MakeNdarray(bias_value)
            self.set_weight(source_node.name, 'bias', bias)
            IR_node.attr['use_bias'].b = True


    def rename_Softmax(self, source_node):
        IR_node = self._convert_identity_operation(source_node)
        kwargs = {'shape' : self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]}
        IR_node.attr["dim"].i = 1
        assign_IRnode_values(IR_node, kwargs)


    def rename_BiasAdd(self, source_node):
        # Skip BiasAdd
        source_node.real_name =  self.src_graph.get_node(source_node.in_edges[0]).real_name


    def rename_QuantizeV2(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'QuantizeV2')
        TensorflowParser2._copy_shape(source_node, IR_node)


    def rename_QuantizedRelu(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = "QuantizedRelu")
        kwargs = {'shape' : self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]}
        assign_IRnode_values(IR_node, kwargs)


    def rename_QuantizedReshape(self, source_node):
        IR_node = self._convert_identity_operation(source_node, end_idx = 1)
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


    def rename_Requantize(self, source_node):
        input_node = self.get_parent(source_node.name, [0])
        son_node = self.get_son(source_node.name, [0])

        son_node.real_name = source_node.name

    def rename_RequantizationRange(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'RequantizationRange')
        TensorflowParser2._copy_shape(source_node, IR_node)


    def rename_ZerosLike(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'ZerosLike')


    def rename_Rank(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'Rank')


    def rename_Transpose(self, source_node):
        IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op = 'Transpose')

        input_node_perm = self.get_parent(source_node.name, [1])
        # input_node_perm = self.check_const(self.get_parent(source_node.name, [1], True))
        tensor_content = input_node_perm.get_attr('value')
        perm = tensor_util.MakeNdarray(tensor_content).tolist()
        assign_IRnode_values(IR_node, {'perm' : perm})

    def rename_GreaterEqual(self, source_node):
        IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op = 'GreaterEqual')


    def rename_Greater(self, source_node):
        IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op = 'Greater')


    def rename_Equal(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'Equal')


    def rename_All(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'All')


    def rename_LogicalAnd(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'Mul')


    def rename_Pad(self, source_node):
        IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op = 'Pad')
        kwargs = {}
        kwargs['mode'] = 'constant'

        # paddings
        padding = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor
        shapes = tensor_util.MakeNdarray(padding)
        kwargs['pads'] = convert_tf_pad_to_onnx(shapes)

        assign_IRnode_values(IR_node, kwargs)


    def rename_FusedBatchNorm(self, source_node):
        scalenode = self.check_const(self.get_parent(source_node.name, [1], True))
        if ':' in source_node.in_edges[1]: # ?
            scalenode = None

        if scalenode:
            scale_value = scalenode.get_attr('value')
            IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op = 'BatchNorm')
            # for attr.shape >= 2
            for i in range(len(IR_node.attr["_output_shapes"].list.shape)-1):
                IR_node.attr["_output_shapes"].list.shape.pop()

        else:
            # For models built by slim.batch_norm, remove duplicate BN (eg.facenet)
            return

        scale = tensor_util.MakeNdarray(scale_value)
        self.set_weight(source_node.name, 'scale', scale)
        IR_node.attr['scale'].b = True


        IR_node.attr['epsilon'].f = source_node.get_attr('epsilon', 0)
        biasnode = self.check_const(self.get_parent(source_node.name, [2], True))
        if biasnode:
            bias_value = biasnode.get_attr('value')
        else:
            innode = self.get_parent(source_node.name, [2], True)
            name = innode.name.split(':')[0]
            bias_value = self.check_const(self.src_graph.layer_map[name]).get_attr('value')
        bias = tensor_util.MakeNdarray(bias_value)
        self.set_weight(source_node.name, 'bias', bias)
        IR_node.attr['bias'].b = True

        meannode = self.check_const(self.get_parent(source_node.name, [3], True))
        mean_value = meannode.get_attr('value')
        mean = tensor_util.MakeNdarray(mean_value)
        self.set_weight(source_node.name, 'mean', mean)

        variancenode = self.check_const(self.get_parent(source_node.name, [4], True))
        variance_value = variancenode.get_attr('value')
        variance = tensor_util.MakeNdarray(variance_value)
        self.set_weight(source_node.name, 'var', variance)

    def rename_FusedBatchNormV3(self, source_node):
        self.rename_FusedBatchNorm(source_node)


    def rename_SpaceToBatchND(self, source_node):
        IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op = 'SpaceToBatchND')


    def rename_BatchToSpaceND(self, source_node):
        IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op = 'BatchToSpaceND')


    def rename_ArgMax(self, source_node):
        IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op = 'ArgMax')


    def rename_Slice(self, source_node):
        input_node_begin = self.get_parent(source_node.name, [1])
        input_node_size = self.get_parent(source_node.name, [2])

        begin = tensor_util.MakeNdarray(input_node_begin.layer.attr['value'].tensor)
        size = tensor_util.MakeNdarray(input_node_size.layer.attr['value'].tensor)

        IR_node = self._convert_identity_operation(source_node, new_op='Slice')

        # TODO:  only for 1D
        end = size + begin
        kwargs = {
            'starts': begin,
            'ends': end
        }

        assign_IRnode_values(IR_node, kwargs)


    def rename_Split(self, source_node):
        if source_node.get_attr('num_split') == 1:
            source_node.real_name = self.get_parent(source_node.name, [1]).real_name

        else:
            IR_node = self._convert_identity_operation(source_node, start_idx=1, new_op = 'Split')
            kwargs = {
                'axis' : self.get_parent(source_node.name, [0]).layer.attr['value'].tensor.int_val[0],
                'split' : source_node.get_attr('num_split')
            }
            assign_IRnode_values(IR_node, kwargs)


    def rename_Tile(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'Tile')


    def rename_Sqrt(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'Sqrt')


    def rename_Tanh(self, source_node):
        IR_node = self._convert_identity_operation(source_node)

        kwargs = {}
        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]

        assign_IRnode_values(IR_node, kwargs)

    def rename_Log(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = 'Log')