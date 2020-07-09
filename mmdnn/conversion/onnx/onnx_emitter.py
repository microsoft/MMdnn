from mmdnn.conversion.common.DataStructure.emitter import Emitter
from mmdnn.conversion.common.IR.IR_graph import IRGraph
import os.path
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
import numpy as np
import sys


class OnnxEmitter(Emitter):
    dtype_map = {
        graph_pb2.DT_FLOAT32: "TensorProto.FLOAT"
    }

    transpose_map = {
        1: 2,
        2: 3,
        -1: 1
    }

    def __init__(self, architecture, weight):
        super(OnnxEmitter, self).__init__()
        if os.path.exists(architecture) == False:
            raise ValueError("IR architecture file [{}] is not found.".format(architecture))
        else:
            self.IR_graph = IRGraph(architecture)
            self.IR_graph.build()

        if os.path.exists(weight) == False:
            raise ValueError("IR weight file [{}] is not found.".format(weight))
        else:
            self._load_weights(weight)

    @property
    def header_code(self):
        return """import numpy as np
from onnx import helper, TensorProto
import onnx

_weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file = None):
    global _weights_dict
    _weights_dict = load_weights(weight_file)

"""

    def gen_code(self, phase):
        self.phase = phase
        self.add_body(0, self.header_code)

        self.inputs = []
        self.outputs = []
        self.nodes = []
        self.initializer = []

        for layer in self.IR_graph.topological_sort:
            current_node = self.IR_graph.get_node(layer)
            node_type = current_node.type

            if hasattr(self, "emit_" + node_type):
                func = getattr(self, "emit_" + node_type)
                func(current_node)
            else:
                print("OnnxEmitter has not supported operator [%s]." % (node_type))
                self.emit_UNKNOWN(current_node)

        self._process_output_layers()

        self.add_body(1, "graph = helper.make_graph([{}], 'mmdnn', [{}], [{}], [{}])".format(', '.join(self.nodes),
                                                                                             ', '.join(self.inputs),
                                                                                             ', '.join(self.outputs),
                                                                                             ', '.join(
                                                                                                 self.initializer))
                      )
        self.add_body(1, "return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 6)])")
        return self.body_code

    def run(self, dstNetworkPath, dstWeightPath=None, phase='test'):
        super(OnnxEmitter, self).run(dstNetworkPath, dstWeightPath, phase)
        self.save_weights(self.weights_dict, dstWeightPath)

    def check_if_need_transpose(self, IR_node):
        parent = self.IR_graph.get_parent(IR_node.name, [0])
        while parent.type == 'Flatten' or parent.type == 'Dropout':
            parent = self.IR_graph.get_parent(parent.name, [0])
        dim = len(parent.layer.attr['_output_shapes'].list.shape[0].dim)
        if dim > 2:
            original_dims = self.weights_dict[IR_node.name]['weights'].shape
            dims = [i.size for i in parent.layer.attr['_output_shapes'].list.shape[0].dim[1:]] + [-1]
            self.weights_dict[IR_node.name]['weights'] = self.weights_dict[IR_node.name]['weights']
            self.weights_dict[IR_node.name]['weights'] = np.reshape(self.weights_dict[IR_node.name]['weights'], dims)
            self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], [dim - 2] + list(range(0, dim - 2)) + [dim - 1])
            self.weights_dict[IR_node.name]['weights'] = np.reshape(self.weights_dict[IR_node.name]['weights'], original_dims)

    def _process_output_layers(self):
        for name in self.IR_graph.output_layers:
            IR_node = self.IR_graph.get_node(self.IR_graph.get_node(name).real_name)
            # omit node of some type
            if IR_node.type == 'Shape' or IR_node.type == 'Pack':
                continue
            shape_str = IRGraph.shapeToStr(IR_node.layer.attr["_output_shapes"].list.shape[0])

            if IR_node.layer.attr['dtype'].type == graph_pb2.DT_UNDEFINED:
                IR_node.layer.attr['dtype'].type = graph_pb2.DT_FLOAT32
            dtype_str = self.dtype_map[IR_node.layer.attr['dtype'].type]
            self.add_body(1, "{:<15} = helper.make_tensor_value_info('{}', {}, ({},))".format(
                IR_node.variable_name + '_out',
                IR_node.variable_name,
                dtype_str,
                shape_str
                ))
            self.outputs.append(IR_node.variable_name + '_out')

    def emit_DataInput(self, IR_node):
        shape = [dim.size if dim.size != -1 else 1 for dim in IR_node.IR_layer.attr["shape"].shape.dim]
        shape_str = ', '.join('%s' % i for i in shape)
        if IR_node.layer.attr['dtype'].type == graph_pb2.DT_UNDEFINED:
            IR_node.layer.attr['dtype'].type = graph_pb2.DT_FLOAT32
        dtype_str = self.dtype_map[IR_node.layer.attr['dtype'].type]
        self.add_body(1, "{:<15} = helper.make_tensor_value_info('{}', {}, ({},))".format(
            IR_node.variable_name + '_orig',
            IR_node.variable_name + '_orig',
            dtype_str,
            shape_str))
        self.add_body(1, "{:15} = helper.make_node('Transpose', inputs=['{}'], outputs=['{}'], perm=[0, 3, 1, 2], name='{}')".format(
            IR_node.variable_name,
            IR_node.variable_name + '_orig',
            IR_node.variable_name,
            IR_node.variable_name
            ))
        self.inputs.append(IR_node.variable_name + '_orig')
        self.nodes.append(IR_node.variable_name)

    def emit_Conv(self, IR_node):
        kernel_shape = list(IR_node.get_attr('kernel_shape'))[:-2]
        dilations = list(IR_node.get_attr('dilations', [1] * (len(kernel_shape) + 2)))[1:-1]
        group = IR_node.get_attr('group', 1)
        if IR_node.type == 'DepthwiseConv':
            group = IR_node.IR_layer.attr["kernel_shape"].list.i[-2]
            self.weights_dict[IR_node.name]['weights'] = np.swapaxes(self.weights_dict[IR_node.name]['weights'], -1, -2)
        pads = IR_node.get_attr('pads')
        pad_length = len(pads)
        pads = pads[1:pad_length // 2 - 1] + pads[pad_length // 2 + 1:pad_length - 1]
        strides = list(IR_node.get_attr('strides'))[1:-1]
        use_bias=IR_node.get_attr('use_bias')
        self.add_body(1, "{:15} = _weights_dict['{}']['weights']".format(
            IR_node.variable_name + '_weight_array',
            IR_node.name))
        self.add_body(1, "{} = {}.transpose([3,2,0,1])".format(
            IR_node.variable_name + '_weight_array',
            IR_node.variable_name + '_weight_array'))
        self.add_body(1, "{:15} = helper.make_tensor_value_info('{}', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], list({}.shape))".format(
                          IR_node.variable_name + '_weight',
                          IR_node.variable_name + '_weight',
                          IR_node.variable_name + '_weight_array',
                          IR_node.variable_name + '_weight_array'))

        self.add_body(1, "{:15} = helper.make_tensor(name='{}', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], dims={}.shape, vals={}.flatten().astype(float))".format(
                          IR_node.variable_name + '_weight_init',
                          IR_node.variable_name + '_weight',
                          IR_node.variable_name + '_weight_array',
                          IR_node.variable_name + '_weight_array',
                          IR_node.variable_name + '_weight_array'))

        if use_bias:
            self.add_body(1, "{:15} = _weights_dict['{}']['bias'].squeeze()".format(
                IR_node.variable_name + '_bias_array',
                IR_node.name))

            self.add_body(1, "{:15} = helper.make_tensor_value_info('{}', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], list({}.shape))".format(
                            IR_node.variable_name + '_bias',
                            IR_node.variable_name + '_bias',
                            IR_node.variable_name + '_bias_array',
                            IR_node.variable_name + '_bias_array'))

            self.add_body(1, "{:15} = helper.make_tensor(name='{}', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], dims={}.shape, vals={}.flatten().astype(float))".format(
                              IR_node.variable_name + '_bias_init',
                              IR_node.variable_name + '_bias',
                              IR_node.variable_name + '_bias_array',
                              IR_node.variable_name + '_bias_array',
                              IR_node.variable_name + '_bias_array'))
            self.add_body(1, "{:15} = helper.make_node('Conv', inputs=['{}', '{}', '{}'],outputs=['{}'], dilations={}, group={}, kernel_shape={}, pads={}, strides={}, name='{}')".format(
                              IR_node.variable_name,
                              self.parent_variable_name(IR_node),
                              IR_node.variable_name + '_weight',
                              IR_node.variable_name + '_bias',
                              IR_node.variable_name,
                              dilations,
                              group,
                              kernel_shape,
                              pads,
                              strides,
                              IR_node.variable_name))
            # self.nodes.append(IR_node.variable_name + '_bias')
            self.initializer.append(IR_node.variable_name + '_bias_init')
            self.inputs.append(IR_node.variable_name + '_bias')
        else:
            self.add_body(1, "{:15} = helper.make_node('Conv', inputs=['{}', '{}'], outputs=['{}'], dilations={}, group={}, kernel_shape={}, pads={}, strides={}, name='{}')".format(
                              IR_node.variable_name,
                              self.parent_variable_name(IR_node),
                              IR_node.variable_name + '_weight',
                              IR_node.variable_name,
                              dilations,
                              group,
                              kernel_shape,
                              pads,
                              strides,
                              IR_node.variable_name))
        # self.nodes.append(IR_node.variable_name + '_weight')
        self.initializer.append(IR_node.variable_name + '_weight_init')
        self.inputs.append(IR_node.variable_name + '_weight')
        self.nodes.append(IR_node.variable_name)

    def emit_BatchNorm(self, IR_node):
        epsilon = IR_node.get_attr('epsilon')
        if IR_node.get_attr('scale'):
            self.add_body(1, "{:15} = _weights_dict['{}']['scale'].squeeze()".format(
                IR_node.variable_name + '_scale_array',
                IR_node.name))
        else:
            self.add_body(1, "{:15} = np.ndarray(_weights_dict['{}']['bias'].shape, dtype=_weights_dict['{}']['bias'].dtype).squeeze()".format(
                              IR_node.variable_name + '_scale_array',
                              IR_node.name,
                              IR_node.name))
            self.add_body(1, "{:15}.fill(1)".format(IR_node.variable_name + '_scale_array'))

        self.add_body(1, "{:15} = helper.make_tensor_value_info('{}', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], list({}.shape))".format(
            IR_node.variable_name + '_scale',
            IR_node.variable_name + '_scale',
            IR_node.variable_name + '_scale_array',
            IR_node.variable_name + '_scale_array'))
    
        self.add_body(1, "{:15} = helper.make_tensor(name='{}', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], dims={}.shape, vals={})".format(
                          IR_node.variable_name + '_scale_init',
                          IR_node.variable_name + '_scale',
                          IR_node.variable_name + '_scale_array',
                          IR_node.variable_name + '_scale_array',
                          IR_node.variable_name + '_scale_array'))
        self.add_body(1, "{:15} = _weights_dict['{}']['bias'].squeeze()".format(
            IR_node.variable_name + '_bias_array',
            IR_node.name))
        self.add_body(1, "{:15} = helper.make_tensor_value_info('{}', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], list({}.shape))".format(
                        IR_node.variable_name + '_bias',
                        IR_node.variable_name + '_bias',
                        IR_node.variable_name + '_bias_array',
                        IR_node.variable_name + '_bias_array'))

        self.add_body(1, "{:15} = helper.make_tensor(name='{}', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], dims={}.shape, vals={}.flatten().astype(float))".format(
                            IR_node.variable_name + '_bias_init',
                            IR_node.variable_name + '_bias',
                            IR_node.variable_name + '_bias_array',
                            IR_node.variable_name + '_bias_array',
                            IR_node.variable_name + '_bias_array'))

        self.add_body(1, "{:15} = _weights_dict['{}']['mean'].squeeze()".format(
            IR_node.variable_name + '_mean_array',
            IR_node.name))
        self.add_body(1, "{:15} = helper.make_tensor_value_info('{}', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], list({}.shape))".format(
                        IR_node.variable_name + '_mean',
                        IR_node.variable_name + '_mean',
                        IR_node.variable_name + '_mean_array',
                        IR_node.variable_name + '_mean_array'))

        self.add_body(1, "{:15} = helper.make_tensor(name='{}', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], dims={}.shape, vals={}.flatten().astype(float))".format(
                            IR_node.variable_name + '_mean_init',
                            IR_node.variable_name + '_mean',
                            IR_node.variable_name + '_mean_array',
                            IR_node.variable_name + '_mean_array',
                            IR_node.variable_name + '_mean_array'))

        self.add_body(1, "{:15} = _weights_dict['{}']['var'].squeeze()".format(
                          IR_node.variable_name + '_var_array',
                          IR_node.name))
        self.add_body(1, "{:15} = helper.make_tensor_value_info('{}', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], list({}.shape))".format(
                            IR_node.variable_name + '_var',
                            IR_node.variable_name + '_var',
                            IR_node.variable_name + '_var_array',
                            IR_node.variable_name + '_var_array'))
    
        self.add_body(1, "{:15} = helper.make_tensor(name='{}', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], dims={}.shape, vals={}.flatten().astype(float))".format(
                            IR_node.variable_name + '_var_init',
                            IR_node.variable_name + '_var',
                            IR_node.variable_name + '_var_array',
                            IR_node.variable_name + '_var_array',
                            IR_node.variable_name + '_var_array'))                         

        self.add_body(1, "{:15} = helper.make_node('BatchNormalization', inputs=['{}', '{}', '{}', '{}', '{}'],outputs=['{}'], epsilon={}, is_test={}, name='{}')".format(
                          IR_node.variable_name,
                          self.parent_variable_name(IR_node),
                          IR_node.variable_name + '_scale',
                          IR_node.variable_name + '_bias',
                          IR_node.variable_name + '_mean',
                          IR_node.variable_name + '_var',
                          IR_node.variable_name,
                          epsilon,
                          0 if self.phase == 'train' else 1,
                          IR_node.variable_name))
        self.initializer.append(IR_node.variable_name + '_scale_init')
        self.initializer.append(IR_node.variable_name + '_bias_init')
        self.initializer.append(IR_node.variable_name + '_mean_init')
        self.initializer.append(IR_node.variable_name + '_var_init')
        self.inputs.append(IR_node.variable_name + '_scale')
        self.inputs.append(IR_node.variable_name + '_bias')
        self.inputs.append(IR_node.variable_name + '_mean')
        self.inputs.append(IR_node.variable_name + '_var')
        self.nodes.append(IR_node.variable_name)


    def emit_Scale(self, IR_node):
        dims = [i.size for i in IR_node.layer.attr['_output_shapes'].list.shape[0].dim[1:]]
        units = dims[-1]
        epsilon = 1e-5
        if IR_node.get_attr('scale'):
            self.add_body(1, "{:15} = _weights_dict['{}']['scale'].squeeze()".format(
                IR_node.variable_name + '_scale_array',
                IR_node.name))
        else:
            self.add_body(1, "{:15} = np.ndarray(_weights_dict['{}']['bias'].shape, dtype=_weights_dict['{}']['bias'].dtype).squeeze()".format(
                              IR_node.variable_name + '_scale_array',
                              IR_node.name,
                              IR_node.name))
            self.add_body(1, "{:15}.fill(1)".format(IR_node.variable_name + '_scale_array'))
        self.add_body(1, "{:15} = helper.make_tensor_value_info('{}', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], list({}.shape))".format(
            IR_node.variable_name + '_scale',
            IR_node.variable_name + '_scale',
            IR_node.variable_name + '_scale_array',
            IR_node.variable_name + '_scale_array'))
    
        self.add_body(1, "{:15} = helper.make_tensor(name='{}', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], dims={}.shape, vals={})".format(
                            IR_node.variable_name + '_scale_init',
                            IR_node.variable_name + '_scale',
                            IR_node.variable_name + '_scale_array',
                            IR_node.variable_name + '_scale_array',
                            IR_node.variable_name + '_scale_array'))
        self.add_body(1, "{:15} = _weights_dict['{}']['bias'].squeeze()".format(
            IR_node.variable_name + '_bias_array',
            IR_node.name))
        self.add_body(1, "{:15} = helper.make_tensor_value_info('{}', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], list({}.shape))".format(
            IR_node.variable_name + '_bias',
            IR_node.variable_name + '_bias',
            IR_node.variable_name + '_bias_array',
            IR_node.variable_name + '_bias_array'))

        self.add_body(1, "{:15} = helper.make_tensor(name='{}', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], dims={}.shape, vals={}.flatten().astype(float))".format(
                            IR_node.variable_name + '_bias_init',
                            IR_node.variable_name + '_bias',
                            IR_node.variable_name + '_bias_array',
                            IR_node.variable_name + '_bias_array',
                            IR_node.variable_name + '_bias_array'))
        self.add_body(1, "{:15} = np.zeros({}, dtype=np.float32)".format(
            IR_node.variable_name + '_mean_array',
            units))
        self.add_body(1, "{:15} = helper.make_tensor_value_info('{}', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], list({}.shape))".format(
            IR_node.variable_name + '_mean',
            IR_node.variable_name + '_mean',
            IR_node.variable_name + '_mean_array',
            IR_node.variable_name + '_mean_array'))

        self.add_body(1, "{:15} = helper.make_tensor(name='{}', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], dims={}.shape, vals={}.flatten().astype(float))".format(
                            IR_node.variable_name + '_mean_init',
                            IR_node.variable_name + '_mean',
                            IR_node.variable_name + '_mean_array',
                            IR_node.variable_name + '_mean_array',
                            IR_node.variable_name + '_mean_array'))
        self.add_body(1, "{:15} = np.ones({}, dtype=np.float32)".format(
                          IR_node.variable_name + '_var_array',
                          units))
        self.add_body(1, "{:15} = helper.make_tensor_value_info('{}', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], list({}.shape))".format(
                            IR_node.variable_name + '_var',
                            IR_node.variable_name + '_var',
                            IR_node.variable_name + '_var_array',
                            IR_node.variable_name + '_var_array'))
    
        self.add_body(1, "{:15} = helper.make_tensor(name='{}', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], dims={}.shape, vals={}.flatten().astype(float))".format(
                            IR_node.variable_name + '_var_init',
                            IR_node.variable_name + '_var',
                            IR_node.variable_name + '_var_array',
                            IR_node.variable_name + '_var_array',
                            IR_node.variable_name + '_var_array'))   
        self.add_body(1, "{:15} = helper.make_node('BatchNormalization', inputs=['{}', '{}', '{}', '{}', '{}'],outputs=['{}'], epsilon={}, is_test={}, name='{}')".format(
                          IR_node.variable_name,
                          self.parent_variable_name(IR_node),
                          IR_node.variable_name + '_scale',
                          IR_node.variable_name + '_bias',
                          IR_node.variable_name + '_mean',
                          IR_node.variable_name + '_var',
                          IR_node.variable_name,
                          epsilon,
                          0 if self.phase == 'train' else 1,
                          IR_node.variable_name))
        self.inputs.append(IR_node.variable_name + '_scale')
        self.inputs.append(IR_node.variable_name + '_bias')
        self.inputs.append(IR_node.variable_name + '_mean')
        self.inputs.append(IR_node.variable_name + '_var')
        self.initializer.append(IR_node.variable_name + '_scale_init')
        self.initializer.append(IR_node.variable_name + '_bias_init')
        self.initializer.append(IR_node.variable_name + '_mean_init')
        self.initializer.append(IR_node.variable_name + '_var_init')
        self.nodes.append(IR_node.variable_name)


    def emit_Relu(self, IR_node):
        self.add_body(1, "{:15} = helper.make_node('Relu', inputs=['{}'], outputs=['{}'], name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.variable_name,
            IR_node.variable_name))
        self.nodes.append(IR_node.variable_name)

    def emit_Add(self, IR_node):
        input_layers = ', '.join(
            ("'" + self.IR_graph.get_parent(IR_node.name, [num]).real_variable_name) + "'" for num in
            range(0, len(IR_node.in_edges)))
        self.add_body(1, "{:15} = helper.make_node('Add', inputs=[{}], outputs=['{}'], name='{}')".format(
            IR_node.variable_name,
            input_layers,
            IR_node.variable_name,
            IR_node.variable_name))
        self.nodes.append(IR_node.variable_name)

    def emit_Pool(self, IR_node):
        pooling_type = IR_node.get_attr('pooling_type')
        if IR_node.layer.attr['global_pooling'].b:
            if pooling_type == 'AVG':
                self.add_body(1, "{:15} = helper.make_node('GlobalAveragePool', inputs=['{}'], outputs=['{}'], name='{}')".format(
                    IR_node.variable_name,
                    self.parent_variable_name(IR_node),
                    IR_node.variable_name,
                    IR_node.variable_name))
                self.nodes.append(IR_node.variable_name)
            else:
                print("OnnxEmitter has not supported Global Pool type [%s]." % (pooling_type))
                self.emit_UNKNOWN(IR_node)
        else:
            if pooling_type in ['AVG', 'MAX']:
                if pooling_type == 'AVG':
                    op_name = 'AveragePool'
                elif pooling_type == 'MAX':
                    op_name = 'MaxPool'
                kernel_shape = list(IR_node.get_attr('kernel_shape')[1:-1])
                pads = IR_node.get_attr('pads')
                pad_length = len(pads)
                pads = pads[1:pad_length // 2 - 1] + pads[pad_length // 2 + 1:pad_length - 1]
                strides = list(IR_node.get_attr('strides')[1:-1])
                self.add_body(1, "{:15} = helper.make_node('{}', inputs=['{}'],outputs=['{}'], kernel_shape={}, pads={}, strides={}, name='{}')".format(
                                  IR_node.variable_name,
                                  op_name,
                                  self.parent_variable_name(IR_node),
                                  IR_node.variable_name,
                                  kernel_shape,
                                  pads,
                                  strides,
                                  IR_node.variable_name))
                self.nodes.append(IR_node.variable_name)
            else:
                print("OnnxEmitter has not supported Pool type [%s]." % (pooling_type))
                self.emit_UNKNOWN(IR_node)

    def emit_FullyConnected(self, IR_node):
        self.check_if_need_transpose(IR_node)
        use_bias = IR_node.get_attr('use_bias', True)
        units = IR_node.get_attr('units')

        self.add_body(1, "{:15} = _weights_dict['{}']['weights']".format(
            IR_node.variable_name + '_weight_array',
            IR_node.name))

        self.add_body(1, "{:15} = helper.make_tensor_value_info('{}', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], list({}.shape))".format(
            IR_node.variable_name + '_weight',
            IR_node.variable_name + '_weight',
            IR_node.variable_name + '_weight_array',
            IR_node.variable_name + '_weight_array'))

        self.add_body(1, "{:15} = helper.make_tensor(name='{}', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], dims={}.shape, vals={}.flatten().astype(float))".format(
            IR_node.variable_name + '_weight_init',
            IR_node.variable_name + '_weight',
            IR_node.variable_name + '_weight_array',
            IR_node.variable_name + '_weight_array',
            IR_node.variable_name + '_weight_array'))

        if use_bias:
            self.add_body(1, "{:15} = _weights_dict['{}']['bias'].squeeze()".format(
                IR_node.variable_name + '_bias_array',
                IR_node.name))
        else:
            self.add_body(1, "{:15} = np.zeros({})".format(
                IR_node.variable_name + '_bias_array',
                units))
        self.add_body(1, "{:15} = helper.make_tensor_value_info('{}', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], list({}.shape))".format(
            IR_node.variable_name + '_bias',
            IR_node.variable_name + '_bias',
            IR_node.variable_name + '_bias_array',
            IR_node.variable_name + '_bias_array'))

        self.add_body(1, "{:15} = helper.make_tensor(name='{}', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], dims={}.shape, vals={}.flatten().astype(float))".format(
                            IR_node.variable_name + '_bias_init',
                            IR_node.variable_name + '_bias',
                            IR_node.variable_name + '_bias_array',
                            IR_node.variable_name + '_bias_array',
                            IR_node.variable_name + '_bias_array'))
        self.add_body(1, "{:15} = helper.make_node('Gemm', inputs=['{}', '{}', '{}'], outputs=['{}'], name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.variable_name + '_weight',
            IR_node.variable_name + '_bias',
            IR_node.variable_name,
            IR_node.variable_name))
        self.initializer.append(IR_node.variable_name + '_weight_init')
        self.initializer.append(IR_node.variable_name + '_bias_init')
        self.inputs.append(IR_node.variable_name + '_weight')
        self.inputs.append(IR_node.variable_name + '_bias')
        self.nodes.append(IR_node.variable_name)

    def emit_Pad(self, IR_node):
        mode = IR_node.layer.attr['mode'].s.decode()
        pads = IR_node.get_attr('pads')
        pad_length = len(pads)
        pads = [0, 0] + pads[1:pad_length // 2 - 1] + [0, 0] + pads[pad_length // 2 + 1:pad_length - 1]
        self.add_body(1, "{:15} = helper.make_node('Pad', inputs=['{}'], outputs=['{}'], mode='{}', pads={}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.variable_name,
            mode,
            pads,
            IR_node.variable_name))
        self.nodes.append(IR_node.variable_name)

    def emit_Concat(self, IR_node):
        axis = IR_node.get_attr('axis') - 2
        inputs = ', '.join("'" + self.IR_graph.get_node(i).real_variable_name + "'" for i in IR_node.in_edges)
        self.add_body(1, "{:15} = helper.make_node('Concat', inputs=[{}], outputs=['{}'], axis={}, name='{}')".format(
            IR_node.variable_name,
            inputs,
            IR_node.variable_name,
            axis,
            IR_node.variable_name))
        self.nodes.append(IR_node.variable_name)

    def emit_Flatten(self, IR_node):
        self.add_body(1, "{:15} = helper.make_node('Flatten', inputs=['{}'], outputs=['{}'], name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.variable_name,
            IR_node.variable_name))
        self.nodes.append(IR_node.variable_name)

    def emit_Softmax(self, IR_node):
        self.add_body(1, "{:15} = helper.make_node('Softmax', inputs=['{}'], outputs=['{}'], name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.variable_name,
            IR_node.variable_name))
        self.nodes.append(IR_node.variable_name)

    def emit_Constant(self, IR_node):
        if IR_node.get_attr('value'):
            value = 'np.array({}, dtype=np.float32)'.format(IR_node.get_attr('value'))
            self.add_body(1, "{:15} = {}".format(
                IR_node.variable_name + '_value_array',
                value))
        else:
            self.add_body(1, "{:15} = _weights_dict['{}']['value']".format(
                IR_node.variable_name + '_value_array',
                IR_node.name))
        self.add_body(1, "{:15} = helper.make_node('Constant', inputs=[], outputs=['{}'], value=helper.make_tensor(name='const_tensor', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], dims={}.shape, vals={}.flatten().astype(float)), name='{}')".format(
                          IR_node.variable_name,
                          IR_node.variable_name,
                          IR_node.variable_name + '_value_array',
                          IR_node.variable_name + '_value_array',
                          IR_node.variable_name + '_value_array',
                          IR_node.variable_name))
        self.nodes.append(IR_node.variable_name)

    def emit_Sub(self, IR_node):
        inputs = ', '.join("'" + self.IR_graph.get_node(i).real_variable_name + "'" for i in IR_node.in_edges)
        self.add_body(1, "{:15} = helper.make_node('Sub', inputs=[{}], outputs=['{}'], broadcast=1, name='{}')".format(
            IR_node.variable_name,
            inputs,
            IR_node.variable_name,
            IR_node.variable_name))
        self.nodes.append(IR_node.variable_name)

    def emit_Mul(self, IR_node):
        inputs = ', '.join("'" + self.IR_graph.get_node(i).real_variable_name + "'" for i in IR_node.in_edges)
        
        if IR_node.name in self.weights_dict and 'weights' in self.weights_dict[IR_node.name]:
            self.add_body(1,"{:15} = np.array([_weights_dict['{}']['weights']])".format(
                IR_node.variable_name+'_weight_array',
                IR_node.name
            ))
            self.add_body(1, "{:15} = helper.make_node('Constant', inputs=[], outputs=['{}'], value=helper.make_tensor(name='const_tensor', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], dims={}.shape, vals={}), name='{}')".format(
                    IR_node.variable_name + '_weight',
                    IR_node.variable_name + '_weight',
                    IR_node.variable_name + '_weight_array',
                    IR_node.variable_name + '_weight_array',
                    IR_node.variable_name + '_weight_array',
                    IR_node.variable_name + '_weight'
                    ))
            inputs += ', '+''.join("'"+IR_node.variable_name +"_weight'")
            self.nodes.append(IR_node.variable_name+'_weight')

        self.add_body(1, "{:15} = helper.make_node('Mul', inputs=[{}], outputs=['{}'], broadcast=1, name='{}')".format(
            IR_node.variable_name,
            inputs,
            IR_node.variable_name,
            IR_node.variable_name))
        self.nodes.append(IR_node.variable_name)

    def emit_Dropout(self, IR_node):
        self.add_body(1, "{:15} = helper.make_node('Dropout', inputs=['{}'], outputs=['{}'], is_test={}, ratio={}, name='{}')".format(
                          IR_node.variable_name,
                          self.parent_variable_name(IR_node),
                          IR_node.variable_name,
                          0 if self.phase == 'train' else 1,
                          1 - IR_node.get_attr('keep_prob'),
                          IR_node.variable_name))
        self.nodes.append(IR_node.variable_name)

    def emit_Squeeze(self, IR_node):
        IR_node.real_name = self.IR_graph.get_node(IR_node.in_edges[0]).real_name

    def emit_ReduceMean(self, IR_node):
        axes = IR_node.layer.attr['axes'].list.i[:]
        axes = ','.join('%s' % OnnxEmitter.transpose_map[i] for i in axes)
        self.add_body(1, "{:15} = helper.make_node('ReduceMean', inputs=['{}'], outputs=['{}'], axes=[{}], keepdims={}, name='{}')".format(
                          IR_node.variable_name,
                          self.parent_variable_name(IR_node),
                          IR_node.variable_name,
                          axes,
                          1 if IR_node.layer.attr['keepdims'].b else 0,
                          IR_node.variable_name))
        self.nodes.append(IR_node.variable_name)

    def emit_Reshape(self, IR_node):
        shape = [item if item != -1 else 1 for item in IR_node.get_attr('shape')]
        if len(shape) == 4:
            shape = [shape[i] for i in [0, 3, 1, 2]]
        shape_str = ', '.join('%s' % i for i in shape)
        self.add_body(1, "{:15} = np.array([{}], dtype=np.int64)".format(
            IR_node.variable_name + '_shape_array',
            shape_str
        ))
        self.add_body(1, "{:15} = helper.make_node('Constant', inputs=[], outputs=['{}'], value=helper.make_tensor(name='const_tensor', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[{}.dtype], dims={}.shape, vals={}), name='{}')".format(
                          IR_node.variable_name + '_shape',
                          IR_node.variable_name + '_shape',
                          IR_node.variable_name + '_shape_array',
                          IR_node.variable_name + '_shape_array',
                          IR_node.variable_name + '_shape_array',
                          IR_node.variable_name + '_shape'))
        self.add_body(1, "{:15} = helper.make_node('Reshape', inputs=['{}', '{}'], outputs=['{}'], name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.variable_name + '_shape',
            IR_node.variable_name,
            IR_node.variable_name))
        self.nodes.append(IR_node.variable_name + '_shape')
        self.nodes.append(IR_node.variable_name)

    def emit_LRN(self, IR_node):
        output_name = IR_node.variable_name
        input_name = self.parent_variable_name(IR_node)
        IR_name = IR_node.name
        alpha = IR_node.get_attr('alpha')
        beta = IR_node.get_attr('beta')
        bias = IR_node.get_attr('bias', 1.0)
        size = IR_node.get_attr('size')

        self.add_body(1, "{:15} = helper.make_node('LRN', inputs=['{}'], outputs=['{}'], alpha={}, beta={}, bias={}, size={}, name='{}')".format(
                          output_name,
                          input_name,
                          output_name,
                          alpha,
                          beta,
                          bias,
                          size,
                          IR_name))
        self.nodes.append(IR_node.variable_name)

    def emit_Relu6(self, IR_node):
        self.add_body(1, "{:15} = helper.make_node('Clip', inputs=['{}'], outputs=['{}'], min=0.0, max=6.0, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.variable_name,
            IR_node.variable_name))
        self.nodes.append(IR_node.variable_name)

    def emit_DepthwiseConv(self, IR_node):
        self.emit_Conv(IR_node)

    def emit_Slice(self, IR_node):
        if self.IR_graph.get_parent(IR_node.name, [0]).type == 'Shape':
            pass
        else:
            starts = IR_node.get_attr('starts')
            starts = [starts[0], starts[-1]] + starts[1:-1]
            ends = IR_node.get_attr('ends')
            ends = [ends[0], ends[-1]] + ends[1:-1]
            ends = [i if i != 0 else sys.maxsize for i in ends]
            self.add_body(1, "{:15} = helper.make_node('Slice', inputs=['{}'], outputs=['{}'], starts={}, ends={}, name='{}')".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                IR_node.variable_name,
                starts,
                ends,
                IR_node.variable_name))
            self.nodes.append(IR_node.variable_name)

    def emit_LeakyRelu(self, IR_node):
        alpha = IR_node.get_attr('alpha')
        self.add_body(1, "{:15} = helper.make_node('LeakyRelu', inputs=['{}'], outputs=['{}'], alpha={}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.variable_name,
            alpha,
            IR_node.variable_name))
        self.nodes.append(IR_node.variable_name)

    def emit_PRelu(self, IR_node):
      slope = IR_node.get_attr('gamma')
      self.add_body(1, "{:15} = helper.make_node('PRelu', inputs=['{}'], outputs=['{}'], slope={}, name='{}')".format(
          IR_node.variable_name,
          self.parent_variable_name(IR_node),
          IR_node.variable_name,
          slope,
          IR_node.variable_name))
      self.nodes.append(IR_node.variable_name)

    def emit_SpaceToDepth(self, IR_node):
        blocksize = IR_node.get_attr('blocksize')
        self.add_body(1, "{:15} = helper.make_node('SpaceToDepth', inputs=['{}'], outputs=['{}'], blocksize={}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.variable_name,
            blocksize,
            IR_node.variable_name))
        self.nodes.append(IR_node.variable_name)

    def emit_Sigmoid(self, IR_node):
        self.add_body(1, "{: <15} = helper.make_node('Sigmoid', inputs=['{}'], outputs=['{}'], name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.variable_name,
            IR_node.variable_name
        ))
        self.nodes.append(IR_node.variable_name)

    def emit_UNKNOWN(self, IR_node):
        print(IR_node.IR_layer.name)
