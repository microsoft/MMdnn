#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import os

import os
import numpy as np
from six import string_types as _string_types
from mmdnn.conversion.common.IR.IR_graph import IRGraph, IRGraphNode
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.DataStructure.emitter import Emitter
from mmdnn.conversion.common.utils import *
from mmdnn.conversion.coreml.coreml_utils import _infer_coreml_input_shape

from coremltools.models.neural_network import NeuralNetworkBuilder as _NeuralNetworkBuilder
from coremltools.models import datatypes
from coremltools.models import MLModel as _MLModel
from coremltools.models.utils import save_spec as _save_spec




class CoreMLEmitter(Emitter):

    dtype_map = {
        graph_pb2.DT_FLOAT16 : "np.float16",
        graph_pb2.DT_FLOAT32 : "np.float32",
        graph_pb2.DT_FLOAT64 : "np.float64",
        graph_pb2.DT_INT16 : "np.int16",
        graph_pb2.DT_INT32 : "np.int32",
        graph_pb2.DT_INT64 : "np.int64",
        graph_pb2.DT_UINT8 : "np.uint8",
        graph_pb2.DT_UINT16 : "np.uint16"
        }


    def __init__(self, architecture, weight):
        super(CoreMLEmitter, self).__init__()
        if os.path.exists(architecture) == False:
            raise ValueError("IR architecture file [{}] is not found.".format(architecture))
        else:
            self.IR_graph = IRGraph(architecture)
            self.IR_graph.build()

        if os.path.exists(weight) == False:
            raise ValueError("IR weight file [{}] is not found.".format(weight))
        else:
            self._load_weights(weight)


    def _get_inout(self):
        input_features = []
        output_features = []
        for input_node in self.IR_graph.input_layers:
            shape = shape_to_list(self.IR_graph.get_node(input_node).get_attr('shape'))
            shape = _infer_coreml_input_shape(shape)
            input_features.append((input_node.encode(), shape))
            print ("CoreML Model Input Layer: [{}] {}".format(input_node, shape))

        for output_node in self.IR_graph.output_layers:
            print (self.IR_graph.get_node(output_node).layer)
            shape = self.IR_graph.get_node(output_node).get_attr('_output_shapes', [[1]])[0]
            print(shape)
            shape = shape_to_list(shape)
            shape = _infer_coreml_input_shape(shape)
            output_features.append((output_node.encode(), shape))
            print ("CoreML Model Input Layer: [{}] {}".format(output_node, shape))

        return list(input_features), list(output_features)


    def gen_model(self,
                  input_names=None,
                  output_names=None,
                  image_input_names=None,
                  is_bgr=False,
                  red_bias=0.0,
                  green_bias=0.0,
                  blue_bias=0.0,
                  gray_bias=0.0,
                  image_scale=1.0,
                  class_labels=None,
                  predicted_feature_name=None,
                  predicted_probabilities_output=''):

        input_features, output_features = self._get_inout()
        is_classifier = class_labels is not None
        mode = 'classifier' if is_classifier else None
        self.builder = _NeuralNetworkBuilder(input_features, output_features, mode=mode)

        self.builder = self._get_builder()

        for layer in self.IR_graph.topological_sort:
            current_node = self.IR_graph.get_node(layer)
            print ("Converting the layer [{}]".format(current_node.name))
            node_type = current_node.type

            if hasattr(self, "emit_" + node_type):
                func = getattr(self, "emit_" + node_type)
                func(current_node)
            else:
                print("CntkEmitter has not supported operator [%s]." % (node_type))
                self.emit_UNKNOWN(current_node)

        '''
        # Set the right inputs and outputs on the model description (interface)
        builder.set_input(input_names, input_dims)
        builder.set_output(output_names, output_dims)

        # Since we aren't mangling anything the user gave us, we only need to update
        # the model interface here
        builder.add_optionals(graph.optional_inputs, graph.optional_outputs)

        # Add classifier classes (if applicable)
        if is_classifier:
            classes_in = class_labels
            if isinstance(classes_in, _string_types):
                import os
                if not os.path.isfile(classes_in):
                    raise ValueError("Path to class labels (%s) does not exist." % classes_in)
                with open(classes_in, 'r') as f:
                    classes = f.read()
                classes = classes.splitlines()
            elif type(classes_in) is list: # list[int or str]
                classes = classes_in
            else:
                raise ValueError('Class labels must be a list of integers / strings, or a file path')

            if predicted_feature_name is not None:
                builder.set_class_labels(classes, predicted_feature_name = predicted_feature_name,
                                        prediction_blob = predicted_probabilities_output)
            else:
                builder.set_class_labels(classes)

        # Set pre-processing paramsters
        builder.set_pre_processing_parameters(image_input_names = image_input_names,
                                            is_bgr = is_bgr,
                                            red_bias = red_bias,
                                            green_bias = green_bias,
                                            blue_bias = blue_bias,
                                            gray_bias = gray_bias,
                                            image_scale = image_scale)

        '''

        return _MLModel(self.builder.spec)


    @staticmethod
    def _emit_merge(IR_node, func):
        assert False
        inputs = listToStr(IR_node.in_edges)
        code = "{:<15} = layers.{}(name = '{}', inputs = [{}])".format(
                IR_node.name,
                func,
                IR_node.name,
                inputs)
        return code


    def emit_Convolution(self, IR_node):
        strides = IR_node.IR_layer.attr["strides"].list.i[1:-1]

        padding = IR_node.IR_layer.attr["padding"].s
        padding = padding.lower()

        weight_dict = self.weights[IR_node.name]

        # Get input and output names
        input_name, output_name = self._get_in_out_names(IR_node)

        # Get weights
        # Dimensions and weights
        W = weight_dict['weights']
        height, width, channels, n_filters = W.shape

        # Bias
        has_bias = IR_node.IR_layer.attr['use_bias'].b
        b = weight_dict['bias'] if has_bias else None

        stride_height, stride_width = strides

        # Dilations
        dilations = [1, 1]
        if len(IR_node.IR_layer.attr["dilation_rate"].list.i) > 0:
            dilations = IR_node.IR_layer.attr["dilation_rate"].list.i[1:-1]
        else:
            dilations = [1, 1]

        self.builder.add_convolution(name = IR_node.name,
                kernel_channels = channels,
                output_channels = n_filters,
                height = height,
                width = width,
                stride_height = stride_height,
                stride_width = stride_width,
                border_mode = padding,
                groups = 1,
                W = W,
                b = b,
                has_bias = has_bias,
                is_deconv = False,
                output_shape = None,
                input_name = input_name,
                output_name = output_name,
                dilation_factors = dilations)


    def emit_Pool(self, IR_node):
        """
        Convert pooling layer from keras to coreml.

        Parameters
        ----------
        keras_layer: layer
            A keras layer object.

        builder: NeuralNetworkBuilder
            A neural network builder object.
        """
        # Get input and output names
        input_name = self.IR_graph.layer_name_map[IR_node.in_edges[0]]
        output_name = self.IR_graph.layer_name_map[IR_node.out_edges[0]]

        # Pooling layer type
        if IR_node.layer.attr['pooling_type'].s == b'MAX':
            layer_type_str = 'MAX'
        elif IR_node.layer.attr['pooling_type'].s == b'AVG':
            layer_type_str = 'AVERAGE'
        else:
            assert False

        dim = len(IR_node.layer.attr['strides'].list.i) - 2
        # if it's global, set the global flag
        if IR_node.layer.attr['global_pooling'].b:
            if dim == 2:
                # 2D global pooling
                global_pooling = True
                height, width = (0, 0)
                stride_height, stride_width = (0,0)
                padding_type = 'VALID'
            else:
                assert dim == 1
                # 1D global pooling: 1D global pooling seems problematic in the backend,
                # use this work-around
                global_pooling = False
                _, width, channels = keras_layer.input_shape
                height = 1
                stride_height, stride_width = height, width
                padding_type = 'VALID'
        else:
            global_pooling = False
            # Set pool sizes and strides
            # 1D cases:
            if dim == 1:
                pool_size = IR_node.IR_layer.attr['window_shape'].list.i[1]
                height, width = 1, pool_size
                stride_height, stride_width = 1, IR_node.IR_layer.attr['strides'].list.i[1]
            # 2D cases:
            else:
                assert dim == 2
                height, width = IR_node.IR_layer.attr['window_shape'].list.i[1:-1]
                stride_height, stride_width = IR_node.IR_layer.attr['strides'].list.i[1:-1]

            # Padding
            padding_type = IR_node.IR_layer.attr["padding"].s

        self.builder.add_pooling(name = IR_node.name,
            height = height,
            width = width,
            stride_height = stride_height,
            stride_width = stride_width,
            layer_type = layer_type_str,
            padding_type = padding_type,
            input_name = input_name,
            output_name = output_name,
            exclude_pad_area = True,
            is_global = global_pooling)


    def emit_UNKNOWN(self, IR_node):
        print(IR_node.IR_layer.name)


    def emit_DataInput(self, IR_node):
        """ Layers that can be skipped (because they are train time only. """
        return


    def emit_Dropout(self, IR_node):
        """ Layers that can be skipped (because they are train time only. """
        self.IR_graph.layer_name_map[IR_node.name] = self.IR_graph.layer_name_map[IR_node.in_edges[0]]


    def emit_FullyConnected(self, IR_node):
        assert False
        units = IR_node.IR_layer.attr["units"].i
        use_bias = IR_node.IR_layer.attr["use_bias"].b

        ret = "{:<15} = Dense(name = '{}', units = {}, use_bias = {})({})".format(
                IR_node.replace_scope(IR_node.name),
                IR_node.name,
                units,
                use_bias,
                IR_node.replace_scope(IR_node.in_edges[0]))

        return ret


    def emit_Flatten(self, IR_node):
        """
        Convert a flatten layer from keras to coreml.
        ----------
        Parameters
        keras_layer: layer
            A keras layer object.

        builder: NeuralNetworkBuilder
            A neural network builder object.
        """
        input_name, output_name = self._get_in_out_names(IR_node)

        # blob_order == 0 if the input blob needs not be rearranged
        # blob_order == 1 if the input blob needs to be rearranged
        # Kit TODO: set blob_order
        blob_order = 1
        '''
        # using keras_layer.input.shape have a "?" (Dimension[None] at the front),
        # making a 3D tensor with unknown batch size 4D
        if len(keras_layer.input.shape) == 4:
            blob_order = 1
        '''

        self.builder.add_flatten(name = IR_node.name, mode = blob_order, input_name = input_name, output_name = output_name)


    def emit_Reshape(self, IR_node):
        assert False
        shape_str = IRGraph.shapeToStr(IR_node.IR_layer.attr["shape"].shape, True)
        code = "{:<15} = Reshape(name = \"{}\", target_shape = ({}))({})".format(
            IR_node.replace_scope(IR_node.name),
            IR_node.name,
            shape_str,
            IR_node.replace_scope(IR_node.in_edges[0]))
        return code


    def emit_Tanh(self, IR_node):
        assert False
        code = "{:<15} = Activation(name = '{}', activation = tanh)({})".format(
                IR_node.replace_scope(IR_node.name),
                IR_node.name,
                IR_node.replace_scope(IR_node.in_edges[0]))
        return code


    def emit_Relu(self, IR_node):
        input_name, output_name = self._get_in_out_names(IR_node)
        self.builder.add_activation(name = IR_node.name,
            non_linearity = "RELU",
            input_name = input_name, output_name = output_name,
            params = None)


    def emit_Softmax(self, IR_node):
        assert False
        code = "{:<15} = Activation(name = '{}', activation = 'softmax')({})".format(
                IR_node.replace_scope(IR_node.name),
                IR_node.name,
                IR_node.replace_scope(IR_node.in_edges[0]))
        return code


    def emit_Sigmoid(self, IR_node):
        assert False
        code = "{:<15} = Activation(name = '{}', activation = 'sigmoid')({})".format(
                IR_node.replace_scope(IR_node.name),
                IR_node.name,
                IR_node.replace_scope(IR_node.in_edges[0]))
        return code


    def emit_Embedding(self, IR_node):
        assert False
        ret = "{:<15} = Embedding(input_dim = {}, output_dim = {}, mask_zero = {})({})".format(
                IR_node.name,
                IR_node.IR_layer.attr['input_dim'].i,
                IR_node.IR_layer.attr['output_dim'].i,
                IR_node.IR_layer.attr['mask_zero'].b,
                IR_node.in_edges[0])

        return ret


    def emit_RNNs(self, IR_node, func):
        assert False
        # for Keras
        if "dropout" in IR_node.IR_layer.attr:
            dropout_str = ",dropout = {}, recurrent_dropout = {}".format(
                    IR_node.IR_layer.attr['dropout'].f,
                    IR_node.IR_layer.attr['recurrent_dropout'].f)
        else:
            dropout_str = ""

        code = "{:<15} = {}(units = {}, use_bias = {} {})({})".format(
                IR_node.name,
                func,
                IR_node.IR_layer.attr['units'].i,
                IR_node.IR_layer.attr['use_bias'].b,
                dropout_str,
                IR_node.in_edges[0])

        return code


    def emit_LSTM(self, IR_node):
        return self.emit_RNNs(IR_node, "LSTM")


    def emit_GRU(self, IR_node):
        return self.emit_RNNs(IR_node, "GRU")


    def emit_Add(self, IR_node):
        assert False
        code = Keras2Emitter._emit_merge(IR_node, "add")
        return code


    def emit_Concat(self, IR_node):
        assert False
        code = Keras2Emitter._emit_merge(IR_node, "concatenate")
        return code


    def emit_BatchNorm(self, IR_node):
        assert False
        code = "{:<15} = BatchNormalization(name = '{}', axis = {}, center = {}, scale = {})({})".format(
                IR_node.replace_scope(IR_node.name),
                IR_node.name,
                IR_node.IR_layer.attr['axis'].i,
                IR_node.IR_layer.attr['center'].b,
                IR_node.IR_layer.attr['scale' ].b,
                IR_node.replace_scope(IR_node.in_edges[0]))
        return code


    def emit_pad(self, IR_node):
        assert False
        if IR_node.IR_layer.attr['mode'].s == "CONSTANT":
            func = "ZeroPadding"

        dim = len(IR_node.IR_layer.attr['padding'].list.i) // 2

        padding_str = ""
        for idx in range(0, dim):
            padding_str += "({}, {}),".format(
                    IR_node.IR_layer.attr['padding'].list.i[idx + idx],
                    IR_node.IR_layer.attr['padding'].list.i[idx + idx + 1])

        code = "{:<15} = {}{}D(name = \"{}\", padding = ({}))({})".format(
                IR_node.replace_scope(IR_node.name),
                func,
                dim,
                IR_node.name,
                padding_str,
                IR_node.replace_scope(IR_node.in_edges[0]))

        return code


    def emit_Squeeze(self, IR_node):
        self.emit_Flatten(IR_node)