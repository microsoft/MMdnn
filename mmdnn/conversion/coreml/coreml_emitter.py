#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

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
            print("CoreML Model Input Layer: [{}] {}".format(input_node, shape))

        for output_node in self.IR_graph.output_layers:
            node = self.IR_graph.get_node(output_node)
            node.out_edges.append(node.name)
            shape = node.get_attr('_output_shapes')
            if shape:
                shape = shape_to_list(shape[0])
            else:
                shape = [1]
            shape = _infer_coreml_input_shape(shape)

            output_features.append((output_node.encode(), shape))
            print("CoreML Model Output Layer: [{}] {}".format(output_node, shape))

        return list(input_features), list(output_features)

    def _connect_coreml_layers(self):
        for layer in self.builder.nn_spec.layers:
            for i, out_node in enumerate(layer.output):
                layer.output[i] = self.IR_graph.get_node(out_node).real_name

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

        for layer in self.IR_graph.topological_sort:
            current_node = self.IR_graph.get_node(layer)
            print("Converting layer {}({})".format(current_node.name, current_node.type))
            node_type = current_node.type

            if hasattr(self, "emit_" + node_type):
                func = getattr(self, "emit_" + node_type)
                func(current_node)
            else:
                print("CntkEmitter has not supported operator [%s]." % (node_type))
                self.emit_UNKNOWN(current_node)

            self._connect_coreml_layers()
        # Add classifier classes (if applicable)
        if is_classifier:
            classes_in = class_labels
            if isinstance(classes_in, _string_types):
                if not os.path.isfile(classes_in):
                    raise ValueError("Path to class labels [{}] does not exist.".format(classes_in))
                with open(classes_in, 'r') as f:
                    classes = f.read()
                classes = classes.splitlines()
            elif type(classes_in) is list: # list[int or str]
                classes = classes_in
            else:
                raise ValueError('Class labels must be a list of integers / strings, or a file path')

            if predicted_feature_name is not None:
                self.builder.set_class_labels(classes, predicted_feature_name = predicted_feature_name,
                                        prediction_blob = predicted_probabilities_output)
            else:
                self.builder.set_class_labels(classes)

        # Set pre-processing paramsters
        self.builder.set_pre_processing_parameters(image_input_names=image_input_names,
                                            is_bgr=is_bgr,
                                            red_bias=red_bias,
                                            green_bias=green_bias,
                                            blue_bias=blue_bias,
                                            gray_bias=gray_bias,
                                            image_scale=image_scale)

        # Return the protobuf spec
        return _MLModel(self.builder.spec)


    @staticmethod
    def _get_padding(IR_node):
        auto_pads = IR_node.get_attr('auto_pads')
        if auto_pads is not None:
            if auto_pads == 'VALID':
                return auto_pads
            else:
                return 'SAME'

        pads = IR_node.get_attr('pads')
        if is_valid_padding(pads):
            return 'VALID'
        else:
            return 'SAME'

    def _emit_merge(IR_node, func):
        assert False
        inputs = listToStr(IR_node.in_edges)
        code = "{:<15} = layers.{}(name = '{}', inputs = [{}])".format(
                IR_node.name,
                func,
                IR_node.name,
                inputs)
        return code


    def emit_Conv(self, IR_node):
        """
        Convert convolution layer to coreml.
        """
        # Get input and output names
        input_name = self.IR_graph.get_node(IR_node.in_edges[0]).real_name
        output_name = IR_node.out_edges[0]

        has_bias = IR_node.get_attr('use_bias')
        is_deconv = False # TODO: Deconv

        # Get the weights.
        output_channels = IR_node.get_attr('kernel_shape')[-1]

        # Dimensions and weights
        if is_deconv:
            raise NotImplementedError()
            height, width, n_filters, channels = weightList[0].shape
            W = weightList[0].transpose([0,1,3,2])
            output_shape = output_blob_shape[:-1]
        else:
            W = self.weights_dict[IR_node.name]['weights']
            height, width, channels, n_filters = W.shape
            output_shape = None
        b = self.weights_dict[IR_node.name]['bias'] if has_bias else None

        stride_height, stride_width = IR_node.get_attr('strides')[1], IR_node.get_attr('strides')[2]

        # Dilations
        dilations = IR_node.get_attr('dilations', [1, 1])
        if is_deconv and not dilations == [1, 1]:
            raise ValueError("Unsupported non-unity dilation for Deconvolution layer")

        groups = IR_node.get_attr('groups', 1)
        kernel_channels = channels

        padding = self._get_padding(IR_node).lower()

        self.builder.add_convolution(name=IR_node.real_name,
                                     kernel_channels=kernel_channels,
                                     output_channels=output_channels,
                                     height=height,
                                     width=width,
                                     stride_height=stride_height,
                                     stride_width=stride_width,
                                     border_mode=padding,
                                     groups=groups,
                                     W=W,
                                     b=b,
                                     has_bias=has_bias,
                                     is_deconv=is_deconv,
                                     output_shape=output_shape,
                                     input_name=input_name,
                                     output_name=output_name,
                                     dilation_factors=dilations)


    def emit_Pool(self, IR_node):
        """
        Convert pooling layer to coreml.
        """
        # Get input and output names
        input_name = self.IR_graph.get_node(IR_node.in_edges[0]).real_name
        output_name = IR_node.out_edges[0]

        # Pooling layer type
        pooling_type = IR_node.get_attr('pooling_type')
        if pooling_type == 'MAX':
            layer_type_str = 'MAX'
        elif pooling_type == 'AVG':
            layer_type_str = 'AVERAGE'
        else:
            raise TypeError("Pooling type %s not supported" % pooling_type)

        # if it's global, set the global flag
        global_pooling = IR_node.get_attr('global_pooling', False)
        dim = len(IR_node.get_attr('strides')) - 2
        if global_pooling:
            if dim == 2:
                height, width = (0, 0)
                stride_height = stride_width = 0
                padding_type = 'VALID'
            elif dim == 1:
                raise NotImplementedError()
                global_pooling = False
                _, width, channels = keras_layer.input_shape
                height = 1
                stride_height, stride_width = height, width
                padding_type = 'VALID'
            else:
                raise NotImplementedError()

        else:
            height, width = tuple(IR_node.get_attr('kernel_shape')[1:-1])
            stride_height, stride_width = tuple(IR_node.get_attr('strides')[1:-1])

            # Padding
            padding = self._get_padding(IR_node)
            self.builder.add_pooling(name=IR_node.name,
                                     height=height,
                                     width=width,
                                     stride_height=stride_height,
                                     stride_width=stride_width,
                                     layer_type=layer_type_str,
                                     padding_type=padding,
                                     input_name=input_name,
                                     output_name=output_name,
                                     exclude_pad_area=True,
                                     is_global=global_pooling)


    def emit_UNKNOWN(self, IR_node):
        print(IR_node.name)


    def emit_DataInput(self, IR_node):
        """ Layers that can be skipped. """
        return


    def emit_Dropout(self, IR_node):
        """ Layers that can be skipped (because they are train time only. """
        IR_node.real_name = self.IR_graph.get_parent(IR_node.name, [0]).real_name


    def emit_FullyConnected(self, IR_node):
        """
        Convert a dense layer to coreml.
        """
        # Get input and output names
        input_name = self.IR_graph.get_node(IR_node.in_edges[0]).real_name
        output_name = IR_node.out_edges[0]

        has_bias = IR_node.get_attr('use_bias')

        # Get the weights from keras
        W = self.weights_dict[IR_node.name]['weights'].T
        Wb = self.weights_dict[IR_node.name]['bias'].T if has_bias else None
        output_channels, input_channels = W.shape

        self.builder.add_inner_product(name=IR_node.name,
                                       W=W,
                                       b=Wb,
                                       input_channels=input_channels,
                                       output_channels=output_channels,
                                       has_bias=has_bias,
                                       input_name=input_name,
                                       output_name=output_name)


    def emit_Flatten(self, IR_node):
        """
        Convert a flatten layer from keras to coreml.
        """
        # Get input and output names
        input_name = self.IR_graph.get_node(IR_node.in_edges[0]).real_name
        output_name = IR_node.out_edges[0]

        """
        # blob_order == 0 if the input blob needs not be rearranged
        # blob_order == 1 if the input blob needs to be rearranged
        blob_order = 0

        # using keras_layer.input.shape have a "?" (Dimension[None] at the front),
        # making a 3D tensor with unknown batch size 4D
        if len(keras_layer.input.shape) == 4:
            blob_order = 1
        """

        self.builder.add_flatten(name=IR_node.name, mode=1,
                                 input_name=input_name, output_name=output_name)


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


    def _emit_activation(self, IR_node, act, params=None):
        # Get input and output names
        input_name = self.IR_graph.get_parent(IR_node.name, [0]).real_name
        output_name = IR_node.out_edges[0]
        self.builder.add_activation(name=IR_node.name,
                                    non_linearity=act,
                                    input_name=input_name,
                                    output_name=output_name,
                                    params=params)


    def emit_Relu(self, IR_node):
        self._emit_activation(IR_node, 'RELU')


    def emit_Softmax(self, IR_node):
        # Get input and output names
        input_name = self.IR_graph.get_node(IR_node.in_edges[0]).real_name
        output_name = IR_node.out_edges[0]
        self.builder.add_softmax(name=IR_node.name, input_name=input_name,
                                 output_name=output_name)


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
