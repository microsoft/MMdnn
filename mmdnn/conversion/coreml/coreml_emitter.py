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
            input_features.append((str(input_node), shape))
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

            output_features.append((str(output_node), shape))
            print("CoreML Model Output Layer: [{}] {}".format(output_node, shape))

        return list(input_features), list(output_features)

    def _connect_coreml_layers(self):
        for layer in self.builder.nn_spec.layers:
            # for i, in_node in enumerate(layer.input):
            #     layer.input[i] = self.IR_graph.get_node(in_node).real_name

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
        # assert False
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
                print("CoreMLEmitter has not supported operator [%s]." % (node_type))
                self.emit_UNKNOWN(current_node)
                assert False

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
        self.builder.set_pre_processing_parameters(
            image_input_names=[input_features[0][0]],
            #image_input_names,
            is_bgr=is_bgr,
            red_bias=red_bias,
            green_bias=green_bias,
            blue_bias=blue_bias,
            gray_bias=gray_bias,
            image_scale=image_scale)

        # Return the protobuf spec
        # model = _MLModel(self.builder.spec)

        print (self.builder.spec.description)

        return self.builder.spec, input_features, output_features


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

    def _emit_merge(self, IR_node, func):
        """
        Convert concat layer to coreml.
        """
        # Get input and output names
        input_names = [self.IR_graph.get_node(inp).real_name for inp in IR_node.in_edges]

        self.builder.add_elementwise(name=IR_node.name, input_names=input_names,
            output_name=IR_node.name, mode=func)

    def emit_Conv(self, IR_node):
        """
        Convert convolution layer to coreml.
        """
        has_bias = IR_node.get_attr('use_bias', False)
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

        input_name = self.IR_graph.get_parent(IR_node.name, [0]).real_name
        # print(self.IR_graph.get_parent(IR_node.name, [0]).layer)
        # print(input_name)
        # print(IR_node.real_name)

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
                                     output_name=IR_node.real_name,
                                     dilation_factors=dilations)


    def emit_DepthwiseConv(self, IR_node):
        # depth-wise convolution

        input_name = self.IR_graph.get_parent(IR_node.name, [0]).real_name
        kernel_channels = 1
        is_deconv = False
        has_bias = IR_node.get_attr('use_bias', False)

        depth_multiplier = IR_node.get_attr('kernel_shape')[-1]

        W = self.weights_dict[IR_node.name]['weights']
        height, width, channels, n_filters = W.shape
        output_shape = None
        W = np.reshape(W,(height, width,1,channels * depth_multiplier))
        b = self.weights_dict[IR_node.name]['bias'] if has_bias else None

        # Dilations
        dilations = IR_node.get_attr('dilations', [1, 1])

        padding = self._get_padding(IR_node).lower()
        output_channels = W.shape[-1]
        groups = W.shape[-1]
        stride_height, stride_width = IR_node.get_attr('strides')[1], IR_node.get_attr('strides')[2]

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
                                     output_name=IR_node.real_name,
                                     dilation_factors=dilations)


    def emit_Pool(self, IR_node):
        """
        Convert pooling layer to coreml.
        """
        # Get input and output names
        input_name = self.IR_graph.get_node(IR_node.in_edges[0]).real_name

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
            padding_type = self._get_padding(IR_node)

        self.builder.add_pooling(name=IR_node.name,
                                    height=height,
                                    width=width,
                                    stride_height=stride_height,
                                    stride_width=stride_width,
                                    layer_type=layer_type_str,
                                    padding_type=padding_type,
                                    input_name=input_name,
                                    output_name=IR_node.name,
                                    exclude_pad_area=True,
                                    is_global=global_pooling)


    def emit_UNKNOWN(self, IR_node):
        print(IR_node.name)


    def emit_Crop(self, IR_node):
        input_name = self.IR_graph.get_parent(IR_node.name, [0]).real_name
        output_name=IR_node.real_name

        is_1d = False
        border = IR_node.get_attr('border')

        if is_1d:
            raise ValueError("Unrecognized padding option: %s" % (str(border)))
        else:
            if type(border) is int:
                top = left = bottom = right = border
            elif type(border) is list:
                top, left = border[1], border [0]
                bottom, right = border[2], border [3]
            else:
                raise ValueError("Unrecognized padding option: %s" % (str(border)))

        # Now add the layer
        self.builder.add_crop(name = IR_node.name,
            left = left, right=right, top=top, bottom=bottom, offset = [0,0],
            input_names = [input_name], output_name=output_name
            )
        # assert False



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
                                       output_name=IR_node.name)


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
                                 input_name=input_name, output_name=IR_node.name)


    def emit_Reshape(self, IR_node):
        def ShapetrToTuple(string, batch_none = False):
            if batch_none == True:
                ls = [int(item) for item in string.split(', ')]
                ls.insert(0,None)
                return tuple(ls)
            else:
                ls = [int(item) for item in string.split(', ')]
                return tuple(ls)

        last_node = self.IR_graph.get_node(IR_node.in_edges[0]).layer
        input_shape_dims = last_node.attr["_output_shapes"].list.shape
        target_shape_dims = IR_node.IR_layer.attr["_output_shapes"].list.shape

        input_shape = ShapetrToTuple(IRGraph.shapeToStr(input_shape_dims[0]),True)
        target_shape = ShapetrToTuple(IRGraph.shapeToStr(target_shape_dims[0]))

        def get_coreml_target_shape(target_shape):
            if len(target_shape) == 1: #(D,)
                coreml_shape = (1,target_shape[0],1,1)
            elif len(target_shape) == 2: #(S,D)
                coreml_shape = target_shape + (1,1)
            elif len(target_shape) == 3: #(H,W,C)
                coreml_shape = (1, target_shape[2], target_shape[0], target_shape[1])
            else:
                coreml_shape = None
            return coreml_shape

        def get_mode(input_shape, target_shape):
            in_shape = input_shape[1:]
            if len(in_shape) == 3 or len(target_shape) == 3:
                    return 1
            else:
                return 0
        input_name = self.IR_graph.get_node(IR_node.in_edges[0]).real_name
        new_shape = get_coreml_target_shape(target_shape)
        mode = get_mode(input_shape, target_shape)

        self.builder.add_reshape(
            name=IR_node.real_name,
            input_name=input_name,
            output_name=IR_node.real_name,
            target_shape=new_shape,
            mode=mode)



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
        output_name = IR_node.real_name
        self.builder.add_activation(name=IR_node.real_name,
            non_linearity=act,
            input_name=input_name,
            output_name=output_name,
            params=params)


    def emit_Relu(self, IR_node):
        self._emit_activation(IR_node, 'RELU')

    def emit_PRelu(self, IR_node):
        self._emit_activation(IR_node, 'PRELU', self.weights_dict[IR_node.name]['gamma'])


    def emit_Softmax(self, IR_node):
        # Get input and output names
        input_name = self.IR_graph.get_node(IR_node.in_edges[0]).real_name
        output_name = IR_node.out_edges[0]
        self.builder.add_softmax(name=IR_node.name, input_name=input_name,
                                 output_name=IR_node.name)


    def emit_Sigmoid(self, IR_node):
        assert False
        code = "{:<15} = Activation(name = '{}', activation = 'sigmoid')({})".format(
                IR_node.replace_scope(IR_node.name),
                IR_node.name,
                IR_node.replace_scope(IR_node.in_edges[0]))
        return code


    def emit_Relu6(self, IR_node):
        # print(IR_node.name)
        layer = IR_node.real_name
        input_name, output_name = (IR_node.IR_layer.input[0], IR_node.IR_layer.name)
        # input_name =
        relu_output_name = output_name + '_relu'
        self.builder.add_activation(layer, 'RELU', input_name, relu_output_name)
        # negate it
        neg_output_name = relu_output_name + '_neg'
        self.builder.add_activation(layer+'__neg__', 'LINEAR', relu_output_name,
                neg_output_name,[-1.0, 0])
        # apply threshold
        clip_output_name = relu_output_name + '_clip'
        self.builder.add_unary(layer+'__clip__', neg_output_name, clip_output_name,
                'threshold', alpha = -6.0)
        # negate it back
        self.builder.add_activation(
            layer + '_neg2',
            'LINEAR',
            clip_output_name,
            output_name,
            [-1.0, 0])

    def emit_Gather(self, IR_node):
        raise NotImplementedError()
        W = self.weights_dict[IR_node.name]['weights']
        if W.ndim == 2:
            vocab_size = W.shape[0]
            output_channels = W.shape[1]
            builder.add_embedding(
                name=IR_node.real_name,
                W = W,
                b = None,
                input_dim = vocab_size,
                output_channels = output_channels,
                has_bias=False,
                input_name=input_name,
                output_name=IR_node.real_name)
        else:
            raise NotImplementedError()

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
        self._emit_merge(IR_node, 'ADD')


    def emit_Concat(self, IR_node):
        self._emit_merge(IR_node, "CONCAT")


    def emit_BatchNorm(self, IR_node):
        """
        Convert a Batch Normalization layer.
        """

        # Get input and output names
        input_name = self.IR_graph.get_parent(IR_node.name, [0]).real_name
        # print(input_name)
        # print(IR_node.real_name)
        axis = IR_node.get_attr('axis', -1)
        nb_channels = IR_node.get_attr('_output_shapes')[0].dim[axis].size

        # Set parameters
        # Parameter arrangement in Keras: gamma, beta, mean, variance
        weights = self.weights_dict[IR_node.name]
        mean = weights['mean']
        std = weights['var']
        gamma = weights.get('scale', np.ones(mean.shape))
        beta = weights.get('bias', np.zeros(mean.shape))

        # compute adjusted parameters
        variance = std * std
        f = 1.0 / np.sqrt(std + IR_node.get_attr('epsilon'))
        gamma1 = gamma*f
        beta1 = beta - gamma*mean*f
        mean[:] = 0.0 #mean
        variance[:] = 1.0 - .00001 #stddev
        self.builder.add_batchnorm(
            name=IR_node.real_name,
            channels = nb_channels,
            gamma = gamma1,
            beta = beta1,
            mean = mean,
            variance = variance,
            input_name = input_name,
            output_name=IR_node.real_name)
        # assert False


    def emit_Pad(self, IR_node):
        input_name = self.IR_graph.get_parent(IR_node.name, [0]).real_name
        output_name=IR_node.real_name
        is_1d = False
        padding = IR_node.get_attr('pads')

        if is_1d:
            raise ValueError("Unrecognized padding option: %s" % (str(padding)))
        else:
            if type(padding) is int:
                top = left = bottom = right = padding
            elif type(padding) is list:
                top, left = padding[1], padding [2]
                bottom, right = padding[5], padding [6]
            else:
                raise ValueError("Unrecognized padding option: %s" % (str(padding)))

        # Now add the layer
        self.builder.add_padding(name = IR_node.name,
            left = left, right=right, top=top, bottom=bottom, value = 0,
            input_name = input_name, output_name=output_name
            )


    def emit_Squeeze(self, IR_node):
        self.emit_Flatten(IR_node)
        # if IR_node.name != "MMdnn_Output" :
            # self.emit_Flatten(IR_node)
            # self.emit_Reshape(IR_node)


    def emit_LRN(self, IR_node):
        input_name = self.IR_graph.get_parent(IR_node.name, [0]).real_name
        output_name = IR_node.real_name
        C = IR_node.get_attr('size')
        alpha = IR_node.get_attr('alpha')
        beta = IR_node.get_attr('beta')
        k = IR_node.get_attr('k')
        depth_radius = int(IR_node.get_attr('size'))
        self.builder.add_lrn(output_name, input_name, output_name,
                          alpha=alpha * C,
                          beta=beta,
                          local_size=depth_radius,
                          k=k)


    def emit_SeparableConv(self, IR_node):

        input_name = self.IR_graph.get_parent(IR_node.name, [0]).real_name
        output_name = IR_node.real_name

        assert len(IR_node.get_attr("strides")) == 4
        strides = IR_node.get_attr('strides')
        stride_height, stride_width = (strides[1], strides[2])

        # Get the weights
        W0 = self.weights_dict[IR_node.name]['depthwise_filter']
        W1 = self.weights_dict[IR_node.name]['pointwise_filter']

        padding = IR_node.get_attr('auto_pad').split('_')[0].lower()
        has_bias = IR_node.get_attr('use_bias')
        b = self.weights_dict[IR_node.name]['bias'] if has_bias else None

        output_blob_shape = IR_node.get_attr('_output_shapes')
        shape = shape_to_list(output_blob_shape[0])
        output_channels = shape[-1]

        height, width, input_channels, depth_mult = W0.shape

        W0 = np.reshape(W0, (height, width, 1, input_channels * depth_mult))

        intermediate_name = input_name + '_intermin_'

        self.builder.add_convolution(name = IR_node.name + '_step_1',
             kernel_channels = 1,
             output_channels = input_channels * depth_mult,
             height = height,
             width = width,
             stride_height = stride_height,
             stride_width = stride_width,
             border_mode = padding,
             groups = input_channels,
             W = W0,
             b = None,
             has_bias = False,
             is_deconv = False,
             output_shape = None,
             input_name = input_name,
             output_name = intermediate_name,
             dilation_factors = [1,1])

        self.builder.add_convolution(name = IR_node.name + '_step_2',
                kernel_channels = input_channels * depth_mult,
                output_channels = output_channels,
                height = 1,
                width = 1,
                stride_height = 1,
                stride_width = 1,
                border_mode = padding,
                groups = 1,
                W = W1,
                b = b,
                has_bias = has_bias,
                is_deconv = False,
                output_shape = None,
                input_name = intermediate_name,
                output_name = output_name,
                dilation_factors = [1,1])

