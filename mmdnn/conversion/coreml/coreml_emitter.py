#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import division

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
            if self.IR_graph.get_node(input_node).type == 'Const':
                continue
            shape = shape_to_list(self.IR_graph.get_node(input_node).get_attr('shape'))
            shape = _infer_coreml_input_shape(shape)
            input_features.append((str(input_node), shape))
            print("CoreML Model Input Layer: [{}] {}".format(input_node, shape))

        for output_node in self.IR_graph.output_layers:

            node = self.IR_graph.get_node(output_node)

            if node.type == 'Pack':
                continue

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

        auto_pad = IR_node.get_attr('auto_pad')
        if auto_pad is not None:
            if auto_pad == 'VALID':
                pass
            else:
                return 'SAME'

        pads = IR_node.get_attr('pads', [0,0,0,0,0,0,0,0])

        return pads

    def emit_Mul(self, IR_node):
        """
        Not implement yet
        """
        pass
        # if IR_node.name in self.weights_dict and 'weights' in self.weights_dict[IR_node.name]:
        #     pass
        
        # self._emit_merge(IR_node,'DOT')
        

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
        is_deconv = False


        # Dimensions and weights
        kernel_shape = IR_node.get_attr('kernel_shape')

        if len(kernel_shape) == 4:
            height, width, input_channels, output_channels = kernel_shape
        elif len(kernel_shape) == 5:
            depth, height, width, input_channels, output_channels = kernel_shape
        else:
            raise NotImplementedError()

        output_shape = None

        # W should have shape (height, width, kernel_channels, output_channels), where kernel_channel = input_channels / groups
        W = self.weights_dict[IR_node.name]['weights']
        b = self.weights_dict[IR_node.name]['bias'] if has_bias else None


        stride_height, stride_width = IR_node.get_attr('strides')[1], IR_node.get_attr('strides')[2]

        # Dilations
        dilations = IR_node.get_attr('dilations', [1, 1])
        if is_deconv and not dilations == [1, 1]:
            raise ValueError("Unsupported non-unity dilation for Deconvolution layer")

        groups = IR_node.get_attr('group', 1)

        kernel_channels = input_channels // groups
        padding = self._get_padding(IR_node)

        if isinstance(padding, list):
            border_mode = "valid"
            # see protobuf
            padding_top, padding_left, padding_bottom, padding_right = padding[1], padding [2], padding[5], padding [6]
        else:
            border_mode = "same"
            padding_top, padding_left, padding_bottom, padding_right = 0, 0, 0, 0


        input_name = self.IR_graph.get_parent(IR_node.name, [0]).real_name

        self.builder.add_convolution(name=IR_node.real_name,
                                     kernel_channels=kernel_channels,
                                     output_channels=output_channels,
                                     height=height,
                                     width=width,
                                     stride_height=stride_height,
                                     stride_width=stride_width,
                                     border_mode= border_mode,
                                     groups=groups,
                                     W=W,
                                     b=b,
                                     has_bias=has_bias,
                                     is_deconv=is_deconv,
                                     output_shape=output_shape,
                                     input_name=input_name,
                                     padding_top= padding_top,
                                     padding_left= padding_left,
                                     padding_bottom= padding_bottom,
                                     padding_right= padding_right,
                                     output_name=IR_node.real_name,
                                     dilation_factors=dilations)




    def emit_ConvTranspose(self, IR_node):
        """
        Convert convolution layer to coreml.
        """

        # assert False
        has_bias = IR_node.get_attr('use_bias', False)
        is_deconv = True

        # Get the weights.

        kernel_shape = IR_node.get_attr('kernel_shape')

        if len(kernel_shape) == 4:
            height, width, output_channels, kernel_channels = kernel_shape
            W = self.weights_dict[IR_node.name]['weights']
            W = W.reshape(kernel_shape)
            W = W.transpose((0, 1, 3, 2))
        elif len(kernel_shape) == 5:
            depth, height, width, output_channels, kernel_channels = kernel_shape
            W = self.weights_dict[IR_node.name]['weights']
            W = W.reshape(kernel_shape)
            W = W.transpose((0, 1, 2, 4, 3))
        else:
            raise NotImplementedError()


        output_shape = None
        b = self.weights_dict[IR_node.name]['bias'] if has_bias else None

        stride_height, stride_width = IR_node.get_attr('strides')[1], IR_node.get_attr('strides')[2]

        # Dilations
        dilations = IR_node.get_attr('dilations', [1, 1])
        if is_deconv and not dilations == [1, 1]:
            raise ValueError("Unsupported non-unity dilation for Deconvolution layer")

        groups = IR_node.get_attr('group', 1)

        padding = self._get_padding(IR_node)

        if isinstance(padding, list):
            border_mode = "valid"
            # see protobuf
            padding_top, padding_left, padding_bottom, padding_right = padding[1], padding [2], padding[5], padding [6]
        else:
            border_mode = "same"
            padding_top, padding_left, padding_bottom, padding_right = 0, 0, 0, 0


        input_name = self.IR_graph.get_parent(IR_node.name, [0]).real_name

        self.builder.add_convolution(name=IR_node.real_name,
                                     kernel_channels=kernel_channels,
                                     output_channels=output_channels,
                                     height=height,
                                     width=width,
                                     stride_height=stride_height,
                                     stride_width=stride_width,
                                     border_mode= border_mode,
                                     groups=groups,
                                     W=W,
                                     b=b,
                                     has_bias=has_bias,
                                     is_deconv=is_deconv,
                                     output_shape=output_shape,
                                     input_name=input_name,
                                     padding_top= padding_top,
                                     padding_left= padding_left,
                                     padding_bottom= padding_bottom,
                                     padding_right= padding_right,
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


        padding = self._get_padding(IR_node)



        if isinstance(padding, list):
            border_mode = "valid"
            # see protobuf
            padding_top, padding_left, padding_bottom, padding_right = padding[1], padding [2], padding[5], padding [6]
        else:
            border_mode = "same"
            padding_top, padding_left, padding_bottom, padding_right = 0, 0, 0, 0





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
                                     border_mode=border_mode,
                                     groups=groups,
                                     W=W,
                                     b=b,
                                     has_bias=has_bias,
                                     is_deconv=is_deconv,
                                     output_shape=output_shape,
                                     padding_top= padding_top,
                                     padding_left= padding_left,
                                     padding_bottom= padding_bottom,
                                     padding_right= padding_right,
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



                stride_height, stride_width = tuple(IR_node.get_attr('strides')[1:-1])
                height, width = 1, 1

                # TODO  global pooling modification

                # Padding
                padding = self._get_padding(IR_node)

                if isinstance(padding, list):
                    padding_type = "VALID"
                    # see protobuf
                    padding_top, padding_left, padding_bottom, padding_right = padding[1], padding[2], padding[5], padding[6]
                else:
                    padding_type = "SAME"
                    padding_top, padding_left, padding_bottom, padding_right = 0, 0, 0, 0


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
            if isinstance(padding, list):

                padding_type = "VALID"
                # see protobuf
                padding_top, padding_left, padding_bottom, padding_right = padding[1], padding [2], padding[5], padding [6]
            else:
                padding_type = "SAME"
                padding_top, padding_left, padding_bottom, padding_right = 0, 0, 0, 0


        self.builder.add_pooling(name=IR_node.name,
                                    height=height,
                                    width=width,
                                    stride_height=stride_height,
                                    stride_width=stride_width,
                                    layer_type=layer_type_str,
                                    padding_type=padding_type,
                                    padding_top= padding_top,
                                    padding_left= padding_left,
                                    padding_bottom= padding_bottom,
                                    padding_right= padding_right,
                                    input_name=input_name,
                                    output_name=IR_node.name,
                                    exclude_pad_area=True,
                                    is_global=global_pooling)


    def emit_Scale(self, IR_node):
        # Get input and output names
        input_name = self.IR_graph.get_node(IR_node.in_edges[0]).real_name

        weights = IR_node.get_attr('scale', False)
        weights = self.weights_dict[IR_node.name]['scale']
        has_bias = IR_node.get_attr('use_bias', False)
        if has_bias:
            bias = self.weights_dict[IR_node.name]['bias']


        shape_scale = self.weights_dict[IR_node.name]['shapeScale']
        if has_bias:
            shape_bias = self.weights_dict[IR_node.name]['shapeBias']



        self.builder.add_scale(name = IR_node.real_name,
                                        W = weights,
                                        b = bias,
                                        has_bias = has_bias,
                                        input_name = input_name,
                                        output_name =IR_node.name,
                                        shape_scale= [shape_scale],
                                        shape_bias= [shape_bias])



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
                # type: "list(int). A 1-D values of (leftBorder, topBorder, rightBorder, bottomBorder)."
                # This is central crop
                top, left = border[1], border[0]
                bottom, right = border[1], border[0]
            else:
                raise ValueError("Unrecognized padding option: %s" % (str(border)))

        # Now add the layer
        self.builder.add_crop(name = IR_node.name,
            left = left, right=right, top=top, bottom=bottom, offset = [0,0],
            input_names = [input_name], output_name=output_name
            )


    def emit_ReduceMean(self, IR_node):
        """
        Convert ReduceMean layer to coreml.
        """

        axis = IR_node.get_attr('axes', [1,2])

#       Allowed values: 'CHW', 'HW', 'C', 'H', 'W'
        if len(axis) == 1:
            if axis[0] == 0:
                axis_str = 'C'
            elif axis[0] == 1:
                axis_str = 'H'
            elif axis[0] == 2:
                axis_str = 'W'
        elif len(axis) == 2:
            axis_str = 'HW'
        elif len(axis) == 3:
            axis_str = 'CHW'

        # Get input and output names
        input_name = self.IR_graph.get_node(IR_node.in_edges[0]).real_name


        self.builder.add_reduce(IR_node.name,
                            input_name = input_name,
                            output_name = IR_node.name,
                            axis = axis_str,
                            mode = 'avg',
                            epsilon = 1e-6)


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



    def _emit_activation(self, IR_node, act, params=None):
        # Get input and output names
        input_name = self.IR_graph.get_parent(IR_node.name, [0]).real_name
        output_name = IR_node.real_name
        if not isinstance(params, list):
            params = [params]

        self.builder.add_activation(name=IR_node.real_name,
            non_linearity=act,
            input_name=input_name,
            output_name=output_name,
            params=params)


    # activation emit
    def emit_Relu(self, IR_node):
        self._emit_activation(IR_node, 'RELU')
    def emit_Tanh(self, IR_node):
        self._emit_activation(IR_node, 'TANH')
    def emit_PRelu(self, IR_node):
        self._emit_activation(IR_node, 'PRELU', IR_node.get_attr('gamma', 0) )
    def emit_LeakyRelu(self, IR_node):
        self._emit_activation(IR_node, 'LEAKYRELU', IR_node.get_attr('alpha', 0) )
    def emit_Elu(self,IR_node):
        self._emit_activation(IR_node, 'ELU',  IR_node.get_attr('alpha', 0)  )
    def emit_ThresholdedRelu(self, IR_node):
        self._emit_activation(IR_node, 'THRESHOLDEDRELU', IR_node.get_attr('alpha', 0) )
    def emit_ScaledTanh(self, IR_node):
        self._emit_activation(IR_node, 'SCALED_TANH', [IR_node.get_attr('alpha', 0),IR_node.get_attr('beta', 0)])
    def emit_linear(self, IR_node):
        self._emit_activation(IR_node, 'LINEAR', [IR_node.get_attr('alpha', 0),IR_node.get_attr('beta', 0)])
    def emit_SigmoidHard(self, IR_node):
        self._emit_activation(IR_node, 'SIGMOID_HARD', [IR_node.get_attr('alpha', 0),IR_node.get_attr('beta', 0)])
    def emit_ParametricSoftplus(self, IR_node):
        self._emit_activation(IR_node, 'PARAMETRICSOFTPLUS', [ IR_node.get_attr('alpha', 0),IR_node.get_attr('beta', 0) ])




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

        layer = IR_node.real_name
        input_name, output_name = (IR_node.IR_layer.input[0], IR_node.IR_layer.name)

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
        # Reference: parameter transformation https://github.com/apple/coremltools/issues/153
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

        # padding type TODO
        # Type of the padding. Can be one of 'constant', 'reflection' or 'replication
        padding_type = IR_node.get_attr('mode', 'CONSTANT')
        if padding_type == 'CONSTANT':
            padding_type = 'constant'
        elif padding_type == 'REFLECT':
            padding_type = 'reflection'
        elif padding_type == 'SYMMETRIC':
            padding_type = 'replication'


        # Now add the layer
        self.builder.add_padding(name = IR_node.name,
            left = left, right=right, top=top, bottom=bottom, value = 0,
            input_name = input_name, output_name=output_name, padding_type = padding_type
            )


    def emit_Squeeze(self, IR_node):
        input_name = self.IR_graph.get_parent(IR_node.name, [0]).real_name
        output_name=IR_node.real_name

        self.builder.add_bias(name = IR_node.name,
                              b = 0,
                              input_name = input_name,
                              output_name = output_name,
                              shape_bias = [1])
        # self.emit_Flatten(IR_node)


    def emit_LRN(self, IR_node):
        input_name = self.IR_graph.get_parent(IR_node.name, [0]).real_name
        output_name = IR_node.real_name
        alpha = IR_node.get_attr('alpha')
        beta = IR_node.get_attr('beta')
        bias = IR_node.get_attr('bias')
        size = IR_node.get_attr('size')

        self.builder.add_lrn(output_name, input_name, output_name,
                          alpha=alpha,
                          beta=beta,
                          local_size=size,
                          k=bias)


    def emit_SeparableConv(self, IR_node):

        input_name = self.IR_graph.get_parent(IR_node.name, [0]).real_name
        output_name = IR_node.real_name



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


    def emit_Slice(self, IR_node):
        pass
    def emit_Const(self, IR_node):
        pass

    def emit_Shape(self, IR_node):
        pass
    def emit_Pack(self, IR_node):
        pass
