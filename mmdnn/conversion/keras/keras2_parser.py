#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import os
from six import string_types as _string_types
import keras as _keras
from keras import backend as _K
from mmdnn.conversion.keras.keras2_graph import Keras2Graph
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.DataStructure.parser import Parser
from mmdnn.conversion.common.utils import *


class Keras2Parser(Parser):

    dtype_map = {
        "float16" : graph_pb2.DT_FLOAT16,
        "float32" : graph_pb2.DT_FLOAT32,
        "float64" : graph_pb2.DT_FLOAT64,
        "int16"   : graph_pb2.DT_INT16,
        "int32"   : graph_pb2.DT_INT32,
        "int64"   : graph_pb2.DT_INT64,
        "uint8"   : graph_pb2.DT_UINT8,
        "uint16"  : graph_pb2.DT_UINT16
    }

    activation_map = {
        "relu"          : "Relu",
        'softmax'       : "Softmax",
        'sigmoid'       : "Sigmoid",
        "tanh"          : "Tanh",
        "elu"           : "Elu",
        "relu6"         : "Relu6",
        'softplus'      : 'Softplus',
        'softsign'      : 'Softsign',
        'hard_sigmoid'  : 'HardSigmoid'
    }


    def _load_model(self, model_network_path, model_weight_path):
        """Load a keras model from disk

        Parameters
        ----------
        model_network_path: str
            Path where the model network path is (json file)

        model_weight_path: str
            Path where the model network weights are (hd5 file)

        Returns
        -------
        model: A keras model
        """
        from keras.models import model_from_json

        # Load the model network
        json_file = open(model_network_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        # Load the model weights
        loaded_model = model_from_json(loaded_model_json, custom_objects={
            'relu6': _keras.applications.mobilenet.relu6,
            'DepthwiseConv2D': _keras.applications.mobilenet.DepthwiseConv2D})

        if model_weight_path:
            if os.path.isfile(model_weight_path):
                loaded_model.load_weights(model_weight_path)
                self.weight_loaded = True
                print("Network file [{}] and [{}] is loaded successfully.".format(model_network_path, model_weight_path))

            else:
                print("Warning: Weights File [%s] is not found." % (model_weight_path))

        return loaded_model

    @property
    def src_graph(self):
        return self.keras_graph


    def __init__(self, model):
        super(Keras2Parser, self).__init__()

        # load model files into Keras graph
        if isinstance(model, _string_types):
            model = _keras.models.load_model(
                model,
                custom_objects={
                    'relu6': _keras.applications.mobilenet.relu6,
                    'DepthwiseConv2D': _keras.applications.mobilenet.DepthwiseConv2D
                }
            )
            self.weight_loaded = True

        elif isinstance(model, tuple):
            model = self._load_model(model[0], model[1])

        else:
            assert False

        # _keras.utils.plot_model(model, "model.png", show_shapes = True)

        # Build network graph
        self.data_format = _keras.backend.image_data_format()
        self.keras_graph = Keras2Graph(model)
        self.keras_graph.build()
        self.lambda_layer_count = 0


    def gen_IR(self):
        for layer in self.keras_graph.topological_sort:
            current_node = self.keras_graph.get_node(layer)
            node_type = current_node.type
            if hasattr(self, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                func(current_node)
            else:
                print("KerasParser has not supported operator [%s]." % (node_type))
                self.rename_UNKNOWN(current_node)

        _K.clear_session()


    @staticmethod
    def _set_output_shape(source_node, IR_node):
        shape = graph_pb2.TensorShape()
        for dim in source_node.layer.output_shape:
            new_dim = shape.dim.add()
            new_dim.size = dim if dim else -1

        IR_node.attr["_output_shapes"].list.shape.extend([shape])


    @staticmethod
    def _copy_and_reop(source_node, IR_node, new_op = None):
        IR_node.name = source_node.name
        IR_node.op = source_node.type if new_op == None else new_op

        if hasattr(source_node.layer, "dtype"):
            IR_node.attr["dtype"].type = Keras2Parser.dtype_map[source_node.layer.dtype]

        Keras2Parser._set_output_shape(source_node, IR_node)


    @staticmethod
    def _copy_shape(source_node, target_node):
        if hasattr(source_node, "output_shape"):
            for dim in source_node.output_shape:
                new_dim = target_node.attr["shape"].shape.dim.add()
                new_dim.size = -1 if dim == None else dim

        else:
            target_node.attr["shape"].shape.unknown_rank = True


    @staticmethod
    def _convert_dataformat(source_node, target_node):
        if source_node.keras_layer.data_format == 'channels_last':
            target_node.attr["data_format"].s = "NHWC"
        elif source_node.keras_layer.data_format == 'channels_first':
            target_node.attr["data_format"].s = "NCHW"
        else:
            print("Warning: [%s] don't have data format info." % (source_node.keras_layer.name))


    @staticmethod
    def _convert_padding(source_node, IR_node):
        # TODO: Fused conv and pool with padding is different from defused operators
        dims = len(source_node.layer.input_shape)
        if source_node.layer.padding == 'valid':
            assign_IRnode_values(IR_node, {'auto_pad' : "VALID", 'pads' : [0, 0] * dims})

        elif source_node.layer.padding == 'same':
            kernel_shape = source_node.layer.kernel_size if hasattr(source_node.layer, 'kernel_size') else source_node.layer.pool_size
            padding = compute_tf_same_padding(
                source_node.layer.input_shape,
                kernel_shape,
                list(source_node.layer.strides))
            assign_IRnode_values(IR_node, {'auto_pad' : "SAME_LOWER", 'pads' : padding})

        else:
            assert False


    def _defuse_activation(self, source_node):
        if source_node.layer.activation is None or source_node.layer.activation.__name__ == "linear":
            return

        IR_node = self.IR_graph.node.add()
        IR_node.name = source_node.real_name + "_activation"
        IR_node.op = Keras2Parser.activation_map[source_node.layer.activation.__name__]
        IR_node.input.append(source_node.real_name)
        Keras2Parser._set_output_shape(source_node, IR_node)

        # TODO: More activation functions
        # for ELU
        if hasattr(source_node.layer, 'alpha'):
            assign_attr_value(IR_node['alpha'], source_node.layer.alpha)

        source_node.real_name = IR_node.name


    def _convert_convolution(self, source_node, dim):
        IR_node = self.IR_graph.node.add()

        # input edge
        self.convert_inedge(source_node, IR_node)

        # name, op
        if source_node.type.startswith('Separable'):
            Keras2Parser._copy_and_reop(source_node, IR_node, "SeparableConv")
            if self.weight_loaded:
                self.set_weight(source_node.name, 'depthwise_filter', source_node.layer.get_weights()[0])
                self.set_weight(source_node.name, 'pointwise_filter', source_node.layer.get_weights()[1])

        else:
            if source_node.type.startswith('Conv'):
                Keras2Parser._copy_and_reop(source_node, IR_node, "Conv")

            elif source_node.type.startswith('Deconv'):
                Keras2Parser._copy_and_reop(source_node, IR_node, "ConvTranspose")

            elif source_node.type.startswith('Depthwise'):
                Keras2Parser._copy_and_reop(source_node, IR_node, "DepthwiseConv")

            else:
                raise NotImplementedError("Convolution layer [{}] is not supported.".format(source_node.type))

            # weights
            if self.weight_loaded:
                self.set_weight(source_node.name, "weights", source_node.layer.get_weights()[0])
                if source_node.layer.use_bias:
                    self.set_weight(source_node.name, "bias", source_node.layer.get_weights()[1])

        if isinstance(source_node.layer.kernel_size, int):
            source_node.layer.kernel_size = (source_node.layer.kernel_size) * dim

        if isinstance(source_node.layer.strides, int):
            source_node.layer.strides = (source_node.layer.strides) * dim

        if isinstance(source_node.layer.dilation_rate, int):
            source_node.layer.dilation_rate = (source_node.layer.dilation_rate) * dim

        kwargs = dict()

        # pads
        Keras2Parser._convert_padding(source_node, IR_node)

        # filter
        # [kd, kh, kw, channel_size, filter number]
        in_channel = source_node.layer.input_shape[-1] if self.data_format == "channels_last" else source_node.layer.input_shape[1]
        out_channel = source_node.layer.filters or source_node.layer.depth_multiplier

        if source_node.type.startswith("Deconv"):
            kwargs['kernel_shape'] = list(source_node.layer.kernel_size) + [out_channel, in_channel]
        else:
            kwargs['kernel_shape'] = list(source_node.layer.kernel_size) + [in_channel, out_channel]

        # use_bias
        kwargs['use_bias'] = source_node.keras_layer.use_bias

        # strides
        # [1, sd, sh, sw, 1]
        kwargs['strides'] = [1] + list(source_node.layer.strides) + [1]

        # dilations
        # [1, dd, dh, dw, 1]
        kwargs['dilations'] = [1] + list(source_node.layer.dilation_rate) + [1]

        assign_IRnode_values(IR_node, kwargs)

        # activation
        self._defuse_activation(source_node)


    def _convert_pooling(self, source_node, dim, pooling_type, is_global):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node, "Pool")

        # input edge
        self.convert_inedge(source_node, IR_node)

        kwargs = {}

        kwargs['pooling_type'] = pooling_type


        if is_global:
            kwargs['global_pooling'] = True
            kwargs['strides'] = [1] * (dim + 2)
        else:
            if isinstance(source_node.layer.pool_size, int):
                source_node.layer.pool_size = (source_node.layer.pool_size) * dim

            if isinstance(source_node.layer.strides, int):
                source_node.layer.strides = (source_node.layer.strides) * dim

            # padding
            self._convert_padding(source_node, IR_node)

            # strides
            # [1, sd, sh, sw, 1]
            kwargs['strides'] = [1] + list(source_node.layer.strides) + [1]

            # window_shape
            # [1, pd, ph, pw, 1]
            kwargs['kernel_shape'] = [1] + list(source_node.layer.pool_size) + [1]

        assign_IRnode_values(IR_node, kwargs)

        if is_global:
            flatten_node = self.IR_graph.node.add()
            flatten_node.name = source_node.name + '_flatten'
            flatten_node.op = 'Flatten'
            flatten_node.input.append(source_node.name)
            Keras2Parser._set_output_shape(source_node, flatten_node)
            source_node.real_name = flatten_node.name


    def _convert_merge(self, source_node, new_name = None):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node, new_name)

        # input edge
        self.convert_inedge(source_node, IR_node)

        # For concat axis
        if hasattr(source_node.layer, 'axis'):
            IR_node.attr['axis'].i = source_node.layer.axis
        return IR_node


    def _convert_padding_api(self, source_node, IR_node, mode):
         # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node, "Pad")

        # input edge
        self.convert_inedge(source_node, IR_node)

        kwargs = dict()
        kwargs['mode'] = mode

        # padding
        kwargs['pads'] = [0, 0]
        for padding_pair in source_node.layer.padding:
            kwargs['pads'].extend(padding_pair)
        kwargs['pads'] += [0, 0]
        kwargs['pads'] = convert_tf_pad_to_onnx(kwargs['pads'])
        assign_IRnode_values(IR_node, kwargs)


    def rename_UNKNOWN(self, source_node):
        print (source_node.layer.get_config())

        # only for training
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node)

        # input edge
        self.convert_inedge(source_node, IR_node)


    def rename_Activation(self, keras_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(keras_node, IR_node, self.activation_map[keras_node.keras_layer.activation.__name__])

        # input edge
        self.convert_inedge(keras_node, IR_node)


    # Merge Layers
    def rename_Add(self, source_node):
        self._convert_merge(source_node)


    def rename_Conv1D(self, source_node):
        self._convert_convolution(source_node, 1)


    def rename_Conv2D(self, source_node):
        self._convert_convolution(source_node, 2)


    def rename_Conv3D(self, source_node):
        self._convert_convolution(source_node, 3)


    def rename_InputLayer(self, source_node):
        # only for training
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node, "DataInput")

        # input edge
        self.convert_inedge(source_node, IR_node)

        # shape
        Keras2Parser._copy_shape(source_node.keras_layer, IR_node)



    def rename_GlobalMaxPooling1D(self, source_node):
        self._convert_pooling(source_node, 1, "MAX", True)


    def rename_GlobalMaxPooling2D(self, source_node):
        self._convert_pooling(source_node, 2, "MAX", True)


    def rename_GlobalMaxPooling3D(self, source_node):
        self._convert_pooling(source_node, 3, "MAX", True)


    def rename_GlobalAveragePooling1D(self, source_node):
        self._convert_pooling(source_node, 1, "AVG", True)


    def rename_GlobalAveragePooling2D(self, source_node):
        self._convert_pooling(source_node, 2, "AVG", True)


    def rename_GlobalAveragePooling3D(self, source_node):
        self._convert_pooling(source_node, 3, "AVG", True)


    def rename_MaxPooling1D(self, source_node):
        self._convert_pooling(source_node, 1, "MAX", False)


    def rename_MaxPooling2D(self, source_node):
        self._convert_pooling(source_node, 2, "MAX", False)


    def rename_MaxPooling3D(self, source_node):
        self._convert_pooling(source_node, 3, "MAX", False)


    def rename_AveragePooling1D(self, source_node):
        self._convert_pooling(source_node, 1, "AVG", False)


    def rename_AveragePooling2D(self, source_node):
        self._convert_pooling(source_node, 2, "AVG", False)


    def rename_AveragePooling3D(self, source_node):
        self._convert_pooling(source_node, 3, "AVG", False)


    def rename_Dropout(self, source_node):
        # only for training
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node)

        # input edge
        self.convert_inedge(source_node, IR_node)

        IR_node.attr["keep_prob"].f = source_node.keras_layer.rate
        if source_node.keras_layer.seed != None:
            IR_node.attr["seed"].i = source_node.keras_layer.seed


    # Core Layers
    def rename_Dense(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node, "FullyConnected")

        # input edge
        self.convert_inedge(source_node, IR_node)

        # units
        IR_node.attr["units"].i = source_node.keras_layer.units

        # use_bias
        IR_node.attr["use_bias"].b = source_node.keras_layer.use_bias

        # weights
        if self.weight_loaded == True:
            self.set_weight(source_node.name, 'weights', source_node.layer.get_weights()[0])
            if IR_node.attr["use_bias"].b == True:
                self.set_weight(source_node.name, 'bias', source_node.layer.get_weights()[1])

        # activation
        self._defuse_activation(source_node)


    def rename_Flatten(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node)

        # input edge
        self.convert_inedge(source_node, IR_node)


    def rename_Embedding(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node)

        # input edge
        self.convert_inedge(source_node, IR_node)

        # input_dim
        IR_node.attr["input_dim"].i = source_node.keras_layer.input_dim

        # output_dim
        IR_node.attr["output_dim"].i = source_node.keras_layer.output_dim

        # mask_zero
        IR_node.attr["mask_zero"].b = source_node.keras_layer.mask_zero

        # weights
        if self.weight_loaded:
            self.set_weight(source_node.name, 'embedding_weights', source_node.layer.get_weights()[0])


    def rename_LSTM(self, keras_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(keras_node, IR_node)

        # input edge
        self.convert_inedge(keras_node, IR_node)

        # units
        IR_node.attr["units"].i = keras_node.keras_layer.units

        # use_bias
        IR_node.attr["use_bias"].b = keras_node.keras_layer.use_bias

        # for Keras, drop_out and recurrent_dropout
        IR_node.attr["dropout"].f = keras_node.keras_layer.dropout
        IR_node.attr["recurrent_dropout"].f = keras_node.keras_layer.recurrent_dropout

        # activation
        self._defuse_activation(keras_node)


    def rename_GRU(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node)

        # input edge
        self.convert_inedge(source_node, IR_node)

        # units
        IR_node.attr["units"].i = source_node.keras_layer.units

        # activation
        self._defuse_activation(source_node)

        # weights
        if self.weight_loaded:
            self.set_weight(source_node.name, 'gru_weights', source_node.layer.get_weights()[0])
            self.set_weight(source_node.name, 'gru_recurrent_weights', source_node.layer.get_weights()[1])
            if source_node.layer.use_bias:
                self.set_weight(source_node.name, "gru_bias", source_node.layer.get_weights()[2])


    def rename_Multiply(self, source_node):
        self._convert_merge(source_node, 'Mul')


    def rename_Average(self, source_node):
        # Kit TODO : need to search the tf
        self._convert_merge(source_node, 'Avg')


    def rename_Maximum(self, source_node):
        self._convert_merge(source_node)


    def rename_Concatenate(self, source_node):
        IR_node = self._convert_merge(source_node, 'Concat')


    def rename_Reshape(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node, 'Reshape')

        # input edge
        self.convert_inedge(source_node, IR_node)

        # for target shape
        IR_node.attr["shape"].list.i.append(-1)
        IR_node.attr["shape"].list.i.extend(source_node.layer.target_shape)


    def rename_Lambda(self, source_node):
        node_type = source_node.layer.name
        if hasattr(self, "rename_" + node_type):
            print ("Try to convert Lambda function [{}]".format(source_node.layer.name))
            func = getattr(self, "rename_" + node_type)
            func(source_node)
        else:
            raise NotImplementedError("Lambda layer [{}] in keras is not supported yet.".format(node_type))


    def rename_BatchNormalization(self, keras_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(keras_node, IR_node, 'BatchNorm')

        # input edge
        self.convert_inedge(keras_node, IR_node)

        # axis
        IR_node.attr['axis'].i = keras_node.keras_layer.axis

        IR_node.attr['scale'].b = keras_node.keras_layer.scale

        IR_node.attr['bias'].b = keras_node.keras_layer.center

        IR_node.attr['epsilon'].f = keras_node.layer.epsilon

        if self.weight_loaded:
            # Parameter arrangement in Keras: gamma, beta, mean, variance
            idx = 0

            # scale
            if IR_node.attr['scale'].b:
                self.set_weight(keras_node.name, "scale", keras_node.layer.get_weights()[idx])
                idx += 1

            # beta
            if IR_node.attr['bias'].b:
                self.set_weight(keras_node.name, "bias", keras_node.layer.get_weights()[idx])
                idx += 1

            # mean
            self.set_weight(keras_node.name, "mean", keras_node.layer.get_weights()[idx])

            # var
            self.set_weight(keras_node.name, "var", keras_node.layer.get_weights()[idx + 1])


    def rename_ZeroPadding2D(self, keras_node):
        IR_node = self.IR_graph.node.add()
        self._convert_padding_api(keras_node, IR_node, "constant")


    def rename_SeparableConv2D(self, source_node):
        self._convert_convolution(source_node, 2)


    def rename_DepthwiseConv2D(self, source_node):
        self._convert_convolution(source_node, 2)


    def custom_relu6(x):
        return _keras.relu(x, max_value=6)


    def _convert_crop(self, source_node):
        IR_node = self.IR_graph.node.add()

        Keras2Parser._copy_and_reop(source_node, IR_node, "Crop")

        self.convert_inedge(source_node, IR_node)

        border = []
        for i in source_node.layer.cropping:
            for j in i:
                border.append(j)

        assign_IRnode_values(IR_node, {'border' : border})



    def rename_Cropping1D(self, source_node):
        self._convert_crop(source_node)


    def rename_Cropping2D(self, source_node):
        self._convert_crop(source_node)


    def rename_Cropping3D(self, source_node):
        self._convert_crop(source_node)


    def rename_LeakyReLU(self, source_node):
        IR_node = self.IR_graph.node.add()
        Keras2Parser._copy_and_reop(source_node, IR_node, 'LeakyRelu')
        self.convert_inedge(source_node, IR_node)
        assign_IRnode_values(IR_node, {'alpha' : source_node.layer.alpha.tolist()})


    def rename_space_to_depth_x2(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node, 'SpaceToDepth')
        IR_node.name = "Lambda_{}".format(self.lambda_layer_count)

        # input edge
        self.convert_inedge(source_node, IR_node)

        # for target shape
        IR_node.attr["blocksize"].i = 2
        self.lambda_layer_count = self.lambda_layer_count + 1
        source_node.real_name = IR_node.name
