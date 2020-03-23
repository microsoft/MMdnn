#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import absolute_import

import os
import gzip
from six import string_types as _string_types
import paddle.v2 as paddle
import paddle.trainer_config_helpers.layers as layers
import numpy as np
from mmdnn.conversion.paddle.paddle_graph import PaddleGraph
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.DataStructure.parser import Parser
from mmdnn.conversion.common.utils import *


class PaddleParser(Parser):

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

    layer_map = {
        "data"          : "InputLayer",
        "exconv"        : "Conv",
        "addto"         : "Add",
        "batch_norm"    : "BatchNormalization",
        "pool"          : "Pooling",
        "fc"            : "Dense",
        "norm"          : "LRN",


    }


    def _load_model(self, model_network_path, model_weight_path):
        """Load a paddle model from disk

        Parameters
        ----------
        model_network_path: str
            Path where the model network path is (json file)

        model_weight_path: str
            Path where the model network weights are (hd5 file)

        Returns
        -------
        model: A paddle model
        """
        from paddle.proto import ModelConfig_pb2
        from mmdnn.conversion.common.IR.IR_graph import load_protobuf_from_file

        loaded_model = ModelConfig_pb2.ModelConfig()
        load_protobuf_from_file(loaded_model, model_network_path)

        if model_weight_path:
            if os.path.isfile(model_weight_path):
                parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_weight_path, 'r'))
                self.weight_loaded = True
                print("Network file [{}] and [{}] is loaded successfully.".format(model_network_path, model_weight_path))

            else:
                print("Warning: Weights File [%s] is not found." % (model_weight_path))

        return loaded_model, parameters

    @property
    def src_graph(self):
        return self.paddle_graph


    def __init__(self, model):
        super(PaddleParser, self).__init__()

        if isinstance(model, tuple):
            model_network_path, model_weight_path = model

        # Build network graph
        model, parameters = self._load_model(model_network_path, model_weight_path)
        self.paddle_graph = PaddleGraph(model)
        self.paddle_graph.build()
        self.parameters = parameters
        self.shape_dict = dict()





    def gen_IR(self):

        for layer in self.paddle_graph.topological_sort:
            current_node = self.paddle_graph.get_node(layer)
            node_type = PaddleParser.layer_map[current_node.type]
            if hasattr(self, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                func(current_node)
            else:
                print("PaddleParser has not supported operator [%s]." % (node_type))
                self.rename_UNKNOWN(current_node)



    @staticmethod
    def _set_output_shape(source_node, IR_node, output_shapes):
        shape = graph_pb2.TensorShape()
        for output_shape in output_shapes:
            new_dim = shape.dim.add()
            new_dim.size = output_shape
        IR_node.attr["_output_shapes"].list.shape.extend([shape])


    @staticmethod
    def _copy_and_reop(source_node, IR_node, new_op = None):
        IR_node.name = source_node.name.lstrip('_')
        IR_node.op = source_node.type if new_op == None else new_op

        if hasattr(source_node.layer, "dtype"):
            IR_node.attr["dtype"].type = PaddleParser.dtype_map[source_node.layer.dtype]

        # PaddleParser._set_output_shape(source_node, IR_node)


    @staticmethod
    def _copy_shape(source_node, target_node, output_shapes):
        for dim in output_shapes:
            new_dim = target_node.attr["shape"].shape.dim.add()
            new_dim.size =  dim


    @staticmethod
    def _convert_dataformat(source_node, target_node):
        if source_node.keras_layer.data_format == 'channels_last':
            target_node.attr["data_format"].s = "NHWC"
        elif source_node.keras_layer.data_format == 'channels_first':
            target_node.attr["data_format"].s = "NCHW"
        else:
            print("Warning: [%s] don't have data format info." % (source_node.keras_layer.name))




    def _defuse_activation(self, source_node):
        src_spec = source_node.layer

        IR_node = self.IR_graph.node.add()
        IR_node.name = source_node.real_name.lstrip('_') + "_activation"
        IR_node.op = PaddleParser.activation_map[src_spec.active_type.encode()]
        IR_node.input.append(source_node.real_name.lstrip('_'))

        source_node.real_name = IR_node.name
        return IR_node



    def _convert_merge(self, source_node, new_name = None):
        IR_node = self.IR_graph.node.add()

        # name, op
        PaddleParser._copy_and_reop(source_node, IR_node, new_name)

        # input edge
        self.convert_inedge(source_node, IR_node)

        # For concat axis
        if hasattr(source_node.layer, 'axis'):
            IR_node.attr['axis'].i = -1
        return IR_node



    def rename_UNKNOWN(self, source_node):
        print (source_node.layer.get_config())

        # only for training
        IR_node = self.IR_graph.node.add()

        # name, op
        PaddleParser._copy_and_reop(source_node, IR_node)

        # input edge
        self.convert_inedge(source_node, IR_node)


    def rename_Conv(self, source_node):
        IR_node = self.IR_graph.node.add()

        # input edge
        self.convert_inedge(source_node, IR_node)

        # layer and spec
        conv_spec = source_node.layer

        spec = conv_spec.inputs[0].conv_conf

        # width <=> x or height <=> y
        width = spec.filter_size
        height = spec.filter_size_y if spec.HasField('filter_size_y') else spec.filter_size
        inputchannel = spec.channels
        outputchannel = conv_spec.num_filters
        stride_x = spec.stride
        stride_y = spec.stride_y if spec.HasField('stride_y') else stride_x
        padding_x = spec.padding
        padding_y = spec.padding_y if spec.HasField('padding_y') else padding_x
        dilation_x = spec.dilation
        dilation_y = spec.dilation_y if spec.HasField('dilation_y') else dilation_x
        output_x = spec.output_x
        output_y = spec.output_y if spec.HasField('output_y') else output_x
        input_x = spec.img_size
        input_y = spec.img_size_y if spec.HasField('img_size_y') else input_x


        # output shape
        output_shapes = [-1, output_y, output_x, outputchannel]
        self.shape_dict[source_node.name] = output_shapes
        PaddleParser._set_output_shape(source_node, IR_node, output_shapes)


        kwargs = dict()

        if conv_spec.type == 'exconv' or 'cudnn_conv':
            # name, op
            PaddleParser._copy_and_reop(source_node, IR_node, "Conv")
        else:
            kwargs['isDeconvolution'] = True
            PaddleParser._copy_and_reop(source_node, IR_node, "ConvTranspose")


        w_name = conv_spec.inputs[0].input_parameter_name
        w = self.parameters.get(w_name)


        self.set_weight(IR_node.name, 'weights', w.reshape([outputchannel, inputchannel, height, width]).transpose([ 2, 3, 1, 0]))

        #  it should be in the shape of height x width x inputchannel x outputchannel

        # use_bias: TODO
        kwargs['use_bias'] = False
        if conv_spec.HasField('bias_parameter_name'):
            bias_name = conv_spec.bias_parameter_name
            bias = self.parameters.get(bias_name).squeeze()
            self.set_weight(IR_node.name, "bias", bias)
            kwargs['use_bias'] = True



        kwargs['kernel_shape'] = [height, width, inputchannel, outputchannel]



        # pad_dim
        pad_dim = [0, 0, padding_x, padding_y, padding_x, padding_y, 0, 0]

        # fail report because of auto_pad
        # if dilation_x == 1 and dilation_y == 1:
        #     if output_x * stride_x == input_x and output_y * stride_y == input_y:
        #         auto_pad = "SAME"
        #         kwargs['auto_pad'] = auto_pad
        #     elif output_x * stride_x == input_x - width + 1 and output_y * stride_y == input_y - height + 1:
        #         auto_pad = "VALID"
        #         kwargs['auto_pad'] = auto_pad

        if input_x == output_x and input_y == output_y:
            auto_pad = "SAME"
        else:
            auto_pad = "SAME"

        pad_dim = convert_tf_pad_to_onnx(pad_dim)
        kwargs['pads'] = pad_dim

        kwargs['group'] = spec.groups

        kwargs['dilation'] = [1, dilation_x, dilation_y, 1]

        kwargs['strides'] = [1, stride_x, stride_y, 1]

        assign_IRnode_values(IR_node, kwargs)

        # defuse the activation layer

        if conv_spec.HasField('active_type') and  conv_spec.active_type != '':
            IR_node_act = self._defuse_activation(source_node)
            PaddleParser._set_output_shape(source_node, IR_node_act, output_shapes)


    def rename_BatchNormalization(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        PaddleParser._copy_and_reop(source_node, IR_node, "BatchNorm")

        # input edge
        self.convert_inedge(source_node, IR_node)

        # layer and spec
        bn_spec = source_node.layer



        # output shape
        if  bn_spec.inputs[0].HasField("image_conf"):
            img_conf = bn_spec.inputs[0].image_conf
            output_x = img_conf.img_size
            output_y = img_conf.img_size_y if img_conf.HasField('img_size_y') else output_x
            outputchannel = img_conf.channels

            output_shapes = [-1, output_y, output_x, outputchannel]
            self.shape_dict[source_node.name] = output_shapes
            PaddleParser._set_output_shape(source_node, IR_node, output_shapes)


        IR_node.attr['scale'].b = True
        IR_node.attr['bias'].b = bn_spec.HasField('bias_parameter_name')

        w_name = bn_spec.inputs[0].input_parameter_name
        mean_name = bn_spec.inputs[1].input_parameter_name
        var_name = bn_spec.inputs[2].input_parameter_name
        bias_name = bn_spec.bias_parameter_name

        gamma = self.parameters.get(w_name)
        mean = self.parameters.get(mean_name)
        variance = self.parameters.get(var_name)
        beta = self.parameters.get(bias_name)

        # channels_first, then axis = 1
        IR_node.attr['axis'].i = -1

        # epsilon
        IR_node.attr['epsilon'].f = bn_spec.epsilon

        # compute adjusted parameters
        # Reference: parameter transformation https://github.com/apple/coremltools/issues/153
        f = 1.0 / np.sqrt(variance +  bn_spec.epsilon)
        gamma1 = gamma*f
        beta1 = beta - gamma*mean*f
        mean[:] = 0.0 #mean
        variance[:] = 1.0 - .00001 #stddev

        # convert type because of tensorflow
        gamma1 = gamma1.astype(np.float32)
        beta1 = beta1.astype(np.float32)
        mean = mean.astype(np.float32)
        variance = variance.astype(np.float32)

        # flatten
        gamma1 = gamma1.flatten()
        beta1 = beta1.flatten()
        mean = mean.flatten()
        variance = variance.flatten()



        if IR_node.attr['scale'].b:
            self.set_weight(IR_node.name, "scale", gamma1)

        if IR_node.attr['bias'].b:
            self.set_weight(IR_node.name, "bias", beta1)

        # mean
        self.set_weight(IR_node.name, "mean", mean)

        # var
        self.set_weight(IR_node.name, "var", variance)

        # defuse the activation layer

        if bn_spec.HasField('active_type') and  bn_spec.active_type != '':
            IR_node_act = self._defuse_activation(source_node)
            if  bn_spec.inputs[0].HasField("image_conf"):
                PaddleParser._set_output_shape(source_node, IR_node_act, output_shapes)



    def rename_Pooling(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        PaddleParser._copy_and_reop(source_node, IR_node, "Pool")

        # input edge
        self.convert_inedge(source_node, IR_node)

        # layer and spec
        pool_spec = source_node.layer
        spec = pool_spec.inputs[0].pool_conf



        # assert False
        kwargs = dict()

        if spec.pool_type == 'max-projection':
            kwargs['pooling_type'] = 'MAX'
        elif spec.pool_type == 'avg-projection':
            kwargs['pooling_type'] = 'AVG'
        else:
            kwargs['pooling_type'] = 'MAX'



        width = spec.size_x
        height = spec.size_y if spec.HasField('size_y') else width
        channel = spec.channels
        stride_x = spec.stride
        stride_y = spec.stride_y if spec.HasField('stride_y') else stride_x
        padding_x = spec.padding
        padding_y = spec.padding_y if spec.HasField('padding_y') else padding_x
        output_x = spec.output_x
        output_y = spec.output_y if spec.HasField('output_y') else output_x
        input_x = spec.img_size
        input_y = spec.img_size_y if spec.HasField('img_size_y') else input_x


        # output shape
        output_shapes = [-1, output_y, output_x, channel]
        self.shape_dict[source_node.name] = output_shapes
        PaddleParser._set_output_shape(source_node, IR_node, output_shapes)


        kwargs['global_pooling'] = False

        kwargs['strides'] = [1, stride_x, stride_y, 1]
        kwargs['kernel_shape'] = [1, width, height, 1]

        # pad_dim
        pad_dim = [0, 0, padding_x, padding_y, padding_x, padding_y, 0, 0]


        # padding mode
        # If padding == "SAME": output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
        # If padding == "VALID": output_spatial_shape[i] = ceil((input_spatial_shape[i] - (spatial_filter_shape[i]-1) * dilation_rate[i]) / strides[i]).

        if output_x * stride_x == input_x and output_y * stride_y == input_y:
            auto_pad = "SAME"
            kwargs['auto_pad'] = auto_pad
        elif output_x * stride_x == input_x - width + 1 and output_y * stride_y == input_y - height + 1:
            auto_pad = "VALID"
            kwargs['auto_pad'] = auto_pad

        pad_dim = convert_tf_pad_to_onnx(pad_dim)
        kwargs['pads'] = pad_dim



        assign_IRnode_values(IR_node, kwargs)

        if pool_spec.HasField('active_type') and  pool_spec.active_type != '':
            IR_node_act = self._defuse_activation(source_node)
            PaddleParser._set_output_shape(source_node, IR_node_act, output_shapes)


    def rename_Dense(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        PaddleParser._copy_and_reop(source_node, IR_node, "FullyConnected")

        # input edge
        self.convert_inedge(source_node, IR_node)

        # layer and spec
        fc_spec = source_node.layer


        # units
        IR_node.attr['units'].i = fc_spec.size


        # output shape
        output_shapes = [-1, fc_spec.size]
        self.shape_dict[source_node.name] = output_shapes
        PaddleParser._set_output_shape(source_node, IR_node, output_shapes)


        # use_bias
        IR_node.attr['use_bias'].b = fc_spec.HasField('bias_parameter_name')

        w_name = fc_spec.inputs[0].input_parameter_name
        bias_name = fc_spec.bias_parameter_name

        w = self.parameters.get(w_name)

        bias = self.parameters.get(bias_name).flatten()

        # Kit weight tranpose
        # weight: N x M -> C x H x W x M -> H x W x C x M -> N x M
        if self.weight_loaded:
            parent = self.src_graph.get_parent(source_node.name, [0])
            if len(self.shape_dict[parent.name]) == 4:
                #
                original_shape = w.shape
                channel_first_list = self.shape_dict[parent.name][1:]
                dim = len(channel_first_list) + 1
                weight = w.reshape(channel_first_list + [original_shape[1]])
                assert dim > 2
                weight = weight.transpose(list(range(1, dim-1)) + [0, dim-1])
                w = weight.reshape(original_shape)
        if fc_spec.HasField('drop_rate'):
            w = w * fc_spec.drop_rate
            if IR_node.attr['use_bias'].b:
                bias = bias * fc_spec.drop_rate


        # weights
        self.set_weight(IR_node.name, 'weights', w)
        if IR_node.attr['use_bias'].b:
            self.set_weight(IR_node.name, 'bias', bias)

        if fc_spec.HasField('active_type') and  fc_spec.active_type != '':
            IR_node_act = self._defuse_activation(source_node)
            PaddleParser._set_output_shape(source_node, IR_node_act, output_shapes)






    def rename_Add(self, source_node):
        add_spec = source_node.layer
        self._convert_merge(source_node, 'Add')
        if add_spec.HasField('active_type') and  add_spec.active_type != '':
            self._defuse_activation(source_node)


    def rename_InputLayer(self, source_node):
        # need the shape TODO

        # only for training
        IR_node = self.IR_graph.node.add()

        # name, op
        PaddleParser._copy_and_reop(source_node, IR_node, "DataInput")

        # input edge
        self.convert_inedge(source_node, IR_node)

        output_shapes = [-1, 224, 224, 3]
        # shape
        PaddleParser._copy_shape(source_node.layer, IR_node, output_shapes)


    def rename_LRN(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        PaddleParser._copy_and_reop(source_node, IR_node, "LRN")

        # input edge
        self.convert_inedge(source_node, IR_node)

        # layer and spec
        lrn_spec = source_node.layer
        spec = lrn_spec.inputs[0].norm_conf
        channels = spec.channels
        size = spec.size
        alpha = spec.scale
        beta = spec.pow
        img_size_x = spec.img_size
        img_size_y = spec.img_size_y if spec.HasField('img_size_y') else img_size_x
        output_x = spec.output_x
        output_y = spec.output_y if spec.HasField('output_y') else output_x


        # output shape
        output_shapes = [-1, output_y, output_x, channels]
        self.shape_dict[source_node.name] = output_shapes
        PaddleParser._set_output_shape(source_node, IR_node, output_shapes)

        # alpha
        IR_node.attr["alpha"].f = alpha * size
        # beta
        IR_node.attr["beta"].f = beta
        # nsize
        IR_node.attr["size"].i = int(size)


        if lrn_spec.HasField('active_type') and  lrn_spec.active_type != '':
            IR_node_act = self._defuse_activation(source_node)
            PaddleParser._set_output_shape(source_node, IR_node_act, output_shapes)


