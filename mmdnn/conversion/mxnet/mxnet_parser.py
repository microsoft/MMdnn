#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import os
import sys
import math
import mxnet as mx
import numpy as np
from mmdnn.conversion.mxnet.mxnet_graph import MXNetGraph
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.DataStructure.parser import Parser
from mmdnn.conversion.common.utils import *

class MXNetParser(Parser):

    dtype_map = {
        "int8"      : graph_pb2.DT_INT8,
        "int16"     : graph_pb2.DT_INT16,
        "int32"     : graph_pb2.DT_INT32,
        "int64"     : graph_pb2.DT_INT64,
        "uint8"     : graph_pb2.DT_UINT8,
        "uint16"    : graph_pb2.DT_UINT16,
        "uint32"    : graph_pb2.DT_UINT32,
        "uint64"    : graph_pb2.DT_UINT64,
        "float16"   : graph_pb2.DT_FLOAT16,
        "float32"   : graph_pb2.DT_FLOAT32,
        "float64"   : graph_pb2.DT_FLOAT64
    }

    activation_map = {
        "relu"      : "Relu",
        "sigmoid"   : "Sigmoid",
        "tanh"      : "Tanh",
        # Not support yet
        # "softrelu"  : "SoftReLU"
    }

    channels_last = ['NDHWC', 'NHWC', 'NWC']
    channels_first = ['NCDHW', 'NCHW', 'NCW']

    @property
    def src_graph(self):
        return self.mxnet_graph

    @staticmethod
    def str2bool(v):
        return v.lower() in ("1", "true")


    @staticmethod
    def str2intList(v):
        v = v.replace("(", "")
        v = v.replace(")", "")
        if v == "":
            return list()
        else:
            return [int(s) for s in v.split(',')]


    @staticmethod
    def transpose(data, dim):
        if dim == 1:
            data = data.transpose((2, 1, 0))
        elif dim == 2:
            data = data.transpose((2, 3, 1, 0))
        elif dim == 3:
            data = data.transpose((2, 3, 4, 1, 0))
        else:
            print("Warning: The weight of dim {0} cannot transpose" % dim)

        return data


    @staticmethod
    def _convert_axis(IR_node, axis):
        ndim = len(IR_node.attr['_output_shapes'].list.shape[0].dim)
        if axis == 0:
            return 0
        elif axis == 1:
            return ndim - 1
        else:
            return axis - 1


    def trace_shape(self, source_node, IR_node):
        input_node = self.IR_layer_map[IR_node.input[0]]
        while len(input_node.attr['_output_shapes'].list.shape[0].dim) <= 2:
            IR_node = input_node
            input_node = self.IR_layer_map[IR_node.input[0]]

        input_shape = list()
        for e in input_node.attr["_output_shapes"].list.shape[0].dim:
            input_shape.append(e.size)
        C = input_shape.pop()
        ret = [C] + input_shape[1:]
        return ret


    def check_pad_mode(self, source_node, IR_node):
        kernel = MXNetParser.str2intList(source_node.get_attr("kernel"))
        dim = len(kernel)

        pad = source_node.get_attr("pad", "()")
        if pad == "()":
            pad = list([0] * dim)
        else:
            pad = MXNetParser.str2intList(pad)

        stride = source_node.get_attr("stride")
        if stride == None:
            stride = list([1] * dim)
        else:
            stride = MXNetParser.str2intList(stride)

        dilate = source_node.get_attr("dilate")
        if dilate == None:
            dilate = list([1] * dim)
        else:
            dilate = MXNetParser.str2intList(dilate)

        input_shape = list()
        if len(source_node.in_edges) == 0 or IR_node.input[0] not in self.IR_layer_map:
            input_shape = self.data_shape
        else:
            for e in self.IR_layer_map[IR_node.input[0]].attr["_output_shapes"].list.shape[0].dim:
                input_shape.append(e.size)

        valid_flag = True
        same_flag = True

        for i in range(dim):
            if not pad[i] == 0:
                valid_flag = False
            output_shape = int(math.floor(float(input_shape[i] + 2 * pad[i] - dilate[i] * (kernel[i] - 1) - 1) / float(stride[i])) + 1)
            same_pad_shape = int(math.ceil(float(input_shape[i]) / float(stride[i])))
            if not output_shape == same_pad_shape:
                same_flag = False

        if valid_flag:
            return "VALID"
        elif same_flag:
            return "SAME"
        else:
            return "None"


    @staticmethod
    def _load_model(weights, epoch):
        """Load a mxnet model from disk

        Parameters
        ----------
        model_path: str
            Path where the model network/params path is (json/params file)

        prefix: str
            prefix for json file, e.g. prefix-symbol.json

        epoch: int
            save epoch number

        Returns
        -------
        model: A mxnet model
        params: A pair of dictionaries each mapping parameter names to NDArray values
        """

        # Load the model network and weights
        sym, arg_params, aux_params = mx.model.load_checkpoint(weights, int(epoch))

        # digraph = mx.viz.plot_network(sym, save_format='jpg') # For debugging
        # digraph.render()

        model = mx.mod.Module(symbol = sym)
        arg_params.update(aux_params)
        return model, arg_params

        '''
        MXNet new api does not support load data without data_shapes
        '''
        # model.bind(data_shapes = data_shapes)
        # model.init_params()

        # mod.load(model_path, epoch_num)
        # return mod.get_params()


    @staticmethod
    def _load_json_file(model_path):
        """Load a mxnet network json file

        Parameters
        ----------
        model_path: str
            Path where the model network/params path is (json/params file)

        (Deleted)
        prefix: str
            prefix for json file, e.g. prefix-symbol.json

        Returns
        -------
        data["nodes"]: all the layer information(including weights, bias) with format
            data["nodes"][layer_num][params = {"name", "op", "attr", "inputs"}]

        """
        import json

        # load the model network
        with open(model_path, 'r') as data_file:
            data = json.load(data_file)

        # adjust the data format
        assert isinstance(data["nodes"], list)
        return data["nodes"]


    def __init__(self, input_arg):
        super(MXNetParser, self).__init__()

        json_data = list()
        self.data_shape = tuple()
        # load model files into MXNet graph
        # data_shape arguments added to calculate infer_shape(required)
        # if isinstance(input_arg, basestring):
        if len(input_arg) == 2:
            with open(input_arg[0], 'r') as input_json:
                json_string = input_json.read()
                symbol = mx.sym.load_json(json_string)
                self.model = mx.mod.Module(symbol = symbol)
            json_data = MXNetParser._load_json_file(input_arg[0])
            self.data_shape = tuple([1] + list(map(int, input_arg[1])))

        elif len(input_arg) == 4:
            self.model, self.weight_data = MXNetParser._load_model(input_arg[1], input_arg[2])
            json_data = MXNetParser._load_json_file(input_arg[0])
            self.weight_loaded = True
            assert isinstance(input_arg[3], list)
            self.data_shape = tuple([1] + list(map(int, input_arg[3])))

        else:
            raise ValueError("the # of input arguments [{}] is not supported" % len(input_arg))

        # Build network graph
        self.data_format = 'None'
        self.mxnet_graph = MXNetGraph(self.model)
        self.mxnet_graph.build(json_data)


    def gen_IR(self):
        self.IR_layer_map = dict()
        for layer in self.mxnet_graph.topological_sort:
            current_node = self.mxnet_graph.get_node(layer)
            node_type = current_node.type
            if hasattr(self, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                func(current_node)

            else:
                self.rename_UNKNOWN(current_node)


    def _copy_and_reop(self, source_node, IR_node, new_op = None):
        new_op = source_node.type if new_op == None else new_op
        if source_node.name.startswith('_'):
            source_node.real_name = source_node.name[1:]
        IR_node.name = source_node.real_name
        IR_node.op = new_op
        self.IR_layer_map[IR_node.name] = IR_node


    def set_output_shape(self, source_node, IR_node):
        sym_group = self.model.symbol.get_internals()
        for sym in sym_group:
            if source_node.name == sym.name:
                arg_shape, output_shape, aux_shape = sym.infer_shape(data = self.data_shape)
                for idx in range(len(output_shape)):
                    output_list = list(output_shape[idx])

                    # transpose to channel last
                    if not self.data_format in MXNetParser.channels_last:
                        channel = output_list.pop(1)
                        output_list.append(channel)

                    if IR_node.op == "DataInput":
                        MXNetParser._copy_shape(IR_node, [-1] + output_list[1:])

                    shape = graph_pb2.TensorShape()
                    for dim in output_list:
                        new_dim = shape.dim.add()
                        if dim == None:
                            new_dim.size = -1
                        else:
                            new_dim.size = dim

                    IR_node.attr["_output_shapes"].list.shape.extend([shape])
                break

    def _convert_identity_operation(self, source_node, new_op=None):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node, new_op)

        # input edge
        self.convert_inedge(source_node, IR_node)

        # output shape
        self.set_output_shape(source_node, IR_node)

        return IR_node

    def _defuse_padding(self, source_node):
        IR_node = self.IR_graph.node.add()
        IR_node.name = source_node.name + "_pad"
        IR_node.op = "Pad"
        # input edge
        self.convert_inedge(source_node, IR_node)

        self.IR_layer_map[IR_node.name] = IR_node

        # attr
        assign_IRnode_values(IR_node, {'mode' : 'CONSTANT'})
        # print("Warning: MXNet symbol pad does not support channel last")

        pad = MXNetParser.str2intList(source_node.get_attr("pad"))
        args['pads'] = [0, 0]
        for e in pad:
            args['pads'].extend([e, e])
        args['pads'] += [0, 0]
        args['pads'] = convert_tf_pad_to_onnx(args['pads'])
        IR_node.set_attrs(args)

        # IR_node.attr["pads"].list.i.extend([0, 0])
        # for e in pad:
        #     IR_node.attr["pads"].list.i.extend([e, e])
        # IR_node.attr["pads"].list.i.extend([0, 0])

        IR_node.attr["constant_values"].f = 0.


    @staticmethod
    def _copy_shape(IR_node, output_list):
        if not output_list == None:
            for dim in output_list:
                new_dim = IR_node.attr["shape"].shape.dim.add()
                if dim == None:
                    new_dim.size = -1
                else:
                    new_dim.size = dim
        else:
            IR_node.attr["shape"].shape.unknown_rank = True


    def rename_UNKNOWN(self, source_node):
        print("Warning: MXNet Parser has not supported operator %s with name %s."
            % (source_node.type, source_node.name))
        if source_node.type == "null" and source_node.name != 'label':
            print("Warning: convert the null operator with name [%s] into input layer." % source_node.name)
            IR_node = self.IR_graph.node.add()

            # name, op
            self._copy_and_reop(source_node, IR_node, "DataInput")

            # input edge
            self.convert_inedge(source_node, IR_node)

            self.set_output_shape(source_node, IR_node)

        else:
            raise NotImplementedError()



    """
    Here start with Neural Network Symbol
    """

    def rename_Pad(self, source_node):
        IR_node = self._convert_identity_operation(source_node)
        kwargs = dict()
        pad = MXNetParser.str2intList(source_node.get_attr("pad_width"))
        pad += [pad.pop(2), pad.pop(3)]
        kwargs['pads'] = pad
        kwargs['pads'] = convert_tf_pad_to_onnx(kwargs['pads'])
        kwargs['mode'] = 'CONSTANT'
        assign_IRnode_values(IR_node, kwargs)
        IR_node.attr["constant_values"].f = 0.


    def rename_FullyConnected(self, source_node):
        IR_node = self._convert_identity_operation(source_node)

        # units
        IR_node.attr["units"].i = int(source_node.get_attr("num_hidden"))

        # use bias (no_bias default = False)
        IR_node.attr["use_bias"].b = not MXNetParser.str2bool(source_node.get_attr("no_bias", "False"))

        # weights
        if self.weight_loaded:
            if self.data_format == 'NM':
                self.set_weight(source_node.name, "weights", self.weight_data.get(source_node.name + "_weight").asnumpy().transpose((1, 0)))
            else:
                weight = self.weight_data.get(source_node.name + "_weight").asnumpy().transpose((1, 0))
                original_shape = weight.shape

                channel_first_list = self.trace_shape(source_node, IR_node)
                dim = len(channel_first_list) + 1
                weight = weight.reshape(channel_first_list + [original_shape[1]])
                assert dim > 2
                weight = weight.transpose(list(range(1, dim-1)) + [0, dim-1])
                weight = weight.reshape(original_shape)
                self.set_weight(source_node.name, "weights", weight)

            if IR_node.attr["use_bias"].b:
                self.set_weight(source_node.name, "bias", self.weight_data.get(source_node.name + "_bias").asnumpy())

        if not self.data_format == 'NM':
            # print("Warning: Layer [{}] has changed model data format from [{}] to [NM]".format(source_node.name, self.data_format))
            self.data_format = 'NM'


    def rename_Convolution(self, source_node):
        IR_node = self.IR_graph.node.add()

        # input edge
        self.convert_inedge(source_node, IR_node)

        # output shape
        self.set_output_shape(source_node, IR_node)

        dim = 0
        layout = 'None'

        # kernel_shape
        kernel = MXNetParser.str2intList(source_node.get_attr("kernel"))
        dim = len(kernel)
        IR_node.attr["kernel_shape"].list.i.extend(kernel)

        layout = source_node.get_attr("layout")
        if layout == None or layout == 'None':
            if dim == 1:
                layout = "NCW"
            elif dim == 2:
                layout = "NCHW"
            elif dim == 3:
                layout = "NCDHW"

        if not self.data_format == layout:
            # print("Warning: Layer [{}] has changed model data format from [{}] to [{}]".format(source_node.name, self.data_format, layout))
            self.data_format = layout

        # groups
        group = int(source_node.get_attr("num_group", "1"))
        IR_node.attr["group"].i = group
        in_channel = self.IR_layer_map[IR_node.input[0]].attr["_output_shapes"].list.shape[0].dim[-1].size

        if group == in_channel:
            self._copy_and_reop(source_node, IR_node, "DepthwiseConv")
        else:
            self._copy_and_reop(source_node, IR_node, "Conv")
        # in_channel = in_channel // group

        out_channel = int(source_node.get_attr("num_filter"))

        IR_node.attr["kernel_shape"].list.i.extend([in_channel, out_channel])

        # use_bias (no_bias default = False)
        IR_node.attr["use_bias"].b = not MXNetParser.str2bool(source_node.get_attr("no_bias", "False"))

        # strides
        strides = source_node.get_attr("stride")
        IR_node.attr["strides"].list.i.append(1)
        if not strides == None:
            IR_node.attr["strides"].list.i.extend(MXNetParser.str2intList(strides))
        else:
            IR_node.attr["strides"].list.i.extend([1] * dim)
        IR_node.attr["strides"].list.i.append(1)

        # dilations
        dilate = source_node.get_attr("dilate")
        IR_node.attr["dilations"].list.i.append(1)
        if not dilate == None:
            IR_node.attr["dilations"].list.i.extend(MXNetParser.str2intList(dilate))
        else:
            IR_node.attr["dilations"].list.i.extend([1] * dim)
        IR_node.attr["dilations"].list.i.append(1)

        # data_format
        assign_IRnode_values(IR_node, {'data_format' : layout})

        # padding
        if "pad" in source_node.attr:
            pad = MXNetParser.str2intList(source_node.get_attr("pad"))
            IR_node.attr["pads"].list.i.extend(([0]+pad+[0])*2)
        else:
            IR_node.attr["pads"].list.i.extend([0, 0] * (dim + 2))

        # weights
        if self.weight_loaded:
            weight = self.weight_data.get(source_node.name + "_weight").asnumpy()
            if not layout in MXNetParser.channels_last:
                weight = MXNetParser.transpose(weight, dim)
                if IR_node.op == "DepthwiseConv":
                    weight = weight.transpose(0, 1, 3, 2)
            self.set_weight(source_node.name, "weights", weight)

            if IR_node.attr["use_bias"].b:
                self.set_weight(source_node.name, "bias", self.weight_data.get(source_node.name + "_bias").asnumpy())


    def rename_Activation(self, source_node):
        self._convert_identity_operation(source_node, new_op=MXNetParser.activation_map[source_node.get_attr("act_type")])


    def rename_BatchNorm(self, source_node):
        IR_node = self._convert_identity_operation(source_node)

        # axis
        if self.data_format in MXNetParser.channels_first or self.data_format == 'None':
            IR_node.attr["axis"].i = MXNetParser._convert_axis(IR_node, int(source_node.get_attr("axis", "1")))
        else:
            IR_node.attr["axis"].i = int(source_node.get_attr("axis", "1"))

        # scale
        IR_node.attr["scale"].b = not MXNetParser.str2bool(source_node.get_attr("fix_gamma", "True"))
        IR_node.attr["bias"].b = True
        # epsilon
        IR_node.attr["epsilon"].f = float(source_node.get_attr("eps", "0.001"))

        # momentum
        IR_node.attr["momentum"].f = float(source_node.get_attr("momentum", "0.9"))

        # weights
        if self.weight_loaded:
            # gamma
            if IR_node.attr["scale"].b:
                self.set_weight(source_node.name, "scale", self.weight_data.get(source_node.name + "_gamma").asnumpy())

            # beta
            if IR_node.attr["bias"].b:
                self.set_weight(source_node.name, "bias", self.weight_data.get(source_node.name + "_beta").asnumpy())

            # mean
            self.set_weight(source_node.name, "mean", self.weight_data.get(source_node.name + "_moving_mean").asnumpy())

            # var
            self.set_weight(source_node.name, "var", self.weight_data.get(source_node.name + "_moving_var").asnumpy())


    def rename_Pooling(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node, "Pool")

        # input edge
        self.convert_inedge(source_node, IR_node)

        # pooling type (sum not allowed yet)
        pool_type = source_node.get_attr("pool_type")
        if pool_type == "sum":
            print("Warning: sum pooling is not supported yet.")
        elif pool_type == "max":
            assign_IRnode_values(IR_node, {'pooling_type' : 'MAX'})
        elif pool_type == "avg":
            assign_IRnode_values(IR_node, {'pooling_type' : 'AVG'})
        else:
            raise ValueError("Error pool_type {}.".format(pool_type))

        kernel_shape = MXNetParser.str2intList(source_node.get_attr("kernel"))

        if MXNetParser.str2bool(source_node.get_attr("global_pool", "False")):

            IR_node.attr['global_pooling'].b = True
            IR_node.attr["kernel_shape"].list.i[:] = [1] * (len(kernel_shape) + 2)
            IR_node.attr["strides"].list.i[:] = [1] * (len(kernel_shape) + 2)
        else:
            IR_node.attr['global_pooling'].b = False

            # strides
            strides = source_node.get_attr("stride")
            IR_node.attr["strides"].list.i.append(1)
            if not strides == None:
                IR_node.attr["strides"].list.i.extend(MXNetParser.str2intList(strides))
            IR_node.attr["strides"].list.i.append(1)

            # kernel_shape
            IR_node.attr["kernel_shape"].list.i.append(1)
            IR_node.attr["kernel_shape"].list.i.extend(kernel_shape)
            IR_node.attr["kernel_shape"].list.i.append(1)

            # padding
            if "pad" in source_node.attr:
                pad = MXNetParser.str2intList(source_node.get_attr("pad"))
                IR_node.attr["pads"].list.i.extend(([0]+pad+[0])*2)
            else:
                IR_node.attr["pads"].list.i.extend(([0])*8)

        # output shape
        self.set_output_shape(source_node, IR_node)


    def rename_SoftmaxOutput(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node, "Softmax")

        # input edge
        self.convert_inedge(source_node, IR_node)

        if "attr" in source_node.layer or "param" in source_node.layer:
            print("Warning: SoftmaxOutput attrs are not supported in IR.")

        # output shape
        self.set_output_shape(source_node, IR_node)


    def rename_softmax(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='Softmax')

        # dim
        if self.data_format in MXNetParser.channels_first or self.data_format == 'None':
            IR_node.attr["dim"].i = MXNetParser._convert_axis(IR_node, int(source_node.get_attr("axis", "-1")))
        else:
            IR_node.attr["dim"].i = int(source_node.get_attr("axis", "-1"))


    # def rename_log_softmax(self, source_node):
    #   raise NotImplementedError("not support yet")


    # def rename_Correlation(self, source_node):
    #   raise NotImplementedError("not support yet")


    def rename_Deconvolution(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node, "ConvTranspose")

        # input edge
        self.convert_inedge(source_node, IR_node)

        dim = 0
        layout = 'None'

        # padding
        if "pad" in source_node.attr:
            pad = MXNetParser.str2intList(source_node.get_attr("pad"))
            IR_node.attr["pads"].list.i.extend(([0]+pad+[0])*2)
        else:
            IR_node.attr["pads"].list.i.extend([0, 0] * (dim + 2))

        # output shape
        self.set_output_shape(source_node, IR_node)

        # kernel_shape
        kernel = MXNetParser.str2intList(source_node.get_attr("kernel"))
        dim = len(kernel)
        IR_node.attr["kernel_shape"].list.i.extend(kernel)

        layout = source_node.get_attr("layout")
        if layout == None or layout == 'None':
            if dim == 1:
                layout = "NCW"
            elif dim == 2:
                layout = "NCHW"
            elif dim == 3:
                layout = "NCDHW"

        if not self.data_format == layout:
            # print("Warning: Layer [{}] has changed model data format from [{}] to [{}]".format(source_node.name, self.data_format, layout))
            self.data_format = layout

        in_channel = self.IR_layer_map[IR_node.input[0]].attr["_output_shapes"].list.shape[0].dim[-1].size

        out_channel = int(source_node.get_attr("num_filter"))

        IR_node.attr["kernel_shape"].list.i.extend([out_channel, in_channel])

        # use_bias (no_bias default = False)
        IR_node.attr["use_bias"].b = not MXNetParser.str2bool(source_node.get_attr("no_bias", "False"))

        # strides
        strides = source_node.get_attr("strides")
        IR_node.attr["strides"].list.i.append(1)
        if not strides == None:
            IR_node.attr["strides"].list.i.extend(MXNetParser.str2intList(strides))
        else:
            IR_node.attr["strides"].list.i.extend([1] * dim)
        IR_node.attr["strides"].list.i.append(1)

        # dilations
        dilate = source_node.get_attr("dilate")
        IR_node.attr["dilations"].list.i.append(1)
        if not dilate == None:
            IR_node.attr["dilations"].list.i.extend(MXNetParser.str2intList(dilate))
        else:
            IR_node.attr["dilations"].list.i.extend([1] * dim)
        IR_node.attr["dilations"].list.i.append(1)

        # data_format
        IR_node.attr["data_format"].s = layout

        # groups
        IR_node.attr["group"].i = int(source_node.get_attr("num_group", "1"))

        # weights
        if self.weight_loaded:
            weight = self.weight_data.get(source_node.name + "_weight").asnumpy()
            if not layout in MXNetParser.channels_last:
                weight = MXNetParser.transpose(weight, dim)
            self.set_weight(source_node.name, "weights", weight)

            if IR_node.attr["use_bias"].b:
                self.set_weight(source_node.name, "bias", self.weight_data.get(source_node.name + "_bias").asnumpy())


    # def rename_RNN(self, source_node):
    #   raise NotImplementedError("RNN not support yet")


    def rename_Embedding(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node)

        # input edge
        self.convert_inedge(source_node, IR_node)

        # input_dim
        IR_node.attr["input_dim"].i = int(source_node.get_attr("input_dim"))

        # output_dim
        IR_node.attr["output_dim"].i = int(source_node.get_attr("output_dim"))

        # dtype
        IR_node.attr["dtype"].type = MXNetParser.dtype_map[source_node.get_attr("dtype", "float32")]

        # output shape
        self.set_output_shape(source_node, IR_node)


    # IR only support elu and prelu from {'elu', 'leaky', 'prelu', 'rrelu'}
    def rename_LeakyReLU(self, source_node):
        act_type = source_node.get_attr('act_type', None)
        if act_type:
            if not act_type == "elu" and not act_type == "prelu":
                print("Warning: Activation Type %s is not supported yet." % act_type)
                # return

        IR_node = self.IR_graph.node.add()

        # name, op
        if act_type == 'prelu':
            self._copy_and_reop(source_node, IR_node, "PRelu")

            # gamma
            self.set_weight(source_node.name, "gamma", self.weight_data.get(source_node.name + "_gamma").asnumpy())

        else:  # All other cases set to 'Elu'
            self._copy_and_reop(source_node, IR_node, "Elu")

        # input edge
        self.convert_inedge(source_node, IR_node)

        # alpha [exp(x) - alpha], but mxnet attr slope [slope*(exp(x) - 1)] when x < 0
        if "slope" in source_node.attr:
            raise ValueError("Attribute Slope is not supported in IR format")
        # IR_node.attr["alpha"].f = float()

        # output shape
        self.set_output_shape(source_node, IR_node)

        # raise NotImplementedError("slope cannot convert to alpha")


    # def rename_InstanceNorm(self, source_node):
    #   raise NotImplementedError


    # def rename_L2Normalization(self, source_node):
    #   raise NotImplementedError


    def rename_LRN(self, source_node):
        IR_node = self._convert_identity_operation(source_node)
        alpha = source_node.get_attr("alpha", "0.0001")
        beta = source_node.get_attr("beta", "0.75")
        bias = source_node.get_attr("knorm", "2")
        size = source_node.get_attr("nsize")

        IR_node.attr["alpha"].f = alpha
        IR_node.attr["beta"].f = beta
        IR_node.attr["bias"].f = bias
        IR_node.attr["size"].i = size


    def rename_ROIPooling(self, source_node):
        raise NotImplementedError()


    def rename_Dropout(self, source_node):
        IR_node = self._convert_identity_operation(source_node)

        # keep_prob
        IR_node.attr["keep_prob"].f = float(source_node.get_attr("p", "0.5"))

        # mode
        assign_IRnode_values(IR_node, {'mode' : 'training'})


    """
    Here start with Symbol manipulation routines
    """

    def rename_UpSampling(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op="UpSampling2D")
        kwargs = dict()
        scale = int(source_node.get_attr("scale"))
        interpolation_type = source_node.get_attr("sample_type")
        scales = [scale, scale]
        kwargs["scales"] = scales
        kwargs["interpolation_type"] = interpolation_type
        assign_IRnode_values(IR_node, kwargs)


    # reverse cannot support yet
    def rename_reshape(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='Reshape')

        # old API target_shape not support yet
        shape = source_node.get_attr("shape")
        if not shape == None:
            shape_list = MXNetParser.str2intList(shape)
            for param in shape_list:
                if param <= 0 and not param == -1:
                    raise ValueError("special value %d for Reshape is not pre-defined in IR." % param)
            IR_node.attr["shape"].list.i.extend(shape_list)

        # output shape
        self.set_output_shape(source_node, IR_node)

        # raise NotImplementedError("adjust output shape")


    def rename_Flatten(self, source_node):
        self._convert_identity_operation(source_node, new_op='Flatten')


    def rename_Concat(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='Concat')

        # dim
        if self.data_format in MXNetParser.channels_first or self.data_format == 'None':
            IR_node.attr["axis"].i = MXNetParser._convert_axis(IR_node, int(source_node.get_attr("dim", "1")))
        else:
            IR_node.attr["axis"].i = int(source_node.get_attr("dim", "1"))


    def rename_cast(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='Cast')

        # dtype
        IR_node.attr["dtype"].type = MXNetParser.dtype_map[source_node.get_attr("dtype")]

        # output shape
        self.set_output_shape(source_node, IR_node)


    def rename_expand_dims(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node)

        # input edge
        self.convert_inedge(source_node, IR_node)

        # output shape
        self.set_output_shape(source_node, IR_node)

        # axis
        if self.data_format in MXNetParser.channels_first or self.data_format == 'None':
            IR_node.attr["axis"].i = MXNetParser._convert_axis(IR_node, int(source_node.get_attr("axis")))
        else:
            IR_node.attr["axis"].i = int(source_node.get_attr("axis"))


    def rename_elemwise_add(self, source_node):
        self._convert_identity_operation(source_node, new_op='Add')


    def rename__Plus(self, source_node):
        self._convert_identity_operation(source_node, new_op='Add')


    def rename_broadcast_add(self, source_node):
        self._convert_identity_operation(source_node, new_op='Add')


    def rename_broadcast_mul(self, source_node):
        self._convert_identity_operation(source_node, new_op='Mul')


    def rename__mul(self, source_node):
        self._convert_identity_operation(source_node, new_op='Mul')


    def rename__copy(self, source_node):
        self._convert_identity_operation(source_node)
        # raise NotImplementedError("No matching IR api")


    def _convert_scalar_operator(self, source_node, new_op):
        value = source_node.get_attr('scalar')
        value_node = self.IR_graph.node.add()
        value_node.name = source_node.real_name + "_second"
        # left strip the "_" at the beginning of the name
        # Issue #85, #135
        value_node.name = value_node.name.lstrip('_')
        value_node.op = 'Constant'
        self.set_weight(value_node.name, 'value', np.array([value], np.float32))

        IR_node = self._convert_identity_operation(source_node, new_op)
        IR_node.input.append(value_node.name)
        return IR_node


    def rename__mul_scalar(self, source_node):
        self._convert_scalar_operator(source_node, 'Mul')


    def rename__minus_scalar(self, source_node):
        self._convert_scalar_operator(source_node, 'Sub')

    def rename__div_scalar(self, source_node):
        self._convert_scalar_operator(source_node, 'Div')

    def rename__copy(self, source_node):
        source_node.real_name = self.get_parent(source_node.name, [0]).real_name


    def rename_BlockGrad(self, source_node):
        return

