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

    channels_last = ['NDHWC', 'NHWC']


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


    def trace_shape(self, source_node, IR_node):
        while (not IR_node.op == "Flatten"):
            IR_node = self.IR_layer_map[IR_node.input[0]]
        IR_node = self.IR_layer_map[IR_node.input[0]]
        input_shape = list()
        for e in IR_node.attr["_output_shapes"].list.shape[0].dim:
            input_shape.append(e.size)
        C = input_shape[-1]
        H = input_shape[1]
        W = input_shape[2]
        return C, H, W


    def check_pad_mode(self, source_node, IR_node):
        assert "attr" in source_node.layer or "param" in source_node.layer
        layer_attr = dict()
        if "attr" in source_node.layer:
            layer_attr = source_node.layer["attr"]
        elif "param" in source_node.layer:
            layer_attr = source_node.layer["param"]

        assert "kernel" in layer_attr
        kernel = MXNetParser.str2intList(layer_attr.get("kernel"))
        dim = len(kernel)

        assert "pad" in layer_attr
        pad = layer_attr.get("pad", "()")
        if pad == "()":
            pad = list([0] * dim)
        else:
            pad = MXNetParser.str2intList(pad)

        stride = layer_attr.get("stride")
        if stride == None:
            stride = list([1] * dim)
        else:
            stride = MXNetParser.str2intList(stride)

        dilate = layer_attr.get("dilate")
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

        # raise NotImplementedError

        
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
        
        # raise NotImplementedError


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

        # raise NotImplementedError


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

        # raise NotImplementedError


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

        # for i, j in self.weights.items():
        #     print ("parameter [{}] have weights [{}]".format(i, len(j)))


    def _copy_and_reop(self, source_node, IR_node, new_op = None):
        new_op = source_node.type if new_op == None else new_op
        if source_node.name.startswith('_'):
            source_node.real_name = source_node.name[1:]
        IR_node.name = source_node.real_name
        IR_node.op = new_op
        self.IR_layer_map[IR_node.name] = IR_node

    
    def _convert_inedge(self, source_node, IR_node, start_idx = 0, end_idx = None):        
        if end_idx == None: end_idx = len(source_node.in_edges)        
        for idx in range(start_idx, end_idx):
            IR_node.input.append(self.src_graph.get_node(source_node.in_edges[idx]).real_name)


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


    def _convert_arithmetic(self, source_node, new_op = None):
        IR_node = self.IR_graph.node.add()

        # name, op
        if new_op == None:
            self._copy_and_reop(source_node, IR_node)
        else:
            self._copy_and_reop(source_node, IR_node, new_op)

        # input edge
        self._convert_inedge(source_node, IR_node)

        # output shape
        self.set_output_shape(source_node, IR_node)        


    def _defuse_padding(self, source_node):
        IR_node = self.IR_graph.node.add()
        IR_node.name = source_node.name + "_pad"
        IR_node.op = "Pad"
        # input edge
        self._convert_inedge(source_node, IR_node)

        self.IR_layer_map[IR_node.name] = IR_node

        # attr
        IR_node.attr["mode"].s = b"CONSTANT"
        # print("Warning: MXNet symbol pad does not support channel last")

        assert "attr" in source_node.layer or "param" in source_node.layer
        layer_attr = dict()
        if "attr" in source_node.layer:
            layer_attr = source_node.layer["attr"]
        elif "param" in source_node.layer:
            layer_attr = source_node.layer["param"]

        assert "pad" in layer_attr
        pad = MXNetParser.str2intList(layer_attr.get("pad"))
        IR_node.attr["paddings"].list.i.extend([0, 0])
        for e in pad:
            IR_node.attr["paddings"].list.i.extend([e, e])
        IR_node.attr["paddings"].list.i.extend([0, 0])

        IR_node.attr["constant_values"].f = 0.

        # output shape
        # self.set_output_shape(source_node, IR_node)

        # self.mxnet_graph.layer_name_map[source_node.name] = IR_node.name
        # raise NotImplementedError


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
        if source_node.type == "null":
            print("Warning: convert the null operator with name [%s] into input layer." % source_node.name)
            IR_node = self.IR_graph.node.add()

            # name, op
            self._copy_and_reop(source_node, IR_node, "DataInput")

            # input edge
            self._convert_inedge(source_node, IR_node)

            self.set_output_shape(source_node, IR_node)
        return 
        # raise NotImplementedError


    """
    Here start with Neural Network Symbol
    """

    def rename_FullyConnected(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node, "FullyConnected")

        # input edge
        self._convert_inedge(source_node, IR_node)

        # attr
        assert "attr" in source_node.layer or "param" in source_node.layer
        layer_attr = dict()
        if "attr" in source_node.layer:
            layer_attr = source_node.layer["attr"]
        elif "param" in source_node.layer:
            layer_attr = source_node.layer["param"]

        # units 
        IR_node.attr["units"].i = int(layer_attr.get("num_hidden"))

        # use bias (no_bias default = False)
        IR_node.attr["use_bias"].b = not MXNetParser.str2bool(layer_attr.get("no_bias", "False"))

        # output shape
        self.set_output_shape(source_node, IR_node)

        # weights
        if self.weight_loaded:
            if self.data_format == 'NM':
                self.set_weight(source_node.name, "weights", self.weight_data.get(source_node.name + "_weight").asnumpy().transpose((1, 0)))
            else:
                weight = self.weight_data.get(source_node.name + "_weight").asnumpy().transpose((1, 0))
                input_channel, input_height, input_width = self.trace_shape(source_node, IR_node)
                output_shape = int(layer_attr.get("num_hidden"))
                weight = weight.reshape((input_channel, input_height, input_width, output_shape))
                weight = weight.transpose((1, 2, 0, 3))
                weight = weight.reshape((input_channel* input_height* input_width, output_shape))
                self.set_weight(source_node.name, "weights", weight)

            if IR_node.attr["use_bias"].b:
                self.set_weight(source_node.name, "bias", self.weight_data.get(source_node.name + "_bias").asnumpy())
        
        if not self.data_format == 'NM':
            # print("Warning: Layer [{}] has changed model data format from [{}] to [NM]".format(source_node.name, self.data_format))
            self.data_format = 'NM'

        # raise NotImplementedError


    def rename_Convolution(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node, "Convolution")

        # input edge
        self._convert_inedge(source_node, IR_node)

        dim = 0
        layout = 'None'
        # attr
        assert "attr" in source_node.layer or "param" in source_node.layer
        layer_attr = dict()
        if "attr" in source_node.layer:
            layer_attr = source_node.layer["attr"]
        elif "param" in source_node.layer:
            layer_attr = source_node.layer["param"]

        # filter
        assert "kernel" in layer_attr
        kernel = MXNetParser.str2intList(layer_attr.get("kernel"))
        dim = len(kernel)
        IR_node.attr["filter"].list.i.extend(kernel)

        in_channel = 0
        layout = layer_attr.get("layout")
        if layout == None or layout == 'None':
            if dim == 1:
                layout = "NCW"
            elif dim == 2:
                layout = "NCHW"
            elif dim == 3:
                layout = "NCDHW"

        if not self.data_format == layout:
            # print("Warning: Layer [{}] has changed model data format from [{}] to [{}]".format(source_node.name, self.data_format, layout))
            self.data_format = layout.encode('utf-8')

        if self.weight_loaded:
            if layout in MXNetParser.channels_last:
                in_channel = self.weight_data.get(source_node.name + "_weight").shape[-1]
            else:
                in_channel = self.weight_data.get(source_node.name + "_weight").shape[1]

        assert "num_filter" in layer_attr
        out_channel = int(layer_attr.get("num_filter"))

        IR_node.attr["filter"].list.i.extend([in_channel, out_channel])

        # use_bias (no_bias default = False)
        IR_node.attr["use_bias"].b = not MXNetParser.str2bool(layer_attr.get("no_bias", "False"))

        # strides
        strides = layer_attr.get("stride")
        IR_node.attr["strides"].list.i.append(1)
        if not strides == None:
            IR_node.attr["strides"].list.i.extend(MXNetParser.str2intList(strides))
        IR_node.attr["strides"].list.i.append(1)

        # dilations
        dilate = layer_attr.get("dilate")
        IR_node.attr["dilation_rate"].list.i.append(1)
        if not dilate == None:
            IR_node.attr["dilation_rate"].list.i.extend(MXNetParser.str2intList(dilate))
        IR_node.attr["dilation_rate"].list.i.append(1)

        # data_format
        IR_node.attr["data_format"].s = layout.encode()

        # padding
        # IR only support pad = "SAME" or "VALID"
        # Using Padding Layer API
        if "pad" in layer_attr:
            # pad = self.check_pad_mode(source_node, IR_node)
            # if pad == "SAME":
            #     IR_node.attr["padding"].s = b"SAME"
            # elif pad == "VALID":
            #     IR_node.attr["padding"].s = b"VALID"
            # else:
            #     self._defuse_padding(source_node)
            #     del IR_node.input[:]
            #     IR_node.input.append(source_node.name + "_pad")

            IR_node.attr["padding"].s = b"VALID"
            self._defuse_padding(source_node)
            del IR_node.input[:]
            IR_node.input.append(source_node.name + "_pad")


        # output shape
        self.set_output_shape(source_node, IR_node)

        # weights
        if self.weight_loaded:
            weight = self.weight_data.get(source_node.name + "_weight").asnumpy()
            if not layout in MXNetParser.channels_last:
                weight = MXNetParser.transpose(weight, dim)
            self.set_weight(source_node.name, "weights", weight)
            
            if IR_node.attr["use_bias"].b:
                self.set_weight(source_node.name, "bias", self.weight_data.get(source_node.name + "_bias").asnumpy())
        
        # raise NotImplementedError


    def rename_Activation(self, source_node):
        IR_node = self.IR_graph.node.add()

        # if source_node.layer.act_type == "relu":
        #     MXNetParser._copy_and_reop(source_node, IR_node, "Relu")
        # elif source_node.layer.act_type == "sigmoid":
        #     MXNetParser._copy_and_reop(source_node, IR_node, "Sigmoid")
        # elif source_node.layer.act_type == "tanh":
        #     MXNetParser._copy_and_reop(source_node, IR_node, "Tanh")
        # elif source_node.layer.act_type == "softrelu":
        #     MXNetParser._copy_and_reop(source_node, IR_node, "SoftReLU")

        assert "attr" in source_node.layer or "param" in source_node.layer
        layer_attr = dict()
        if "attr" in source_node.layer:
            layer_attr = source_node.layer["attr"]
        elif "param" in source_node.layer:
            layer_attr = source_node.layer["param"]

        assert "act_type" in layer_attr
        self._copy_and_reop(
            source_node, IR_node, MXNetParser.activation_map[layer_attr.get("act_type")])

        # output shape
        self.set_output_shape(source_node, IR_node)

        self._convert_inedge(source_node, IR_node)
        
        # raise NotImplementedError


    def rename_BatchNorm(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node, "BatchNorm")

        # input edge
        self._convert_inedge(source_node, IR_node)

        layer_attr = dict()
        if "attr" in source_node.layer:
            layer_attr = source_node.layer["attr"]
        elif "param" in source_node.layer:
            layer_attr = source_node.layer["param"]

        # axis
        # if self.data_format not in MXNetParser.channels_last: 
        # 	IR_node.attr["axis"].i = int(layer_attr.get("axis", "1"))
        # else:
        IR_node.attr["axis"].i = -1

        # scale
        IR_node.attr["scale"].b = not MXNetParser.str2bool(layer_attr.get("fix_gamma", "True"))
        IR_node.attr["bias"].b = True
        # epsilon
        IR_node.attr["epsilon"].f = float(layer_attr.get("eps", "0.001"))

        # momentum
        IR_node.attr["momentum"].f = float(layer_attr.get("momentum", "0.9"))

        # output shape
        self.set_output_shape(source_node, IR_node)

        # weights
        if self.weight_loaded:

            # gamma
            if IR_node.attr["scale"].b:
                self.set_weight(source_node.name, "scale", self.weight_data.get(source_node.name + "_gamma").asnumpy())

            # beta
            if IR_node.attr["bias"].b:
                self.set_weight(source_node.name, "bias", self.weight_data.get(source_node.name + "_beta").asnumpy())

            # if MXNetParser.str2bool(layer_attr.get("use_global_stats", "False")):
            # mean
            self.set_weight(source_node.name, "mean", self.weight_data.get(source_node.name + "_moving_mean").asnumpy())

            # var
            self.set_weight(source_node.name, "var", self.weight_data.get(source_node.name + "_moving_var").asnumpy())
        
        # raise NotImplementedError("bool type for scale and offset")


    def rename_Pooling(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node, "Pool")

        # input edge
        self._convert_inedge(source_node, IR_node)

        layer_attr = dict()
        if "attr" in source_node.layer:
            layer_attr = source_node.layer["attr"]
        elif "param" in source_node.layer:
            layer_attr = source_node.layer["param"]

        # pooling type (sum not allowed yet)
        pool_type = layer_attr.get("pool_type")
        if pool_type == "sum":
            print("Warning: sum pooling is not supported yet.")	
        elif pool_type == "max":
            IR_node.attr['pooling_type'].s = b"MAX"
        elif pool_type == "avg":
            IR_node.attr['pooling_type'].s = b"AVG"
        else:
            raise ValueError("Error pool_type {}.".format(pool_type))

        assert "kernel" in layer_attr
        window_shape = MXNetParser.str2intList(layer_attr.get("kernel"))

        if MXNetParser.str2bool(layer_attr.get("global_pool", "False")):
            IR_node.attr['global_pooling'].b = True
            IR_node.attr["window_shape"].list.i[:] = [1] * (len(window_shape) + 2)
            IR_node.attr["strides"].list.i[:] = [1] * (len(window_shape) + 2)
        else:
            IR_node.attr['global_pooling'].b = False

            # strides
            strides = layer_attr.get("stride")
            IR_node.attr["strides"].list.i.append(1)
            if not strides == None:
                IR_node.attr["strides"].list.i.extend(MXNetParser.str2intList(strides))
            IR_node.attr["strides"].list.i.append(1)

            # window_shape
            IR_node.attr["window_shape"].list.i.append(1)
            IR_node.attr["window_shape"].list.i.extend(window_shape)
            IR_node.attr["window_shape"].list.i.append(1)

            # padding
            # IR only support pad = "SAME" or "Valid"
            if "pad" in layer_attr:
                # pad = self.check_pad_mode(source_node, IR_node)
                # if pad == b"SAME":
                #     IR_node.attr["padding"].s = b"SAME"
                # elif pad == b"VALID":
                #     IR_node.attr["padding"].s = b"VALID"
                # else:
                #     IR_node.attr["padding"].s = b"VALID"
                #     self._defuse_padding(source_node)
                #     del IR_node.input[:]
                #     IR_node.input.append(source_node.name + "_pad")
            
                IR_node.attr["padding"].s = b"VALID"
                self._defuse_padding(source_node)
                del IR_node.input[:]
                IR_node.input.append(source_node.name + "_pad")

        # output shape
        self.set_output_shape(source_node, IR_node)

        # raise NotImplementedError


    def rename_SoftmaxOutput(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node, "Softmax")

        # input edge
        self._convert_inedge(source_node, IR_node)

        if "attr" in source_node.layer or "param" in source_node.layer:
            print("Warning: SoftmaxOutput attrs are not supported in IR.")
        
        # output shape
        self.set_output_shape(source_node, IR_node)

        # raise NotImplementedError


    def rename_softmax(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node, "Softmax")

        # input edge
        self._convert_inedge(source_node, IR_node)

        layer_attr = dict()
        if "attr" in source_node.layer:
            layer_attr = source_node.layer["attr"]
        elif "param" in source_node.layer:
            layer_attr = source_node.layer["param"]

        # dim
        IR_node.attr["dim"].i = int(layer_attr.get("axis", "-1"))

        # output shape
        self.set_output_shape(source_node, IR_node)

        # raise NotImplementedError


    # def rename_log_softmax(self, source_node):
    #   raise NotImplementedError("not support yet")


    # def rename_Correlation(self, source_node):
    #   raise NotImplementedError("not support yet")


    def rename_Deconvolution(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node, "Deconvolution")

        # input edge
        self._convert_inedge(source_node, IR_node)

        dim = 0
        layout = 'None'
        layer_attr = dict()
        if "attr" in source_node.layer:
            layer_attr = source_node.layer["attr"]
        elif "param" in source_node.layer:
            layer_attr = source_node.layer["param"]

        # padding
        if "pad" in layer_attr:
            # pad = self.check_pad_mode(source_node, IR_node)
            # if pad == "SAME":
            #     IR_node.attr["padding"].s = "SAME"
            # elif pad == "VALID":
            #     IR_node.attr["padding"].s = "VALID"
            # else:
            #     self._defuse_padding(source_node)
            #     del IR_node.input[:]
            #     IR_node.input.append(source_node.name + "_pad")
            IR_node.attr["padding"].s = b"VALID"
            self._defuse_padding(source_node)
            del IR_node.input[:]
            IR_node.input.append(source_node.name + "_pad")

        # output shape
        self.set_output_shape(source_node, IR_node)

        # filter
        assert "kernel" in layer_attr
        kernel = MXNetParser.str2intList(layer_attr.get("kernel"))
        dim = len(kernel)
        IR_node.attr["filter"].list.i.extend(kernel)

        in_channel = 0
        layout = layer_attr.get("layout")
        if layout == None or layout == 'None':
            if dim == 1:
                layout = "NCW"
            elif dim == 2:
                layout = "NCHW"
            elif dim == 3:
                layout = "NCDHW"

        if not self.data_format == layout:
            # print("Warning: Layer [{}] has changed model data format from [{}] to [{}]".format(source_node.name, self.data_format, layout))
            self.data_format = layout.encode()
        
        if self.weight_loaded:
            if layout in MXNetParser.channels_last:
                in_channel = self.weight_data.get(source_node.name + "_weight").shape[-1]
            else:
                in_channel = self.weight_data.get(source_node.name + "_weight").shape[1]

        assert "num_filter" in layer_attr
        out_channel = int(layer_attr.get("num_filter"))

        IR_node.attr["filter"].list.i.extend([out_channel, in_channel])

        # use_bias (no_bias default = False)
        IR_node.attr["use_bias"].b = not MXNetParser.str2bool(layer_attr.get("no_bias", "False"))

        # strides
        strides = layer_attr.get("strides")
        IR_node.attr["strides"].list.i.append(1)
        if not strides == None:
            IR_node.attr["strides"].list.i.extend(MXNetParser.str2intList(strides))
        IR_node.attr["strides"].list.i.append(1)

        # dilations
        dilate = layer_attr.get("dilate")
        IR_node.attr["dilation_rate"].list.i.append(1)
        if not dilate == None:
            IR_node.attr["dilation_rate"].list.i.extend(MXNetParser.str2intList(dilate))
        IR_node.attr["dilation_rate"].list.i.append(1)

        # data_format
        IR_node.attr["data_format"].s = layout

        # weights
        if self.weight_loaded:
            weight = self.weight_data.get(source_node.name + "_weight").asnumpy()
            if not layout in MXNetParser.channels_last:
                weight = MXNetParser.transpose(weight, dim)
            self.set_weight(source_node.name, "weights", weight)

            if IR_node.attr["use_bias"].b:
                self.set_weight(source_node.name, "bias", self.weight_data.get(source_node.name + "_bias").asnumpy())

        # raise NotImplementedError


    # def rename_RNN(self, source_node):
    #   raise NotImplementedError("RNN not support yet")


    def rename_Embedding(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node)

        # input edge
        self._convert_inedge(source_node, IR_node)

        # attr
        layer_attr = dict()
        if "attr" in source_node.layer:
            layer_attr = source_node.layer["attr"]
        elif "param" in source_node.layer:
            layer_attr = source_node.layer["param"]

        # input_dim
        IR_node.attr["input_dim"].i = int(layer_attr.get("input_dim"))

        # output_dim
        IR_node.attr["output_dim"].i = int(layer_attr.get("output_dim"))

        # dtype
        IR_node.attr["dtype"].type = MXNetParser.dtype_map[layer_attr.get("dtype", "float32")]

        # output shape
        self.set_output_shape(source_node, IR_node)

        # raise NotImplementedError


    # IR only support elu from {'elu', 'leaky', 'prelu', 'rrelu'}
    def rename_LeakyReLU(self, source_node):
        # judge whether meaningful
        assert "attr"
        layer_attr = dict()
        if "attr" in source_node.layer:
            layer_attr = source_node.layer["attr"]
        elif "param" in source_node.layer:
            if "act_type" in source_node.layer["attr"]:
                if not source_node.layer["attr"]["act_type"] == "elu":
                    print("Warning: Activation Type %s is not supported yet." % source_node.layer["attr"]["act_type"])
                    return

        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node, "Elu")

        # input edge
        self._convert_inedge(source_node, IR_node)

        # attr
        layer_attr = source_node.layer["attr"]

        # alpha [exp(x) - alpha], but mxnet attr slope [slope*(exp(x) - 1)] when x < 0
        if "slope" in layer_attr:
            print("Warning: Attribute Slope cannot convert into IR format.")
        # IR_node.attr["alpha"].f = float()

        # output shape
        self.set_output_shape(source_node, IR_node)

        # raise NotImplementedError("slope cannot convert to alpha")


    # def rename_InstanceNorm(self, source_node):
    #   raise NotImplementedError


    # def rename_L2Normalization(self, source_node):
    #   raise NotImplementedError


    def rename_LRN(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node, "LRN")

        # input edge
        self._convert_inedge(source_node, IR_node)

        layer_attr = dict()
        if "attr" in source_node.layer:
            layer_attr = source_node.layer["attr"]
        elif "param" in source_node.layer:
            layer_attr = source_node.layer["param"]

        # alpha
        IR_node.attr["alpha"].f = float(layer_attr.get("alpha", "0.0001"))
        # beta
        IR_node.attr["beta"].f = float(layer_attr.get("beta", "0.75"))
        # knorm
        IR_node.attr["k"].f = float(layer_attr.get("knorm", "2"))
        # nsize
        assert "nsize" in layer_attr
        IR_node.attr["size"].i = float(layer_attr["nsize"])

        # output shape
        self.set_output_shape(source_node, IR_node)



    # def rename_ROIPooling(self, source_node):
    #   raise NotImplementedError


    def rename_Dropout(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node, "Dropout")

        # input edge
        self._convert_inedge(source_node, IR_node)

        layer_attr = dict()
        if "attr" in source_node.layer:
            layer_attr = source_node.layer["attr"]
        elif "param" in source_node.layer:
            layer_attr = source_node.layer["param"]

        # keep_prob
        IR_node.attr["keep_prob"].f = float(layer_attr.get("p", "0.5"))

        # mode
        IR_node.attr["mode"].s = layer_attr.get("mode", "training").encode()

        # output shape
        self.set_output_shape(source_node, IR_node)

        # raise NotImplementedError


    """
    Here start with Symbol manipulation routines
    """

    # reverse cannot support yet
    def rename_reshape(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node, "Reshape")

        # input edge
        self._convert_inedge(source_node, IR_node)

        layer_attr = dict()
        if "attr" in source_node.layer:
            layer_attr = source_node.layer["attr"]
        elif "param" in source_node.layer:
            layer_attr = source_node.layer["param"]

        # old API target_shape not support yet
        shape = layer_attr.get("shape")
        if not shape == None:
            shape_list = MXNetParser.str2intList(shape)
            for param in shape_list:
                if param <= 0 and not param == -1:
                    print("Warning: special value %d for Reshape is not pre-defined in IR." % param)
            IR_node.attr["shape"].list.i.extend(shape_list)

        # output shape
        self.set_output_shape(source_node, IR_node)

        # raise NotImplementedError("adjust output shape")


    def rename_Flatten(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node, "Flatten")

        # input edge
        self._convert_inedge(source_node, IR_node)

        # output shape
        self.set_output_shape(source_node, IR_node)

        # IR_node.attr["data_format"].s = self.data_format

        # raise NotImplementedError


    @staticmethod
    def _convert_axis(IR_node, axis):
        ndim = len(IR_node.attr['_output_shapes'].list.shape[0].dim)
        if axis == 0:
            return 0
        elif axis == 1:
            return ndim - 1
        else:
            return axis - 1


    def rename_Concat(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node, "Concat")

        # output shape
        self.set_output_shape(source_node, IR_node)

        # input edge
        self._convert_inedge(source_node, IR_node)

        # attr
        layer_attr = dict()
        if "attr" in source_node.layer:
            layer_attr = source_node.layer["attr"]
        elif "param" in source_node.layer:
            layer_attr = source_node.layer["param"]

        # dim 
        IR_node.attr["axis"].i = MXNetParser._convert_axis(IR_node, int(layer_attr.get("dim", "1")))


    def rename_cast(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node, "Cast")

        # input edge
        self._convert_inedge(source_node, IR_node)

        # attr
        assert "attr" in source_node.layer or "param" in source_node.layer
        layer_attr = dict()
        if "attr" in source_node.layer:
            layer_attr = source_node.layer["attr"]
        else:
            layer_attr = source_node.layer["param"]

        # dtype
        IR_node.attr["dtype"].type = MXNetParser.dtype_map[layer_attr.get("dtype")]

        # output shape
        self.set_output_shape(source_node, IR_node)

        # raise NotImplementedError


    def rename_expand_dims(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node)

        # input edge
        self._convert_inedge(source_node, IR_node)

        # attr
        assert "attr" in source_node.layer or "param" in source_node.layer
        layer_attr = dict()
        if "attr" in source_node.layer:
            layer_attr = source_node.layer["attr"]
        else:
            layer_attr = source_node.layer["param"]

        # axis
        IR_node.attr["axis"].i = int(layer_attr.get("axis"))

        # output shape
        self.set_output_shape(source_node, IR_node)

        # raise NotImplementedError


    def rename_elemwise_add(self, source_node):
        self._convert_arithmetic(source_node, 'Add')
        

    def rename__Plus(self, source_node):
        self._convert_arithmetic(source_node, 'Add')


    def rename_broadcast_add(self, source_node):
        self._convert_arithmetic(source_node, 'Add')
        

    def rename_broadcast_mul(self, source_node):
        self._convert_arithmetic(source_node, 'Mul')
        

    def rename__mul(self, source_node):
        self._convert_arithmetic(source_node, 'Mul')


    def rename__copy(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node)

        # input edge
        self._convert_inedge(source_node, IR_node)
        
        # output shape
        self.set_output_shape(source_node, IR_node)

        # raise NotImplementedError("No matching IR api")
