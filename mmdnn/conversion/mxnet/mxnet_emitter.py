#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import os

import math
import mxnet as mx
import numpy as np
from mmdnn.conversion.common.IR.IR_graph import IRGraph, IRGraphNode
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.DataStructure.emitter import Emitter

class MXNetEmitter(Emitter):

    dtype_map = {
        graph_pb2.DT_FLOAT16    : "float16",
        graph_pb2.DT_FLOAT32    : "float32",
        graph_pb2.DT_FLOAT64    : "float64",
        graph_pb2.DT_INT32      : "int32",
        graph_pb2.DT_UINT8      : "uint8"
    }

    activation_map = {
        "relu"    : "Relu",
        "sigmoid" : "Sigmoid",
        "tanh"    : "Tanh",
        "elu"     : "Elu"
        # Not support yet 
        # "softrelu"  : "SoftReLU"
    }

    transpose_map = {
        1 : 2,
        2 : 3,
       -1 : 1
    }
    
    channels_last = ['NDHWC', 'NHWC']

    def __init__(self, model):
        from six import string_types as _string_types
        
        if isinstance(model, _string_types):
            network_path = model
            self.weight_loaded = False
        elif len(model) == 4:
            network_path = model[0]
            weight_path = model[1]
            self.input_shape = model[2]
            self.output_weights_file = model[3]
            self.weights = np.load(weight_path).item()
            self.weight_loaded = True
            self.output_weights = dict()
        else:
            raise ValueError("the # of input arguments [{}] is not supported" % len(model))

        self.IR_graph = IRGraph(network_path)
        self.IR_graph.build()
    

    def _gen_header(self):
        str = """import mxnet as mx
import numpy as np

# mxnet-cpu only support channel first, default convert the model and weight as channel first
"""
        return str


    def gen_codes(self, phase):
        self.IR_layer_map = dict()
        for layer in self.IR_graph.topological_sort:
            self.IR_layer_map[layer] = self.IR_graph.get_node(layer)
        header = self._gen_header()
        network_code = header + "def RefactorModel():\n"

        shape = dict()
        for layer in self.IR_graph.topological_sort:
            current_node = self.IR_graph.get_node(layer)
            node_type = current_node.type

            if len(current_node.in_edges) == 0:
                current_node.in_edges.append('data')
    
            if node_type.lower() in MXNetEmitter.activation_map:
                func = getattr(self, "emit_Activation")
                line = func(current_node, node_type.lower())
                network_code += "    " + line + "\n"
            elif hasattr(self, "emit_" + node_type):
                func = getattr(self, "emit_" + node_type)
                line = func(current_node)
                network_code += "    " + line + "\n"
            else:
                print("MXNet Emitter has not supported operator [%s]." % (node_type))
                self.emit_UNKNOWN(current_node)

            if node_type == "DataInput":
                cur_shape = list()
                first = True
                for dim in current_node.IR_layer.attr["shape"].shape.dim:
                    if dim.size == -1 and first:
                        cur_shape.append(1)
                        print("Detect input layer [{}] using infer batch size, set it as default value [1]".format(current_node.name))
                    else:
                        if dim.size == -1:
                            print("Warning: user should change input size manually")
                        cur_shape.append(dim.size)
                    first = False

                cur_shape.insert(1, cur_shape.pop())    
                shape[current_node.name] = ', '.join('%s' % i for i in cur_shape)

        # output_weights_file = raw_input("Please type the path you want to save your MXNet model weights: ")

        if self.weight_loaded:
            dirname = os.path.dirname(self.output_weights_file)
            if not os.path.exists(dirname):
                os.makedirs(self.output_weights_file)
            with open(self.output_weights_file, 'wb') as outfile:
                np.save(outfile, self.output_weights)

        comment = "\n    # if a GPU is available, change mx.cpu() to mx.gpu()"
        last_line = "{:<15} = mx.mod.Module(symbol = {}, context = mx.cpu(), data_names = ['{}'])".format(
            "model",
            ', '.join([self.IR_graph.get_node(name).real_variable_name for name in self.IR_graph.output_layers]),
            ', '.join([self.IR_graph.get_node(name).real_variable_name for name in self.IR_graph.input_layers]))
        
        network_code += "    " + comment + "\n"
        network_code += "    " + last_line + "\n"
        network_code += "    return model\n\n\n"
            
        weight_code = ""
        if not self.weight_loaded:
            weight_code += "# emitter does not detect any import weights, you may generate weights file manually\n"

        weight_code += self.gen_weight_code(shape, phase)

        main_code = "if __name__ == '__main__':\n    model = RefactorModel()\n"
        if self.weight_loaded:
            main_code += "    # remember to adjust params path\n    model = deploy_weight(model, '{}')\n".format(self.output_weights_file)

        if phase == 'train':
            train_code = """def train(model):
    import logging
    logging.getLogger().setLevel(logging.DEBUG)
    model.fit(train_iter, # train data
            eval_data = val_iter, # validation data
            optimizer = 'sgd', # Defaults to 'sgd'
            optimizer_params = {'learning_rate':0.01}, # use fixed learning rate
            eval_metric = 'acc', # report accuracy during training, other possible predefined metrics are: 'ce', 'f1', 'mae', 'mse', 'rmse', 'top_k_accuracy'
            batch_end_callback = mx.callback.Speedometer(batch_size, 100), # output progress for each 100 data batches
            num_epoch = 10) # train for at most 10 dataset passes\n\n
"""
            code = network_code + weight_code + train_code + main_code
        else:
            test_code = """import matplotlib.pyplot as plt
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])


def get_image(url, show = False):
    import cv2
    # download and show the image
    fname = mx.test_utils.download(url)
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    if show:
        plt.imshow(img)
        plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img


def predict(model, labels, url):
    # to show the image, change the argument show into True
    img = get_image(url, show = False)
    # compute the predict probabilities
    model.forward(Batch([mx.nd.array(img)]))
    prob = model.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    for i in a[0:5]:
        print('prbability = %f, class = %s' %(prob[i], labels[i]))\n\n
"""

            main_code += """
    # # call function predict
    # with open('synset.txt', 'r') as f:
    #     labels = [l.rstrip() for l in f]
    # predict(model, labels, 'http://writm.com/wp-content/uploads/2016/08/Cat-hd-wallpapers.jpg')    
"""

            code = network_code + weight_code + test_code + main_code

        return code
    

    def gen_weight_code(self, shape, phase):
        if len(shape) == 0:
            # var = raw_input("Input layer not detected, please type data shape manually(i.e. X, X, X, X): ")
            shape['data'] = ', '.join('%s' % i for i in self.input_shape)
        str = "def deploy_weight(model, weight_file):\n"
        str += """
    if weight_file == None:
        return
    
    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    arg_params = dict()
    aux_params = dict()
    for weight_name, weight_data in weights_dict.items():
        weight_name = str(weight_name)
        if "moving" in weight_name:
            aux_params[weight_name] = mx.nd.array(weight_data)
        else:
            arg_params[weight_name] = mx.nd.array(weight_data)

"""
        if phase == 'train':
            str += "    model.bind(for_training = True, data_shapes = ["
        else:
            str += "    model.bind(for_training = False, data_shapes = ["
        first = True
        for k, v in shape.items():          
            if not first:
                str += ", "
            str += "('" + k + "', " + "(" + v + "))"
            first = False
        str += "])\n"
        str += "    model.set_params(arg_params = arg_params, aux_params = aux_params, allow_missing = True)\n\n    return model\n\n\n"
        return str
        # raise NotImplementedError


    @staticmethod
    def calculate_same_pad(data_shape, kernel, stride):
        # same_pad = int(math.ceil(float(data_shape) / float(stride)))
        # valid_pad = int(math.ceil(float(data_shape - kernel + 1) / float(stride)))
        # # if (same_pad - valid_pad) % 2 == 0:
        # #     return True, (same_pad - valid_pad)
        # # else:
        # #     return False, (same_pad - valid_pad)
        # return (same_pad - valid_pad)
        # # raise NotImplementedError
        if (data_shape % stride == 0):
            pad = max(kernel - stride, 0)
        else:
            pad = max(kernel - (data_shape % stride), 0)
        if pad % 2 == 0:
            return False, pad
        else:
            return True, pad
        # raise NotImplementedError

    
    @staticmethod
    def transfer_pad(pad_list):
        # if len(stride) == 0:
        #     stride = list([1] * len(kernel))
        # if mode == b'SAME':
        #     # print(data_shape, kernel, stride)
        #     defuse_pad = False
        #     ret = list()
        #     for i in range(len(kernel)):
        #         defuse_pad, same_pad = MXNetEmitter.calculate_same_pad(data_shape[i+1], kernel[i], stride[i])
        #         ret.append(same_pad)
        #     if defuse_pad:
        #         tmp = list([0, 0, 0, 0])
        #         for e in ret:
        #             tmp.extend([int(e / 2), int(e / 2 + 1)])
        #         ret = tmp
        #     else:
        #         ret = [int(e / 2) for e in ret]
        #     return defuse_pad, ret
        # elif mode == b'VALID':
        #     return False, list([0]* len(kernel))
        # else:
        #     raise ValueError("Padding algorithm [{}] is not supported" % mode)
        defuse_pad = False
        pad = list()

        assert len(pad_list) % 2 == 0
        mid = int(len(pad_list)/2)
        pad_first = pad_list[2:mid]
        pad_second = pad_list[mid:-2]

        for i in range(0, mid-2):
            if not pad_first[i] == pad_second[i]:
                defuse_pad = True

        if defuse_pad:
            pad.extend([0] * 4)
            for i in range(0, mid-2):
                pad.extend([pad_first[i], pad_second[i]])
        else:
            pad = pad_first

        return defuse_pad, pad

        # raise NotImplementedError
    

    @staticmethod   
    def transpose(data, dim):
        if dim == 1:
            data = data.transpose((2, 1, 0))
        elif dim == 2:
            data = data.transpose((3, 2, 0, 1))
        elif dim == 3:
            data = data.transpose((4, 3, 0, 1, 2))
        else:
            raise ValueError("The weight of dim {} cannot transpose" % dim)

        return data


    def set_pad(self, IR_node, code, pad):
        code = "{:<15} = mx.sym.pad(data = {}, mode = 'constant', pad_width = ({}), constant_value = 0, name = '{}')".format(
                IR_node.variable_name + "_pad",                
                self.parent_variable_name(IR_node),                
                pad,
                IR_node.name + "_pad")                    

        for e in IR_node.in_edges:
            if e == 'data':
                continue
            self.IR_layer_map[e].out_edges = [x if not self.IR_layer_map[x].name == IR_node.variable_name else IR_node.variable_name + "_pad" for x in self.IR_layer_map[e].out_edges]

        return code     
        
        
    def emit_UNKNOWN(self, IR_node):
        print(IR_node.IR_layer.name)


    def emit_FullyConnected(self, IR_node):
        if self.weight_loaded:
            weight_dict = self.weights[IR_node.name]
            self.output_weights[IR_node.name + "_weight"] = weight_dict['weights'].transpose((1, 0))
        
        num_hidden = IR_node.IR_layer.attr["units"].i
        no_bias = not IR_node.IR_layer.attr["use_bias"].b
        if not no_bias and self.weight_loaded:
            self.output_weights[IR_node.name + "_bias"] = weight_dict['bias']
        
        code = "{:<15} = mx.sym.FullyConnected(data = {}, num_hidden = {}, no_bias = {}, name = '{}')".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                num_hidden,
                no_bias,
                IR_node.name)

        return code


    def emit_Convolution(self, IR_node):
        if self.weight_loaded:
            weight_dict = self.weights[IR_node.name]
            weights = weight_dict['weights']        
    
        dim = len(IR_node.IR_layer.attr["filter"].list.i) - 2

        kernel = list()
        for idx in range(0, dim):
            kernel.append(IR_node.IR_layer.attr["filter"].list.i[idx])
        
        stride = list()
        for e in IR_node.IR_layer.attr["strides"].list.i[1:-1]:
            stride.append(e)

        dilate = list()
        for e in IR_node.IR_layer.attr["dilation_rate"].list.i[1:-1]:
            dilate.append(e)
        dilate = ', '.join('%s' % i for i in dilate)

        defuse_pad = False
        pad = list()
        if "pads" in IR_node.IR_layer.attr:    
            output_shape = list()            
            for e in IR_node.IR_layer.attr["_output_shapes"].list.shape[0].dim:
                output_shape.append(e.size)

            # print("Warning: MXNet Convolution Layer pad does not match IR Convolution Layer pad")
            defuse_pad, pad = MXNetEmitter.transfer_pad(IR_node.IR_layer.attr["pads"].list.i)
        pad = ', '.join('%s' % i for i in pad)

        kernel = ', '.join('%s' % i for i in kernel)        
        stride = ', '.join('%s' % i for i in stride)
        
        num_filter = IR_node.IR_layer.attr["filter"].list.i[-1]
        no_bias = not IR_node.IR_layer.attr["use_bias"].b
        if not no_bias and self.weight_loaded:
            self.output_weights[IR_node.name + "_bias"] = weight_dict['bias']    
        
        layout = IR_node.IR_layer.attr["data_format"].s
        # if layout == '':
        #     if dim == 1:
        #         layout = 'NCW'
        #     elif dim == 2:
        #         layout = 'NHWC'
        #     elif dim == 3:
        #         layout = 'NDHWC'
        layout = 'NCHW'

        if self.weight_loaded:
            # if layout not in MXNetEmitter.channels_last:
            weights = MXNetEmitter.transpose(weights, dim)
            self.output_weights[IR_node.name + "_weight"] = weights

        code = ""
        if not defuse_pad:
            # code = "{:<15} = mx.sym.transpose(data = {}, axes = (0, 3, 1, 2))\n".format(IR_node.replace_scope(IR_node.name) + "_input", IR_node.replace_scope(IR_node.in_edges[0]))
            code += "{:<15} = mx.sym.Convolution(data = {}, kernel = ({}), stride = ({}), dilate = ({}), pad = ({}), num_filter = {}, no_bias = {}, layout = '{}', name = '{}')".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                kernel,
                stride,
                dilate,
                pad,
                num_filter,
                no_bias,
                layout,
                IR_node.name)
            # code += "    {:<15} = mx.sym.transpose(data = {}, axes = (0, 2, 3, 1))\n".format(IR_node.replace_scope(IR_node.name), IR_node.replace_scope(IR_node.name))
        else:
            # code = "{:<15} = mx.sym.transpose(data = {}, axes = (0, 3, 1, 2))\n".format(IR_node.replace_scope(IR_node.name) + "_input", IR_node.replace_scope(IR_node.in_edges[0]))
            code += self.set_pad(IR_node, code, pad)
            code += "\n    {:<15} = mx.sym.Convolution(data = {}, kernel = ({}), stride = ({}), dilate = ({}), num_filter = {}, no_bias = {}, layout = '{}', name = '{}')".format(
                IR_node.variable_name,
                IR_node.variable_name + "_pad",
                kernel,
                stride,
                dilate,
                num_filter,
                no_bias,
                layout,
                IR_node.name)
            # code += "    {:<15} = mx.sym.transpose(data = {}, axes = (0, 2, 3, 1))\n".format(IR_node.replace_scope(IR_node.name), IR_node.replace_scope(IR_node.name))
            
        return code        


    def emit_DataInput(self, IR_node):
        shape = list()
        shape.extend(IR_node.IR_layer.attr["shape"].list.i)

        code = "{:<15} = mx.sym.var('{}')".format(IR_node.variable_name, IR_node.name)
        return code


    # Add LeakyReLU Elu(slope not support)
    def emit_Activation(self, IR_node, act_type):

        act_type = act_type
        func_name = ""

        if act_type == "elu":
            func_name = "LeakyReLU"
        else:
            func_name = "Activation"

        code = "{:<15} = mx.sym.{}(data = {}, act_type = '{}', name = '{}')".format(
                IR_node.variable_name,
                func_name,
                self.parent_variable_name(IR_node),                
                act_type,
                IR_node.name)

        return code


    def emit_BatchNorm(self, IR_node):
        if self.weight_loaded:
            weight_dict = self.weights[IR_node.name]
        
        # axis = IR_node.IR_layer.attr["axis"].i
        axis = 1
        eps = IR_node.IR_layer.attr["epsilon"].f
        momentum = IR_node.IR_layer.attr["momentum"].f

        fix_gamma = not IR_node.IR_layer.attr["scale"].b
        
        if self.weight_loaded:
            if not fix_gamma:
                self.output_weights[IR_node.name + "_gamma"] = weight_dict['scale']
            self.output_weights[IR_node.name + "_beta"] = weight_dict['bias']
        
        # not supported yet
        use_global_stats = "False"
        if self.weight_loaded:
            self.output_weights[IR_node.name + "_moving_var"] = weight_dict['var']
            self.output_weights[IR_node.name + "_moving_mean"] = weight_dict['mean']

        code = "{:<15} = mx.sym.BatchNorm(data = {}, axis = {}, eps = {}, momentum = {}, fix_gamma = {}, use_global_stats = {}, name = '{}')".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),                
                axis,
                eps,
                momentum,
                fix_gamma,
                use_global_stats,
                IR_node.name)

        return code


    def emit_Pool(self, IR_node):

        global_pool = IR_node.IR_layer.attr["global_pooling"].b

        kernel = list()
        if global_pool:
            kernel = [1] * (len(IR_node.IR_layer.attr["strides"].list.i) - 2)
        else:
            for e in IR_node.IR_layer.attr["window_shape"].list.i[1:-1]:
                kernel.append(e)

        pool_type = IR_node.IR_layer.attr["pooling_type"].s.lower().decode()

        stride = list()
        for e in IR_node.IR_layer.attr["strides"].list.i[1:-1]:
            stride.append(e)

        defuse_pad = False
        pad = list()
        if "pads" in IR_node.IR_layer.attr:
            output_shape = list()
            for e in IR_node.IR_layer.attr["_output_shapes"].list.shape[0].dim:
                output_shape.append(e.size)
        
            # print("Warning: MXNet Pooling Layer pad does not match IR Pooling Layer pad")
            defuse_pad, pad = MXNetEmitter.transfer_pad(IR_node.IR_layer.attr["pads"].list.i)
        pad = ', '.join('%s' % i for i in pad)

        kernel = ', '.join('%s' % i for i in kernel)
        stride = ', '.join('%s' % i for i in stride)

        code = ""
        if not defuse_pad:
            # code = "{:<15} = mx.sym.transpose(data = {}, axes = (0, 3, 1, 2))\n".format(IR_node.replace_scope(IR_node.name) + "_input", IR_node.replace_scope(IR_node.in_edges[0]))
            code += "{:<15} = mx.sym.Pooling(data = {}, global_pool = {}, kernel = ({}), pool_type = '{}', stride = ({}), pad = ({}), name = '{}')".format(
                    IR_node.variable_name,                    
                    self.parent_variable_name(IR_node),
                    global_pool,
                    kernel,
                    pool_type,
                    stride,
                    pad,
                    IR_node.name)
            # code += "    {:<15} = mx.sym.transpose(data = {}, axes = (0, 2, 3, 1))\n".format(IR_node.replace_scope(IR_node.name), IR_node.replace_scope(IR_node.name))
        else:
            # code = "{:<15} = mx.sym.transpose(data = {}, axes = (0, 3, 1, 2))\n".format(IR_node.replace_scope(IR_node.name) + "_input", IR_node.replace_scope(IR_node.in_edges[0]))
            code += self.set_pad(IR_node, code, pad)
            code += "\n    {:<15} = mx.sym.Pooling(data = {}, global_pool = {}, kernel = ({}), pool_type = '{}', stride = ({}), name = '{}')". format(
                    IR_node.variable_name,
                    IR_node.variable_name + "_pad",
                    global_pool,
                    kernel,
                    pool_type,
                    stride,
                    IR_node.name)
            # code += "    {:<15} = mx.sym.transpose(data = {}, axes = (0, 2, 3, 1))\n".format(IR_node.replace_scope(IR_node.name), IR_node.replace_scope(IR_node.name))

        return code        


    def emit_SoftmaxOutput(self, IR_node):

        code = "{:<15} = mx.sym.SoftmaxOutput(data = {}, name = 'softmax')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node)
        )

        return code


    def emit_Softmax(self, IR_node):

        code = ""

        if len(IR_node.out_edges) == 0:
            code = "{:<15} = mx.sym.SoftmaxOutput(data = {}, name = 'softmax')".format(
                    IR_node.variable_name,
                    self.parent_variable_name(IR_node))
        else:
            axis = IR_node.IR_layer.attr["dim"].i
            code = "{:<15} = mx.sym.softmax(data = {}, axis = {}, name = '{}')".format(
                    IR_node.variable_name,
                    self.parent_variable_name(IR_node),
                    axis,
                    IR_node.name)

        return code


    def emit_Squeeze(self, IR_node):
        return self.emit_Flatten(IR_node)


    def emit_Deconvolution(self, IR_node):
        if self.weight_loaded:
            weight_dict = self.weights[IR_node.name]
            weights = weight_dict['weights']
        
        dim = len(IR_node.IR_layer.attr["filter"].list.i) - 2

        kernel = list()
        for idx in range(0, dim):
            kernel.append(IR_node.IR_layer.attr["filter"].list.i[idx])

        stride = list()
        for e in IR_node.IR_layer.attr["strides"].list.i[1:-1]:
            stride.append(e)

        dilate = list()
        for e in IR_node.IR_layer.attr["dilation_rate"].list.i[1:-1]:
            dilate.append(e)
        dilate = ', '.join('%s' % i for i in dilate)

        defuse_pad = False
        pad = list()
        if "pads" in IR_node.IR_layer.attr:
            output_shape = list()
            for e in IR_node.IR_layer.attr["_output_shapes"].list.shape[0].dim:
                output_shape.append(e.size)
        
            # print("Warning: MXNet Deconvolution Layer pad does not match IR Deconvolution Layer pad")
            defuse_pad, pad = MXNetEmitter.transfer_pad(IR_node.IR_layer.attr["pads"].list.i)
        pad = ', '.join('%s' % i for i in pad)

        kernel = ', '.join('%s' % i for i in kernel)
        stride = ', '.join('%s' % i for i in stride)

        num_filter = IR_node.IR_layer.attr["filter"].list.i[-2]
        no_bias = not IR_node.IR_layer.attr["use_bias"].b
        if not no_bias and self.weight_loaded:
            self.output_weights[IR_node.replace_scope(IR_node.name) + "_bias"] = weight_dict['bias']
        
        layout = IR_node.IR_layer.attr["data_format"].s
        # if layout == '':
        #     if dim == 1:
        #         layout = 'NCW'
        #     elif dim == 2:
        #         layout = 'NHWC'
        #     elif dim == 3:
        #         layout = 'NDHWC'
        layout = 'NCHW'

        if self.weight_loaded:
            # if layout not in MXNetEmitter.channels_last:
            weights = MXNetEmitter.transpose(weights, dim)
            self.output_weights[IR_node.replace_scope(IR_node.name) + "_weight"] = weights

        code = ""
        if not defuse_pad:
            code = "{:<15} = mx.sym.Deconvolution(data = {}, kernel = ({}), stride = ({}), dilate = ({}), pad = ({}), num_filter = {}, no_bias = {}, layout = '{}', name = '{}')".format(
                    IR_node.replace_scope(IR_node.name),
                    IR_node.replace_scope(IR_node.in_edges[0]),
                    kernel,
                    stride,
                    dilate,
                    pad,
                    num_filter,
                    no_bias,
                    layout,
                    IR_node.replace_scope(IR_node.name))
        else:
            code = self.set_pad(IR_node, code, pad)
            code += "\n    {:<15} = mx.sym.Deconvolution(data = {}, kernel = ({}), stride = ({}), dilate = ({}), num_filter = {}, no_bias = {}, layout = '{}', name = '{}')".format(
                    IR_node.replace_scope(IR_node.name), IR_node.replace_scope(IR_node.name) + "_pad", kernel, stride, dilate, num_filter, no_bias, layout, IR_node.replace_scope(IR_node.name))

        return code        


    def emit_Embedding(self, IR_node):
        
        input_dim = IR_node.IR_layer.attr["input_dim"].i
        output_dim = IR_node.IR_layer.attr["output_dim"].i
        dtype = MXNetEmitter.dtype_map.get(IR_node.layer.attr["dtype"].type, "float32")

        code = "{:<15} = mx.sym.Embedding(data = {}, input_dim = {}, output_dim = {}, dtype = {}, name = '{}')".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                input_dim,
                output_dim,
                dtype,
                IR_node.name)

        return code        


    # def emit_LeakyReLU(self, IR_node):
        
    #     # IR only support Elu, the same problem with func emit_Activation

    #     code = "{:<15} = mx.sym.LeakyReLU(data = {}, )".format()

    #     return code
    #     raise NotImplementedError


    def emit_Dropout(self, IR_node):
        p = IR_node.IR_layer.attr["keep_prob"].f
        mode = IR_node.IR_layer.attr["mode"].s.lower().decode() if 'mode' in IR_node.layer.attr else 'training'
        code = "{:<15} = mx.sym.Dropout(data = {}, p = {}, mode = '{}', name = '{}')".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                p,
                mode,
                IR_node.name)

        return code


    # reverse cannot support yet
    def emit_Reshape(self, IR_node):

        shape = list()
        for e in IR_node.IR_layer.attr["shape"].list.i:
            shape.append(e)        
        shape = ', '.join('%s' % i for i in shape)
        reverse = False

        code = "{:<15} = mx.sym.reshape(data = {}, shape = ({}), reverse = {}, name = '{}')".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                shape,
                reverse,
                IR_node.name)

        return code


    def emit_Flatten(self, IR_node):

        # if "data_format" in IR_node.IR_layer.attr:
        #     data_format = IR_node.IR_layer.attr["data_format"].s
        # else:
        #     data_format = "NHWC"
        #     print("set the conv format before flatten as default value NHWC")

        # if data_format in MXNetEmitter.channels_last:
        code = "{:<15} = mx.sym.transpose(data = {}, axes = (0, 2, 3, 1))\n".format("trans", self.parent_variable_name(IR_node))
        code += "    {:<15} = mx.sym.flatten(data = {}, name = '{}')".format(IR_node.variable_name, "trans", IR_node.name)
        # else:
        #     code += "{:<15} = mx.sym.flatten(data = {}, name = '{}')".format(
        #             IR_node.replace_scope(IR_node.name),
        #             IR_node.replace_scope(IR_node.in_edges[0]),
        #             IR_node.replace_scope(IR_node.name))

        return code


    @staticmethod
    def _convert_axis(IR_node, axis):        
        ndim = len(IR_node.layer.attr['_output_shapes'].list.shape[0].dim)
        if axis == 0:
            return 0
        elif axis == ndim - 1:
            return 1
        else:
            return axis + 1

    
    def emit_Concat(self, IR_node):
        dim = MXNetEmitter._convert_axis(IR_node, IR_node.IR_layer.attr["axis"].i)
        code = "{:<15} = mx.sym.concat({}, dim = {}, name = '{}')".format(
                IR_node.variable_name,
                ', '.join(self.IR_graph.get_node(s).real_variable_name for s in IR_node.in_edges),
                dim,
                IR_node.name)

        return code


    def emit_Cast(self, IR_node):

        dtype = IR_node.IR_layer.attr["dtype"].type

        code = "{:<15} = mx.sym.cast(data = {}, dtype = {}, name = '{}')".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),                
                dtype,
                IR_node.name)

        return code


    def emit_Expand_dims(self, IR_node):
        
        axis = IR_node.IR_layer.attr["axis"].i

        code = "{:<15} = mx.sym.expand_dims(data = {}, axis = {}, name = '{}')".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                axis,
                IR_node.name)

        return code


    def emit_Pad(self, IR_node):
        mode = IR_node.IR_layer.attr["mode"].s.lower().decode()
        pad_width = list()
        pad_width.extend([0, 0, 0, 0])
        for e in IR_node.IR_layer.attr["paddings"].list.i[2:-2]:
            pad_width.append(e)

        # if not pad_width[2] == 0 or not pad_width[3] == 0:
        #     print("Warning: please check padding layer manually")

        pad_width = ', '.join('%s' % i for i in pad_width)

        code = "{:<15} = mx.sym.pad(data = {}, mode = '{}', pad_width = ({}), name = '{}')".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                mode,
                pad_width,
                IR_node.name)

        return code


    def emit_Add(self, IR_node):
        code = "{:<15} = mx.sym.broadcast_add({}, {})".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                self.parent_variable_name(IR_node, [1]))

        return code


    def emit_Mul(self, IR_node):

        code = "{:<15} = mx.sym.broadcast_mul({}, {})".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                self.parent_variable_name(IR_node, [1]))

        return code


    def emit_ReduceMean(self, IR_node):
        axes = IR_node.layer.attr['axes'].list.i[:]
        axes = ','.join('%s' % MXNetEmitter.transpose_map[i] for i in axes)
        
        code = "{:<15} = mx.sym.mean(data = {}, axis = ({}), keepdims = {})".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                axes,
                IR_node.layer.attr['keepdims'].b)

        return code

 
    def emit_LRN(self, IR_node):
        code = "{:<15} = mx.sym.LRN(data = {}, alpha = {}, beta = {}, knorm = {}, nsize = {}, name = '{}')".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),                
                IR_node.layer.attr['alpha'].f,
                IR_node.layer.attr['beta'].f,
                IR_node.layer.attr['k'].f,
                IR_node.layer.attr['size'].i * 2 - 1,
                IR_node.name)

        return code