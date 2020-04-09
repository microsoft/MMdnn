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
from mmdnn.conversion.common.utils import *
from mmdnn.conversion.rewriter.folder import Folder

class MXNetEmitter(Emitter):

    dtype_map = {
        graph_pb2.DT_FLOAT16    : "float16",
        graph_pb2.DT_FLOAT32    : "float32",
        graph_pb2.DT_FLOAT64    : "float64",
        graph_pb2.DT_INT32      : "int32",
        graph_pb2.DT_UINT8      : "uint8"
    }

    activation_map = {
        "relu"      : "Relu",
        "sigmoid"   : "Sigmoid",
        "tanh"      : "Tanh",
        "elu"       : "Elu"
    }

    transpose_map = {
        1 : 2,
        2 : 3,
       -1 : 1
    }

    naive_scope_pattern = []

    channels_last = ['NDHWC', 'NHWC']

    def __init__(self, model):
        super(MXNetEmitter, self).__init__()
        from six import string_types as _string_types

        if isinstance(model, _string_types):
            network_path = model
            self.weight_loaded = False
        elif len(model) == 3:
            network_path = model[0]
            weight_path = model[1]
            self.output_weights_file = model[2]
            self.output_weights = dict()
            self._load_weights(weight_path)
            self.weights = self.weights_dict
        else:
            raise ValueError("the # of input arguments [{}] is not supported" % len(model))

        self.IR_graph = IRGraph(network_path)
        self.IR_graph.build()

        folder = Folder(self.IR_graph, self.weights)
        folder.fold()

    @property
    def header_code(self):
        return """import mxnet as mx
import numpy as np
import math

# mxnet-cpu only support channel first, default convert the model and weight as channel first

def RefactorModel():
"""


    def gen_code(self, phase):
        self.IR_layer_map = dict()
        self.add_body(0, self.header_code)
        for layer in self.IR_graph.topological_sort:
            self.IR_layer_map[layer] = self.IR_graph.get_node(layer)

        shape = dict()
        for layer in self.IR_graph.topological_sort:
            current_node = self.IR_graph.get_node(layer)
            node_type = current_node.type


            if len(current_node.in_edges) == 0:
                current_node.in_edges.append('data')

            if node_type.lower() in MXNetEmitter.activation_map:
                func = getattr(self, "emit_Activation")
                line = func(current_node, MXNetEmitter.activation_map[node_type.lower()].lower())
                self.add_body(1, line)

            elif hasattr(self, "emit_" + node_type):
                func = getattr(self, "emit_" + node_type)
                line = func(current_node)
                if line != None:
                    self.add_body(1, line)
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
                self.input_name_shape = {current_node.name: tuple(cur_shape)}


        if self.weight_loaded:
            fullpath = os.path.abspath(self.output_weights_file)
            dirname = os.path.dirname(fullpath)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            with open(self.output_weights_file, 'wb') as outfile:
                np.save(outfile, self.output_weights)

        comment = "\n    # if a GPU is available, change mx.cpu() to mx.gpu()"
        # We use the real_name for specifying the input layer in data_names
        # since MXNet API wants the actual name of the layer. On the other
        # hand, the module API wants the last symbol in the symbol chain, so
        # for the output node we need to use the actual python variable name
        # of the last layer (real_variable_name).
        last_line = "{:<15} = mx.mod.Module(symbol = {}, context = mx.cpu(), data_names = ['{}'])".format(
            "model",
            ', '.join([self.IR_graph.get_node(name).real_variable_name for name in self.IR_graph.output_layers if self.IR_graph.get_node(name).type !='Pack' and self.IR_graph.get_node(name).type != 'Shape']),
            ', '.join([self.IR_graph.get_node(name).real_name for name in self.IR_graph.input_layers if self.IR_graph.get_node(name).type != 'Const']))

        self.add_body(1, comment)
        self.add_body(1, last_line)
        self.add_body(1, "return model")


        self.add_body(0, "")
        for code in self.layers_codes.values():
            self.add_body(0, code)

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
            code = self.body_code + weight_code + train_code + main_code
        else:
            test_code = """from collections import namedtuple
Batch = namedtuple('Batch', ['data'])


def get_image(url, show=False):
    import cv2
    # download and show the image
    fname = mx.test_utils.download(url)
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    if show:
        import matplotlib.pyplot as plt
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

            code = self.body_code + weight_code + test_code + main_code

        return code


    def gen_weight_code(self, shape, phase):
        str = "def deploy_weight(model, weight_file):\n"
        str += """
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

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
        str += "    model.set_params(arg_params = arg_params, aux_params = aux_params, allow_missing = True, allow_extra=True)\n\n    return model\n\n\n"
        return str


    @staticmethod
    def calculate_same_pad(data_shape, kernel, stride):
        if (data_shape % stride == 0):
            pad = max(kernel - stride, 0)
        else:
            pad = max(kernel - (data_shape % stride), 0)
        if pad % 2 == 0:
            return False, pad
        else:
            return True, pad


    @staticmethod
    def transfer_pad(pad_list):
        defuse_pad = False
        pad = list()

        assert len(pad_list) % 2 == 0
        mid = int(len(pad_list)/2)
        pad_first = pad_list[1:mid-1]
        pad_second = pad_list[mid+1:-1]

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


    def set_pad(self, IR_node, code, pad, _max_pool):
        if _max_pool:
            constant_value = "float('-inf')"
        else:
            constant_value = "0.0"

        code = "{:<15} = mx.sym.pad(data = {}, mode = 'constant', pad_width={}, constant_value = {}, name = '{}')".format(
                IR_node.variable_name + "_pad",
                self.parent_variable_name(IR_node),
                tuple(pad),
                constant_value,
                IR_node.name + "_pad")

        for e in IR_node.in_edges:
            e = e.split(':')[0]
            if e == 'data':
                continue
            self.IR_layer_map[e].out_edges = [x if not self.IR_layer_map[x.split(':')[0]].name == IR_node.variable_name else IR_node.variable_name + "_pad" for x in self.IR_layer_map[e].out_edges]

        return code


    def emit_UNKNOWN(self, IR_node):
        print(IR_node.name)


    def emit_FullyConnected(self, IR_node):
        if self.weight_loaded:
            weight_dict = self.weights[IR_node.name]
            parent = self.IR_graph.get_parent(IR_node.name, [0])
            while parent.type == "Flatten" or parent.type == 'Dropout':
                parent = self.IR_graph.get_parent(parent.name, [0])
            dim = len(parent.layer.attr['_output_shapes'].list.shape[0].dim)
            if dim > 2:
                original_dims = weight_dict['weights'].shape
                dims = [i.size for i in parent.layer.attr['_output_shapes'].list.shape[0].dim[1:]] + [-1]
                weight_dict['weights'] = np.reshape(weight_dict['weights'], dims)
                weight_dict['weights'] = np.transpose(weight_dict['weights'], [dim - 2] + list(range(0, dim - 2)) + [dim - 1])
                weight_dict['weights'] = np.reshape(weight_dict['weights'], original_dims)
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


    def _emit_convolution(self, IR_node, pattern):
        if self.weight_loaded:
            weight_dict = self.weights[IR_node.name]
            weights = weight_dict['weights']

        dim = len(IR_node.IR_layer.attr["kernel_shape"].list.i) - 2

        kernel = list()
        for idx in range(0, dim):
            kernel.append(IR_node.IR_layer.attr["kernel_shape"].list.i[idx])

        stride = list()
        for e in IR_node.IR_layer.attr["strides"].list.i[1:-1]:
            stride.append(e)

        dilate = list()
        for e in IR_node.IR_layer.attr["dilations"].list.i[1:-1]:
            dilate.append(e)
        if dilate == []: dilate = [1, 1]
        dilate = ', '.join('%s' % i for i in dilate)

        defuse_pad = False
        pad = list()
        if "pads" in IR_node.IR_layer.attr:
            output_shape = list()
            for e in IR_node.IR_layer.attr["_output_shapes"].list.shape[0].dim:
                output_shape.append(e.size)

            # print("Warning: MXNet Convolution Layer pad does not match IR Convolution Layer pad")
            defuse_pad, pad = MXNetEmitter.transfer_pad(IR_node.IR_layer.attr["pads"].list.i)

        num_filter = 0
        if pattern == "Deconvolution":
            num_filter = IR_node.IR_layer.attr["kernel_shape"].list.i[-2]
        else:
            num_filter = IR_node.IR_layer.attr["kernel_shape"].list.i[-1]

        use_bias = IR_node.get_attr('use_bias', False)
        if use_bias and self.weight_loaded:
            self.output_weights[IR_node.name + "_bias"] = weight_dict['bias']

        if pattern == "DepthwiseConv":
            num_group = IR_node.IR_layer.attr["kernel_shape"].list.i[-2]
            num_filter = num_filter * num_group
            pattern = "Convolution"
            if self.weight_loaded:
                weights = np.swapaxes(weights, -1, -2)

        else:
            num_group = IR_node.get_attr('group', 1)

        # layout = IR_node.IR_layer.attr["data_format"].s
        if dim == 1:
            layout = 'NCW'
        elif dim == 2:
            layout = 'NCHW'
        elif dim == 3:
            layout = 'NCDHW'

        if self.weight_loaded:
            # if layout not in MXNetEmitter.channels_last:
            weights = MXNetEmitter.transpose(weights, dim)
            self.output_weights[IR_node.name + "_weight"] = weights

        code = ""
        if not defuse_pad:
            code += "{:<15} = mx.sym.{}(data={}, kernel={}, stride={}, dilate = ({}), pad={}, num_filter = {}, num_group = {}, no_bias = {}, layout = '{}', name = '{}')".format(
                IR_node.variable_name,
                pattern,
                self.parent_variable_name(IR_node),
                tuple(kernel),
                tuple(stride),
                dilate,
                tuple(pad),
                num_filter,
                num_group,
                not use_bias,
                layout,
                IR_node.name)
        else:
            code += self.set_pad(IR_node, code, pad, False)
            code += "\n    {:<15} = mx.sym.{}(data={}, kernel={}, stride={}, dilate = ({}), num_filter = {}, num_group = {}, no_bias = {}, layout = '{}', name = '{}')".format(
                IR_node.variable_name,
                pattern,
                IR_node.variable_name + "_pad",
                tuple(kernel),
                tuple(stride),
                dilate,
                num_filter,
                num_group,
                not use_bias,
                layout,
                IR_node.name)

        return code


    def emit_Conv(self, IR_node):
        return self._emit_convolution(IR_node, "Convolution")


    def emit_DepthwiseConv(self, IR_node):
        return self._emit_convolution(IR_node, "DepthwiseConv")


    def emit_ConvTranspose(self, IR_node):
        return self._emit_convolution(IR_node, "Deconvolution")


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
        IR_node_after = self.IR_graph.get_son(IR_node.name, [0])
        if IR_node_after.type == 'Scale':
            if self.weight_loaded:
                weight_dict = self.weights[IR_node.name]
                weight_dict_scale = self.weights[IR_node_after.name]

            # axis = IR_node.IR_layer.attr["axis"].i
            axis = 1
            eps = IR_node.IR_layer.attr["epsilon"].f
            momentum = IR_node.IR_layer.attr["momentum"].f

            fix_gamma = not IR_node.IR_layer.attr["scale"].b

            if self.weight_loaded:
                if not fix_gamma:
                #     self.output_weights[IR_node.name + "_gamma"] = np.multiply(weight_dict['scale'], weight_dict_scale['scale'])
                # self.output_weights[IR_node.name + "_beta"] = np.multiply(weight_dict['bias'], weight_dict_scale['scale']) + weight_dict_scale['bias']
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

        else:
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

    def emit_Scale(self, IR_node):
        if self.weight_loaded:
            weight_dict = self.weights[IR_node.name]

        # axis = IR_node.IR_layer.attr["axis"].i
        axis = 1
        eps = 0.0
        momentum = 0.0

        fix_gamma = not IR_node.IR_layer.attr["scale"].b

        if self.weight_loaded:
            if not fix_gamma:
                self.output_weights[IR_node.name + "_gamma"] = weight_dict['scale']
            self.output_weights[IR_node.name + "_beta"] = weight_dict['bias']

        # not supported yet
        use_global_stats = "False"
        if self.weight_loaded:
            self.output_weights[IR_node.name + "_moving_var"] = weight_dict['scale_var']
            self.output_weights[IR_node.name + "_moving_mean"] = weight_dict['scale_mean']

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
            for e in IR_node.IR_layer.attr["kernel_shape"].list.i[1:-1]:
                kernel.append(e)

        pool_type = IR_node.get_attr('pooling_type').lower()

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
        code = ""
        if not defuse_pad:
            code += "{:<15} = mx.sym.Pooling(data = {}, global_pool = {}, kernel={}, pool_type = '{}', stride={}, pad={}, name = '{}')".format(
                    IR_node.variable_name,
                    self.parent_variable_name(IR_node),
                    global_pool,
                    tuple(kernel),
                    pool_type,
                    tuple(stride),
                    tuple(pad),
                    IR_node.name)
        else:
            code += self.set_pad(IR_node, code, pad, pool_type == "max")
            code += "\n    {:<15} = mx.sym.Pooling(data = {}, global_pool = {}, kernel={}, pool_type = '{}', stride={}, name = '{}')".format(
                    IR_node.variable_name,
                    IR_node.variable_name + "_pad",
                    global_pool,
                    tuple(kernel),
                    pool_type,
                    tuple(stride),
                    IR_node.name)

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


    # def emit_ConvTranspose(self, IR_node):
    #     if self.weight_loaded:
    #         weight_dict = self.weights[IR_node.name]
    #         weights = weight_dict['weights']

    #     dim = len(IR_node.IR_layer.attr["kernel_shape"].list.i) - 2

    #     kernel = list()
    #     for idx in range(0, dim):
    #         kernel.append(IR_node.IR_layer.attr["kernel_shape"].list.i[idx])

    #     stride = list()
    #     for e in IR_node.IR_layer.attr["strides"].list.i[1:-1]:
    #         stride.append(e)

    #     dilate = list()
    #     for e in IR_node.IR_layer.attr["dilations"].list.i[1:-1]:
    #         dilate.append(e)
    #     dilate = ', '.join('%s' % i for i in dilate)

    #     defuse_pad = False
    #     pad = list()
    #     if "pads" in IR_node.IR_layer.attr:
    #         output_shape = list()
    #         for e in IR_node.IR_layer.attr["_output_shapes"].list.shape[0].dim:
    #             output_shape.append(e.size)

    #         # print("Warning: MXNet Deconvolution Layer pad does not match IR Deconvolution Layer pad")
    #         defuse_pad, pad = MXNetEmitter.transfer_pad(IR_node.IR_layer.attr["pads"].list.i)
    #     pad = ', '.join('%s' % i for i in pad)

    #     kernel = ', '.join('%s' % i for i in kernel)
    #     stride = ', '.join('%s' % i for i in stride)

    #     num_filter = IR_node.IR_layer.attr["kernel_shape"].list.i[-2]
    #     no_bias = not IR_node.IR_layer.attr["use_bias"].b
    #     if not no_bias and self.weight_loaded:
    #         self.output_weights[IR_node.replace_scope(IR_node.name) + "_bias"] = weight_dict['bias']

    #     # layout = IR_node.IR_layer.attr["data_format"].s
    #     if dim == 1:
    #         layout = 'NCW'
    #     elif dim == 2:
    #         layout = 'NCHW'
    #     elif dim == 3:
    #         layout = 'NCDHW'

    #     if self.weight_loaded:
    #         # if layout not in MXNetEmitter.channels_last:
    #         weights = MXNetEmitter.transpose(weights, dim)
    #         self.output_weights[IR_node.replace_scope(IR_node.name) + "_weight"] = weights

    #     code = ""
    #     if not defuse_pad:
    #         code = "{:<15} = mx.sym.Deconvolution(data = {}, kernel = ({}), stride = ({}), dilate = ({}), pad = ({}), num_filter = {}, no_bias = {}, layout = '{}', name = '{}')".format(
    #                 IR_node.replace_scope(IR_node.name),
    #                 IR_node.replace_scope(IR_node.in_edges[0]),
    #                 kernel,
    #                 stride,
    #                 dilate,
    #                 pad,
    #                 num_filter,
    #                 no_bias,
    #                 layout,
    #                 IR_node.replace_scope(IR_node.name))
    #     else:
    #         code = self.set_pad(IR_node, code, pad)
    #         code += "\n    {:<15} = mx.sym.Deconvolution(data = {}, kernel = ({}), stride = ({}), dilate = ({}), num_filter = {}, no_bias = {}, layout = '{}', name = '{}')".format(
    #                 IR_node.replace_scope(IR_node.name), IR_node.replace_scope(IR_node.name) + "_pad", kernel, stride, dilate, num_filter, no_bias, layout, IR_node.replace_scope(IR_node.name))

    #     return code


    def emit_Embedding(self, IR_node):

        input_dim = IR_node.IR_layer.attr["input_dim"].i
        output_dim = IR_node.IR_layer.attr["output_dim"].i
        dtype = MXNetEmitter.dtype_map.get(IR_node.layer.attr["dtype"].type, "float32")

        weight_dict = self.weights[IR_node.name]

        if self.weight_loaded:
            self.output_weights[IR_node.name + "_weight"] = weight_dict['weights']

        code = "{:<15} = mx.sym.Embedding(data = {}, input_dim = {}, output_dim = {}, dtype = '{}', name = '{}')".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                input_dim,
                output_dim,
                dtype,
                IR_node.name)

        return code


    def emit_LeakyRelu(self, IR_node):
        alpha = IR_node.IR_layer.attr['alpha'].f
        code = "{:<15} = mx.sym.LeakyReLU(data = {}, slope = {}, name = '{}')".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                alpha,
                IR_node.name
        )
        return code

    def emit_PRelu(self, IR_node):
        slope = IR_node.get_attr('gamma')
        code = "{:<15} = mx.sym.LeakyReLU(data = {}, slope = {}, act_type = '{}', name = '{}')".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                slope,
                'prelu',
                IR_node.name
        )
        return code

    def emit_Elu(self, IR_node):
        alpha = IR_node.IR_layer.attr['alpha'].f
        code = "{:<15} = mx.sym.LeakyReLU(data = {}, slope = {}, act_type = {}, name = '{}')".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                alpha,
                'elu',
                IR_node.name
        )
        return code

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
        # code = "{:<15} = mx.sym.transpose(data = {}, axes = (0, 2, 3, 1))\n".format("trans", self.parent_variable_name(IR_node))
        code = "{:<15} = mx.sym.flatten(data = {}, name = '{}')".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                IR_node.name)

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
                ', '.join(self.parent_variable_name(IR_node, [idx]) for idx in range(len(IR_node.in_edges))),
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
        pad_width.extend([0]*4)
        padding = convert_onnx_pad_to_tf(IR_node.get_attr("pads"))[1:-1]
        for padding_pair in padding:
            pad_width.extend(padding_pair)

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
        output_name = IR_node.variable_name
        input_name = self.parent_variable_name(IR_node)
        IR_name = IR_node.name
        alpha = IR_node.get_attr('alpha')
        beta = IR_node.get_attr('beta')
        bias = IR_node.get_attr('bias')
        size = IR_node.get_attr('size')


        code = "{:<15} = mx.sym.LRN(data = {}, alpha = {}, beta = {}, knorm = {}, nsize = {}, name = '{}')".format(
                output_name,
                input_name,
                alpha,
                beta,
                bias,
                size,
                IR_name)

        return code

    def emit_Constant(self, IR_node):
        # save the constant into weight dict
        if IR_node.get_attr('value'):
            value = IR_node.get_attr('value')
        else:
            value = self.weights[IR_node.name]['value']
    
        if not isinstance(value, list):
            self.output_weights[IR_node.name + '_weight'] = [value] # mxnet's bug, it does not surpport scalar weight. 
            code = "{:<15} = mx.sym.var(name = '{}', shape=(1,))".format(IR_node.variable_name, IR_node.name+'_weight')
        else:
            shape = np.array(value).shape
            self.output_weights[IR_node.name + '_weight'] = value

            code = "{:<15} = mx.sym.var(name = '{}', shape={})".format(IR_node.variable_name, IR_node.name+'_weight', shape)

        return code

    def emit_Sub(self, IR_node):
        code = "{:<15} = mx.sym.broadcast_sub({}, {})".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                self.parent_variable_name(IR_node, [1]))

        return code


    def emit_Relu6(self, IR_node):
        codes = list()
        codes.append(self.emit_Activation(IR_node, 'relu'))
        old_name = IR_node.variable_name
        IR_node.real_name = IR_node.real_name + "_clip"
        codes.append("{:<15} = mx.sym.clip({}, a_min=0, a_max=6, name='{}')".format(
            IR_node.real_variable_name,
            old_name,
            IR_node.real_name))

        return codes


    def emit_Slice(self, IR_node):

        starts = IR_node.get_attr('starts')
        starts = [starts[0], starts[-1]] + starts[1:-1]
        ends = IR_node.get_attr('ends')
        ends = [ends[0], ends[-1]] + ends[1:-1]
        ends = [i if i else None for i in ends]
        strides = IR_node.get_attr('strides')
        if strides:
            strides = [strides[0], strides[-1]] + strides[1:-1]

        code =  "{:<15} = mx.sym.slice({}, begin={}, end={}, step={}, name='{}')".format(
            IR_node.real_variable_name,
            self.parent_variable_name(IR_node),
            starts,
            ends,
            strides,
            IR_node.name
        )
        return code

    def emit_Const(self, IR_node):
        pass

    def emit_Shape(self, IR_node):
        code = "{:<15} = mx.sym.var(init = mx.init.Constant({}.infer_shape({}={})[1][0]), name='{}')".format(
            IR_node.real_variable_name,
            self.parent_variable_name(IR_node),
            list(self.input_name_shape.keys())[0],
            list(self.input_name_shape.values())[0],
            IR_node.name
        )
        return code

    def emit_Pack(self, IR_node):
        pass

    def emit_Unsqueeze(self, IR_node):
        axis = IR_node.get_attr('axes')[0]
        code = "{:<15} = mx.sym.expand_dims(data = {}, axis = {}, name = '{}')".format(
                IR_node.variable_name,
                self.parent_variable_name(IR_node),
                axis,
                IR_node.name)

        return code

    def emit_Unstack(self, IR_node):
        squeeze_axis = axis = IR_node.get_attr('axis')
        num = IR_node.get_attr('num')
        if num is None:
            args_str = ""
            for input_name in self.IR_graph.input_layers:
                if self.IR_graph.get_node(input_name).type!='Const':
                    args_str += '{}={}, '.format(self.IR_graph.get_node(input_name).real_variable_name, self.data_input_shape[input_name])

            args_str = args_str[:-2]
            num_outputs = "{}.infer_shape({})[1][0][{}]".format(
                IR_node.variable_name,
                args_str,
                axis
            )
        else:
            num_outputs = num

        code = "{:<15} = mx.sym.split({}, num_outputs={}, axis={}, squeeze_axis={})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            num_outputs,
            axis,
            squeeze_axis
        )
        return code

    def emit_Fill(self, IR_node):
        value = IR_node.get_attr('value')
        code = "{:<15} = mx.sym.full({}, {})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            value
        )
        return code

    def emit_Split(self, IR_node):
        axis = IR_node.get_attr('axis')
        num_outputs = IR_node.get_attr('split')

        if isinstance(num_outputs, list):
            raise NotImplementedError()
        code = "{:<15} = mx.sym.split({}, num_outputs={}, axis={})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            num_outputs,
            axis)

        return code


    def emit_Sigmoid(self, IR_node):
        code = "{:<15} = mx.sym.sigmoid(data={}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.name
        )
        return code


    def emit_Tanh(self, IR_node):
        code = "{:<15} = mx.sym.tanh(data={}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            IR_node.name
        )
        return code


    def emit_Maxmum(self, IR_node):
        code = "{:<15} = mx.sym.maxmum({}, {}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            self.parent_variable_name(IR_node, [1]),
            IR_node.name
        )
        return code


    def emit_Minimum(self, IR_node):
        code = "{:<15} = mx.sym.minimum({}, {}, name='{}')".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            self.parent_variable_name(IR_node, [1]),
            IR_node.name
        )
        return code


    def emit_Scope(self, IR_node):
        import re
        pattern = IR_node.pattern
        
        if pattern not in self.naive_scope_pattern and re.sub(r'(_\d+)*$', '', IR_node.pattern) not in self.naive_scope_pattern:
            origi_pattern = re.sub(r'(_\d+)*$', '', IR_node.pattern)
            func = getattr(self, "_emit_" + origi_pattern)
            code = func(IR_node)
        else:
            code = "{:<15} = __{}({})".format(
                IR_node.real_variable_name,
                IR_node.pattern,
                ', '.join(self.parent_variable_name(IR_node, s) for s in IR_node.in_edges))
            self._gen_scope_code(IR_node)
        return code


    def _gen_scope_code(self, scope_node):

        def _get_weight_related_op_name(node):
            weight_related_ops = ['Constant', 'Conv', 'FullyConnected', 'BatchNorm']
            op_type = node.type
            if op_type in weight_related_ops:
                return op_type, node.name

        def _scope_func(params, code, return_var):
            code = """
    def __call__(self, {}):
{}
        return {}
    """.format(params, code, ', '.join(return_var))
            return code

        class_inits = dict()

        body_code = str()
        for node_name in scope_node.topology_list:
            node = self.IR_graph.get_node(node_name)
            node_type = node.type

            if hasattr(self, "emit_" + node_type):
                func = getattr(self, "emit_" + node_type)
                line = func(node)
                if line != None:
                    body_code += "        " + line + '\n'
                    inits = _get_weight_related_op_name(node)
                    if inits:
                        if class_inits.get(inits[0], None):
                            class_inits[inits[0]].append(inits[1])
                        else:
                            class_inits[inits[0]] = list([inits[1]])
            else:
                print("MXNetEmitter has not supported operator [%s]." % (node_type))
                self.emit_UNKNOWN(node)

        # param_code does not need parameter slice.
        param_code = ', '.join('%s'  %self.IR_graph.get_node(s).real_variable_name for s in scope_node.in_edges)
        function_code = _scope_func(param_code, body_code, scope_node.return_variables)

        return class_inits, function_code


    def _emit_gru_cell(self, IR_node):
        if not self.layers_codes.get(IR_node.pattern, None):
            class_inits, func_code = self._gen_scope_code(IR_node)
            variables, variable_codes, init_code, func_code = self.process_inits_func_code(class_inits, func_code)

            states = [self.IR_graph.get_node(s).real_variable_name for s in IR_node.in_edges]
            states.pop(0)
            states_code = ', '.join(states)

            class_code ='''
class _{}(mx.rnn.BaseRNNCell):
    def __init__(self, {}):

{}

{}

            '''.format(IR_node.pattern,
            ', '.join(variables),
            init_code,
            func_code)
            self.layers_codes[IR_node.pattern] = class_code

            if not hasattr(self, 'pattern_variables'):
                self.pattern_variables = {IR_node.pattern: variables}
            else:
                self.pattern_variables[IR_node.pattern] = variables

            code = variable_codes
            code.append("{:<15} = _{}({})({})".format(
                IR_node.real_variable_name,
                IR_node.pattern,
                ', '.join(variables),
                ', '.join(self.parent_variable_name(IR_node, s) for s in IR_node.in_edges)))
        else:
            code = "{:<15} = _{}({})({})".format(
                IR_node.real_variable_name,
                IR_node.pattern,
                ', '.join(self.pattern_variables[IR_node.pattern]),
                ', '.join(self.parent_variable_name(IR_node, s) for s in IR_node.in_edges))

        return code


    def _emit_h_zero(self, IR_node):
        code = "{:<15} = mx.sym.full((1, {}), {})".format(
            IR_node.variable_name,
            IR_node.get_attr('fill_size'),
            IR_node.get_attr('fill_value')
        )
        return code
    

    def _emit_lstm_cell(self, IR_node):

        if not self.layers_codes.get(IR_node.pattern, None):
            class_inits, func_code = self._gen_scope_code(IR_node)
            variables, variable_codes, init_code, func_code = self.process_inits_func_code(class_inits, func_code)

            states = [self.IR_graph.get_node(s).real_variable_name for s in IR_node.in_edges]
            states.pop(0)
            states_code = ', '.join(states)

            class_code ='''
class _{}(mx.rnn.BaseRNNCell):
    def __init__(self, {}):

{}

{}

            '''.format(IR_node.pattern,
            ', '.join(variables),
            init_code,
            func_code)
            self.layers_codes[IR_node.pattern] = class_code

            if not hasattr(self, 'pattern_variables'):
                self.pattern_variables = {IR_node.pattern: variables}
            else:
                self.pattern_variables[IR_node.pattern] = variables

            code = variable_codes
            code.append("{:<15} = _{}({})({})".format(
                IR_node.real_variable_name,
                IR_node.pattern,
                ', '.join(variables),
                ', '.join(self.parent_variable_name(IR_node, s) for s in IR_node.in_edges)))
        else:
            code = "{:<15} = _{}({})({})".format(
                IR_node.real_variable_name,
                IR_node.pattern,
                ', '.join(self.pattern_variables[IR_node.pattern]),
                ', '.join(self.parent_variable_name(IR_node, s) for s in IR_node.in_edges))

        return code


    def process_inits_func_code(self, class_inits, func_code):
        init_code = str()
        variables = list()
        variable_codes = list()
        for k, v in class_inits.items():
            if k == 'FullyConnected':
                for i, name in enumerate(class_inits[k]):
                    variable_name = self.IR_graph.get_node(name).variable_name
                    variables.append("W_" + variable_name)
                    variable_codes.append("W_{:<15} = mx.sym.var(name='{}_weight')".format(variable_name, name))
                    init_code += "        self.W_{} = W_{}\n".format(variable_name, variable_name)

                    if self.weight_loaded and self.weights[name].get('bias', None).any() != None:
                        variable_codes.append("B_{:<15} = mx.sym.var(name='{}_bias')".format(variable_name, name))
                        variables.append("B_" + variable_name)
                        init_code += "        self.B_{} = B_{}\n".format(variable_name, variable_name)
                        func_code = func_code.replace("name = '{}'".format(name), "name = '{}', weight = self.W_{}, bias = self.B_{}".format(name, variable_name, variable_name))
                    else:
                        func_code = func_code.replace("name = '{}'".format(name), "name = '{}', weight = self.W_{}".format(name, variable_name))
            elif k == 'Constant':
                for name in class_inits[k]:
                    variable_name = self.IR_graph.get_node(name.replace('_weight', '')).variable_name
                    variables.append(variable_name)
                    constant_line = self.emit_Constant(self.IR_graph.get_node(name.replace('_weight', '')))
                    variable_codes.append("{:<15} = {}".format(variable_name, '='.join(constant_line.split('=')[1:])))
                    init_code += "        self.{} = {}\n".format(variable_name, variable_name)
                    func_code = func_code.replace(constant_line, constant_line.split('=')[0] + ' = self.'+constant_line.split('=')[0])
            else:
                raise NotImplementedError

        return variables, variable_codes, init_code, func_code

