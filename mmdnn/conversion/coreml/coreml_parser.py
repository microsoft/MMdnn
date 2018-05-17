#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------


import os
from six import string_types as _string_types
import numpy as np
import math

from coremltools.models.neural_network import NeuralNetworkBuilder as _NeuralNetworkBuilder
from coremltools.models import datatypes
from coremltools.models import MLModel as _MLModel
from coremltools.models.utils import save_spec as _save_spec
from coremltools.models._infer_shapes_nn_mlmodel import infer_shapes
from coremltools.proto import Model_pb2 ,NeuralNetwork_pb2


from mmdnn.conversion.coreml.coreml_graph import CoremlGraph
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.DataStructure.parser import Parser
from mmdnn.conversion.common.utils import *


class CoremlParser(Parser):

    activation_map = {
        "ReLU"          : "Relu",
        "leakyReLU"     : "LeakyRelu",
        "linear"        : "linear",
        "thresholdedReLU" : "ThresholdedRelu",
        "PReLU"         : "PRelu",
        "tanh"          : "Tanh",
        "scaledTanh"    : "ScaledTanh",
        'sigmoid'       : "Sigmoid",
        "sigmoidHard"   : "SigmoidHard",
        "ELU"           : "Elu",
        'softplus'      : 'Softplus',
        'softsign'      : 'Softsign',
        'parametricSoftplus'    : "ParametricSoftplus"
    }




    def __init__(self, model):
        super(CoremlParser, self).__init__()

        # load model file into Coreml Graph
        if isinstance(model, _string_types):
            # model.encode() convert to str --- python2 may crash due to type 'unicode'
            model = _MLModel(model.encode())
            model = model.get_spec()
            self.weight_loaded = True
        else:
            assert False

        # Build Network Graph

        model_type = model.WhichOneof('Type')
        if model_type == 'neuralNetworkClassifier':
            CoremlParser.shape_dict = infer_shapes(model.neuralNetworkClassifier, model.description.input)
        elif model_type == 'neuralNetwork':
            CoremlParser.shape_dict = infer_shapes(model.neuralNetwork, model.description.input)
        elif model_type == 'neuralNetworkRegressor':
            CoremlParser.shape_dict = infer_shapes(model.neuralNetworkRegressor, model.description.input)
        else:
            assert False

        # self.data_format ? TODO
        self.data_format = 'channels_first'
        self.coreml_graph = CoremlGraph(model)
        self.coreml_graph.build()
        self.lambda_layer_count = 0


    def _load_model(self, model_network_path):
        """Load a Coreml model from disk

        Parameters
        ----------

        model_network_path: str
            Path where the model network path is (mlmodel file)

        Returns
        -------
        model: A coreml model
        """

        from coremltools.models import MLModel

        if os.path.isfile(model_network_path):
            # load the model network
            loaded_model_ml = MLModel(model_network_path)
            # convert to Model_pb2.Model
            loaded_model_pb = loaded_model_ml.get_spec()
            self.weight_loaded = True
            print("Network file [{}] is loaded successfully.".format(model_network_path))
        else:
            print("Warning: Weights File [{}] is not found.".format(model_network_path))

        return loaded_model_pb


    @property
    def src_graph(self):
        return self.coreml_graph



    def gen_IR(self):
        for i, layer in enumerate(self.coreml_graph.topological_sort):

            current_node = self.coreml_graph.get_node(layer)
            current_node_layer = current_node.layer

            # determine the type of the current_node
            node_type = current_node_layer.name

            if isinstance(current_node_layer, Model_pb2.FeatureDescription):
                self.rename_InputLayer(current_node)
            elif isinstance(current_node_layer, NeuralNetwork_pb2.NeuralNetworkLayer):
                if current_node_layer.HasField("convolution"):
                    self.rename_CONV2D(current_node)
                elif current_node_layer.HasField('batchnorm'):
                    self.rename_BatchNormalization(current_node)
                elif current_node_layer.HasField("scale"):
                    self.rename_scale(current_node)
                elif current_node_layer.HasField("pooling"):
                    self.rename_Pooling(current_node)
                elif current_node_layer.HasField("activation"):
                    self.rename_Activation(current_node)
                elif current_node_layer.HasField("softmax"):
                    self.rename_Softmax(current_node)
                elif current_node_layer.HasField("padding"):
                    self.rename_Padding(current_node)
                elif current_node_layer.HasField("add"):
                    self.rename_Add(current_node)
                elif current_node_layer.HasField("flatten"):
                    self.rename_Flatten(current_node)
                elif current_node_layer.HasField("innerProduct"):
                    self.rename_innerProduct(current_node)
                elif current_node_layer.HasField("concat"):
                    self.rename_Concatenate(current_node)
                else:
                    print("CoremlParser has not supported operator [{}]".format(node_type))
                    self.rename_UNKNOWN(current_node)
            else:
                assert False



    # staticmethods
    @staticmethod
    def _set_output_shape(source_node, IR_node):

        shape = graph_pb2.TensorShape()
        source_node_layer = source_node.layer

        layer_name = source_node_layer.output[0]

        shape_coreml = CoremlParser.shape_dict[layer_name]
        # (seq, batch, C, H, W)  & NHWC

        new_dim = shape.dim.add()
        if shape_coreml[1] == 1:
            new_dim.size = -1
        else:
            new_dim.size = shape_coreml[1]
        for index in [3, 4, 2]:
            new_dim = shape.dim.add()
            dim = shape_coreml[index]
            new_dim.size = dim if dim else -1

        IR_node.attr["_output_shapes"].list.shape.extend([shape])


    @staticmethod
    def _copy_and_repo(source_node, IR_node, new_op = None):
        source_node_layer = source_node.layer
        IR_node.name = source_node_layer.name

        if new_op:
            IR_node.op = new_op
        elif source_node_layer.HasField("convolution"):
            IR_node.op = "convolution"
        elif source_node_layer.HasField('batchnorm'):
            IR_node.op = "batchnorm"
        elif source_node_layer.HasField("scale"):
            IR_node.op = "scale"
        elif source_node_layer.HasField("pooling"):
            IR_node.op = "pooling"
        elif source_node_layer.HasField("activation"):
            IR_node.op = "activation"
        elif source_node_layer.HasField("softmax"):
            IR_node.op = "softmax"
        elif source_node_layer.HasField("padding"):
            IR_node.op = "padding"
        elif source_node_layer.HasField("add"):
            IR_node.op = "add"
        elif source_node_layer.HasField("flatten"):
            IR_node.op = "flatten"
        elif source_node_layer.HasField("innerProduct"):
            IR_node.op = "innerProduct"
        elif source_node_layer.HasField("concat"):
            IR_node.op = "concatenate"
        else:
            assert False

        #  TODO dtype_map
        if hasattr(source_node.layer, "dtype"):
            IR_node.attr["dtype"].type = CoremlParser.dtype_map[source_node.layer.dtype]

        CoremlParser._set_output_shape(source_node, IR_node)

    @staticmethod
    def _copy_shape(source_node, target_node):
        if hasattr(source_node, "output_shape"):
            for dim in source_node.output_shape:
                new_dim = target_node.attr['shape'].shape.dim.add()
                new_dim.size = -1 if dim == None else new_dim
        else:
            target_node.attr['shape'].shape.unknown_rank = True

    @staticmethod
    def _convert_dataformat(source_node, target_node):
        if source_node.coreml_layer.data_format == "channels_last":
            target_node.attr['data_format'].s = "NHWC"
        elif source_node.coreml_layer.data_format == 'channels_first':
            target_node.attr['data_format'].s = "NCHW"
        else:
            print("Warning: [%s] don't have data format info" % (source_node.coreml_layer.name))


###### convert methods

    # convolution
    def __convert_convolution(self, source_node, dim):

        IR_node = self.IR_graph.node.add()
        # input edge
        self.convert_inedge(source_node, IR_node)
        source_node_layer = source_node.layer
        source_node_conv = source_node_layer.convolution
        layer_name = source_node_layer.name.split('/')[-1]

        # important!
        if source_node_conv.HasField('weights'):
            # reshape the weight!
            [h , w , k , o] = list(source_node_conv.kernelSize) + [source_node_conv.kernelChannels , source_node_conv.outputChannels]
            # [2, 3, 0, 1]
            weights = np.array(source_node_conv.weights.floatValue, dtype=np.float32).reshape([o, k, h, w]).transpose([2, 3, 1, 0])



        kwargs = dict()
        kwargs['kernel_shape'] = list(source_node_conv.kernelSize) + [source_node_conv.kernelChannels, source_node_conv.outputChannels]

        # pads
        CoremlParser._convert_padding(source_node, IR_node)
        # use_bias
        kwargs['use_bias'] = source_node_conv.hasBias
        # isDeconvolution
        kwargs['isDeconvolution'] = source_node_conv.isDeconvolution
        # name, op
        if layer_name == 'sep':
            CoremlParser._copy_and_repo(source_node, IR_node, "Conv")
        elif layer_name == 'dw':
            CoremlParser._copy_and_repo(source_node, IR_node, "DepthwiseConv")
            weights = weights.transpose((0,1,3,2))
            kwargs['kernel_shape'] = list(source_node_conv.kernelSize) + [source_node_conv.outputChannels, source_node_conv.kernelChannels]


        else:
            if kwargs['isDeconvolution']:
                CoremlParser._copy_and_repo(source_node, IR_node, "ConvTranspose")
            else:
                CoremlParser._copy_and_repo(source_node, IR_node, "Conv")

        self.set_weight(source_node.name, 'weights',  weights)
        if source_node_layer.convolution.HasField('bias'):
            self.set_weight(source_node.name, 'bias', np.array(source_node_conv.bias.floatValue, dtype=np.float32))



        # kwargs['kernel_shape'] = weights.shape

        kwargs['group'] = source_node_conv.nGroups

        # strides
        # [1, sd, sh, sw, 1]
        kwargs['strides'] = [1] + list(source_node_conv.stride) + [1]

        dilation = list(source_node_conv.dilationFactor)
        if dilation == []:
            dilation = [1,1]
        kwargs['dilations'] = [1] + dilation + [1]


        assign_IRnode_values(IR_node, kwargs)


        # activation
        # TODO
        self._defuse_activation(source_node)



    @staticmethod
    def _convert_padding(source_node, IR_node):
        source_node_layer = source_node.layer

        if source_node_layer.HasField('convolution'):
            # padding in conv

            source_node_conv = source_node_layer.convolution


            if source_node_conv.HasField('valid'):
                # pad in IR is [x1_b, x2_b, ..., x1_e, x2_e, ...]

                dim = []
                for i in source_node_conv.valid.paddingAmounts.borderAmounts:
                    dim.extend([i.startEdgeSize, i.endEdgeSize])


                if dim == []:
                    assign_IRnode_values(IR_node, { 'auto_pad': 'VALID'})
                    pad_dim = [0] * 8
                    pad_dim = convert_tf_pad_to_onnx(pad_dim)

                    assign_IRnode_values(IR_node, { 'pads':  pad_dim})
                else:

                    # padding
                    pad_dim = [0, 0]

                    pad_dim.extend(dim)

                    pad_dim += [0, 0]

                    pad_dim = convert_tf_pad_to_onnx(pad_dim)


                    assign_IRnode_values(IR_node, { 'pads':  pad_dim})

            elif source_node_conv.HasField('same'):

                # compute padding for 'same'
                assign_IRnode_values(IR_node, {'auto_pad': "SAME"})


                kernel = list(source_node_conv.kernelSize)
                dilation = list(source_node_conv.dilationFactor)
                if dilation == []:
                    dilation = [1,1]
                stride = list(source_node_conv.stride)
                if stride == []:
                    stride = [1,1]

                kernel[0] = dilation[0] * ( kernel[0] -1 ) + 1
                kernel[1] = dilation[1] * ( kernel[1] -1 ) + 1


                if stride == [1,1]:

                    # https://discuss.mxnet.io/t/pooling-and-convolution-with-same-mode/528/3

                    p0 =  ( kernel[0] -1 ) // 2
                    p1 =  ( kernel[1] -1 ) // 2

                    if kernel[0] % 2 == 0:
                        p00 = p0
                        p01 = p0 + 1
                    else:
                        p00 = p0
                        p01 = p0

                    if kernel[1] % 2 == 0:
                        p10 = p1
                        p11 = p1 + 1
                    else:
                        p10 = p1
                        p11 = p1

                    pad_dim = [0, 0, p00, p01, p10, p11, 0, 0]


                    pad_dim = convert_tf_pad_to_onnx(pad_dim)

                    assign_IRnode_values(IR_node, { 'pads':  pad_dim})
                else:
                    # https://www.jianshu.com/p/05c4f1621c7e
                    pad_dim = [0, 0, 0, 0, 0, 0, 0, 0]

                    pad_dim = convert_tf_pad_to_onnx(pad_dim)

                    assign_IRnode_values(IR_node, { 'pads':  pad_dim})

            else:
                assert False

        elif source_node_layer.HasField('pooling'):
            # padding in pooling
            source_node_pool = source_node_layer.pooling
            if  source_node_pool.HasField('valid'):

                dim = []
                for i in source_node_pool.valid.paddingAmounts.borderAmounts:
                    dim.extend([i.startEdgeSize, i.endEdgeSize])


                if dim == []:
                    assign_IRnode_values(IR_node, { 'auto_pad': 'VALID'})
                    pad_dim = [0] * 8
                    pad_dim = convert_tf_pad_to_onnx(pad_dim)

                    assign_IRnode_values(IR_node, { 'pads':  pad_dim})
                else:
                    # padding
                    pad_dim = [0, 0]

                    pad_dim.extend(dim)

                    pad_dim += [0, 0]
                    pad_dim = convert_tf_pad_to_onnx(pad_dim)
                    assign_IRnode_values(IR_node, { 'pads':  pad_dim})


            elif source_node_pool.HasField('same'):

                assign_IRnode_values(IR_node, { 'auto_pad': 'SAME'})

                kernel = list(source_node_pool.kernelSize)
                stride = list(source_node_pool.stride)
                if stride == []:
                    stride = [1,1]


                if stride == [1,1]:
                    # https://discuss.mxnet.io/t/pooling-and-convolution-with-same-mode/528/3
                    p0 =  ( kernel[0] -1 ) // 2
                    p1 =  ( kernel[1] -1 ) // 2



                    if kernel[0] % 2 == 0:
                        p00 = p0
                        p01 = p0 + 1
                    else:
                        p00 = p0
                        p01 = p0

                    if kernel[1] % 2 == 0:
                        p10 = p1
                        p11 = p1 + 1
                    else:
                        p10 = p1
                        p11 = p1

                    pad_dim = [0, 0, p00, p01, p10, p11, 0, 0]



                    pad_dim = convert_tf_pad_to_onnx(pad_dim)

                    assign_IRnode_values(IR_node, { 'pads':  pad_dim})
                else:
                    # TODO
                    pad_dim = [0, 0, 0, 0, 0, 0, 0, 0]

                    pad_dim = convert_tf_pad_to_onnx(pad_dim)

                    assign_IRnode_values(IR_node, { 'pads':  pad_dim})

            elif source_node_pool.HasField('includeLastPixel'):

                # symmetric padding
                h, w = source_node_pool.includeLastPixel.paddingAmounts
                assign_IRnode_values(IR_node, { 'pads':  [ 0,h, h,0,0, w, w,0]})
            else:
                assert False

        else:
            assert False




    def _convert_merge(self, source_node, new_name = None):

        IR_node = self.IR_graph.node.add()

        # name, op
        CoremlParser._copy_and_repo(source_node, IR_node, new_name)

        # input edge
        self.convert_inedge(source_node, IR_node)

        # For concat axis
        # NO axis in coreml, so set the last axis
        IR_node.attr['axis'].i = len(CoremlParser.shape_dict[source_node.layer.output[0]])-1 -1
        # The first -1 means in coreml there is one-more axis,
        # The second -1 means the last axis

        return IR_node


    def _convert_padding_api(self, source_node, IR_node):
        # name, op
        CoremlParser._copy_and_repo(source_node, IR_node, "Pad")

        # input edge
        self.convert_inedge(source_node, IR_node)

        kwargs = dict()

        source_node_layer = source_node.layer
        source_node_pad = source_node_layer.padding

        if source_node_pad.HasField('constant'):
            kwargs['mode'] = 'CONSTANT'
        elif source_node_pad.HasField('reflection'):
            kwargs['mode'] = 'REFLECT'
        elif source_node_pad.HasField('replication'):
            kwargs['mode'] = 'SYMMETRIC'
        else:
            assert False


        dim = []
        for i in source_node_pad.paddingAmounts.borderAmounts:
            dim.extend([i.startEdgeSize, i.endEdgeSize])


        if dim == []:
            dim = [0,0,0,0]

        # padding
        kwargs['pads'] = [0, 0]

        kwargs['pads'].extend(dim)

        kwargs['pads'] += [0, 0]
        kwargs['pads'] = convert_tf_pad_to_onnx(kwargs['pads'])

        assign_IRnode_values(IR_node, kwargs)

    def _defuse_activation(self, source_node):
        # Future Module TODO
        pass
        return


##### rename methods


    def rename_UNKNOWN(self, source_node):
        print(source_node.layer.get_config())
        IR_node = self.IR_graph.node.add()
        CoremlParser._copy_and_repo(source, IR_node)
        self.convert_inedge(source_node, IR_node)



    def rename_Activation(self, coreml_node):
        IR_node = self.IR_graph.node.add()


        coreml_node_layer = coreml_node.layer
        coreml_node_activation = coreml_node_layer.activation

        # name, op
        for activation_name in self.activation_map.keys():
            if coreml_node_activation.HasField(activation_name):
                CoremlParser._copy_and_repo(coreml_node, IR_node, self.activation_map[activation_name])


        # activation type
        activation_type = coreml_node_activation.WhichOneof("NonlinearityType")


        if activation_type == 'leakyReLU':
            assign_IRnode_values(IR_node, {'alpha' : coreml_node_activation.leakyReLU.alpha})
        elif activation_type == 'PReLU':
            assign_IRnode_values(IR_node, {'gamma' : coreml_node_activation.PReLU.alpha})
        elif activation_type == 'ELU':
            assign_IRnode_values(IR_node, {'alpha' : coreml_node_activation.ELU.alpha})
        elif activation_type == 'thresholdedRelu':
            assign_IRnode_values(IR_node, {'alpha' : coreml_node_activation.thresholdedReLU.alpha})
        elif activation_type == 'scaledTanh':
            assign_IRnode_values(IR_node, {'alpha' : coreml_node_activation.scaledTanh.alpha})
            assign_IRnode_values(IR_node, {'beta' : coreml_node_activation.scaledTanh.beta})
        elif activation_type == 'linear':
            assign_IRnode_values(IR_node, {'alpha' : coreml_node_activation.linear.alpha})
            assign_IRnode_values(IR_node, {'beta' : coreml_node_activation.linear.beta})
        elif activation_type == 'sigmoidHard':
            assign_IRnode_values(IR_node, {'alpha' : coreml_node_activation.sigmoidHard.alpha})
            assign_IRnode_values(IR_node, {'beta' : coreml_node_activation.sigmoidHard.beta})
        elif activation_type == 'parametricSoftplus':
            assign_IRnode_values(IR_node, {'alpha' : coreml_node_activation.parametricSoftplus.alpha})
            assign_IRnode_values(IR_node, {'beta' : coreml_node_activation.parametricSoftplus.beta})
        # else:
            # assert False

        # input edge
        self.convert_inedge(coreml_node, IR_node)

    # Merge layers
    def rename_Add(self, source_node):
        self._convert_merge(source_node, 'Add')

    def rename_CONV2D(self, source_node):
        self.__convert_convolution(source_node, 2)


    def rename_InputLayer(self, source_node):
        # only for training
        IR_node = self.IR_graph.node.add()

        # name, op
        IR_node.name = source_node.name
        IR_node.op = "DataInput"
        graph_shape = graph_pb2.TensorShape()
        coreml_node_layer = source_node.layer

        new_dim = graph_shape.dim.add()
        new_dim.size = -1
        new_dim = graph_shape.dim.add()
        new_dim.size = coreml_node_layer.type.imageType.width
        new_dim = graph_shape.dim.add()
        new_dim.size = coreml_node_layer.type.imageType.height
        new_dim = graph_shape.dim.add()

        if coreml_node_layer.type.imageType.colorSpace == 10:
            new_dim.size = 2
        elif coreml_node_layer.type.imageType.colorSpace == 20:
            new_dim.size = 3
        elif coreml_node_layer.type.imageType.colorSpace == 30:
            new_dim.size = 3
        else:
            assert False
        IR_node.attr["_output_shapes"].list.shape.extend([graph_shape])



        # input edge
        self.convert_inedge(source_node, IR_node)




        # shape
        # NHWC channel last
        # in fact, here is NWHC
        new_dim = IR_node.attr['shape'].shape.dim.add()
        new_dim.size = -1
        new_dim = IR_node.attr['shape'].shape.dim.add()
        new_dim.size = coreml_node_layer.type.imageType.width
        new_dim = IR_node.attr['shape'].shape.dim.add()
        new_dim.size = coreml_node_layer.type.imageType.height
        new_dim = IR_node.attr['shape'].shape.dim.add()

        if coreml_node_layer.type.imageType.colorSpace == 10:
            new_dim.size = 2
        elif coreml_node_layer.type.imageType.colorSpace == 20:
            new_dim.size = 3
        elif coreml_node_layer.type.imageType.colorSpace == 30:
            new_dim.size = 3
        else:
            assert False


    def rename_BatchNormalization(self, coreml_node):

        IR_node = self.IR_graph.node.add()

        coreml_node_layer = coreml_node.layer
        coreml_node_bn = coreml_node_layer.batchnorm


        # name, op
        CoremlParser._copy_and_repo(coreml_node, IR_node, "BatchNorm")

        # input edge
        self.convert_inedge(coreml_node, IR_node)


        # axis TODO
        # channels_first, then axis = 1
        IR_node.attr['axis'].i = -1

        # scale
        IR_node.attr['scale'].b = coreml_node_bn.HasField("gamma")

        # bias
        IR_node.attr['bias'].b = coreml_node_bn.HasField("beta")

        # epsilon
        IR_node.attr['epsilon'].f = coreml_node_bn.epsilon

        if IR_node.attr['scale'].b:
            self.set_weight(coreml_node_layer.name, "scale", np.array(coreml_node_bn.gamma.floatValue, dtype=np.float32))

        if IR_node.attr['bias'].b:
            self.set_weight(coreml_node_layer.name, "bias", np.array(coreml_node_bn.beta.floatValue, dtype=np.float32))



        gamma, beta = None, None
        if IR_node.attr['scale'].b:
            gamma = np.array(coreml_node_bn.gamma.floatValue, dtype=np.float32)
        if IR_node.attr['bias'].b:
            beta =  np.array(coreml_node_bn.beta.floatValue, dtype=np.float32)

        mean = np.array(coreml_node_bn.mean.floatValue)
        variance  =  np.array(coreml_node_bn.variance.floatValue)

        gamma = np.ones(mean.shape) if gamma is None else gamma
        beta = np.zeros(mean.shape) if beta is None else beta

        # compute adjusted parameters
        # Reference: parameter transformation https://github.com/apple/coremltools/issues/153
        f = 1.0 / np.sqrt(variance +  coreml_node_bn.epsilon)
        gamma1 = gamma*f
        beta1 = beta - gamma*mean*f
        mean[:] = 0.0 #mean
        variance[:] = 1.0 - .00001 #stddev

        # convert type because of tensorflow
        gamma1 = gamma1.astype(np.float32)
        beta1 = beta1.astype(np.float32)
        mean = mean.astype(np.float32)
        variance = variance.astype(np.float32)


        if IR_node.attr['scale'].b:
            self.set_weight(coreml_node_layer.name, "scale", gamma1)

        if IR_node.attr['bias'].b:
            self.set_weight(coreml_node_layer.name, "bias", beta1)

        # mean
        self.set_weight(coreml_node_layer.name, "mean", mean)

        # var
        self.set_weight(coreml_node_layer.name, "var", variance)

    def rename_scale(self, coreml_node):

        IR_node = self.IR_graph.node.add()

        coreml_node_layer = coreml_node.layer
        coreml_node_scale = coreml_node_layer.scale



        # name, op
        CoremlParser._copy_and_repo(coreml_node, IR_node, "Scale")

        # input edge
        self.convert_inedge(coreml_node, IR_node)

        # bias
        IR_node.attr['use_bias'].b = coreml_node_scale.hasBias


        self.set_weight(coreml_node_layer.name, "scale", np.array(coreml_node_scale.scale.floatValue))

        self.set_weight(coreml_node_layer.name, "shapeScale", coreml_node_scale.shapeScale[0])



        if IR_node.attr['use_bias'].b:
            self.set_weight(coreml_node_layer.name, "bias", np.array(coreml_node_scale.bias.floatValue))
            self.set_weight(coreml_node_layer.name, "shapeBias", coreml_node_scale.shapeBias[0])

    def rename_Pooling(self, coreml_node):


        IR_node = self.IR_graph.node.add()

        coreml_node_layer = coreml_node.layer
        coreml_node_pool = coreml_node_layer.pooling





        # name, op
        CoremlParser._copy_and_repo(coreml_node, IR_node, "Pool")

        # input edge
        self.convert_inedge(coreml_node, IR_node)

        kwargs = {}

        # MAX = 0, AVERAGE = 1, L2 = 2
        if coreml_node_pool.type == 0:
            kwargs['pooling_type'] = 'MAX'
        elif coreml_node_pool.type == 1:
            kwargs['pooling_type'] = 'AVG'
        elif coreml_node_pool.type == 2:
            kwargs['pooling_type'] = 'L2'



        is_global = coreml_node_pool.globalPooling


        if is_global:
            kwargs['global_pooling'] = True
            kwargs['global_pooling_coreml'] = True
            kwargs['shape_coreml'] = [self.shape_dict[coreml_node_layer.name][3], self.shape_dict[coreml_node_layer.name][4], self.shape_dict[coreml_node_layer.name][2] ]


        # padding
        self._convert_padding(coreml_node, IR_node)

        # strides
        # [1, sd, sh, sw, 1]
        kwargs['strides'] = [1] + list(coreml_node_pool.stride) + [1]


        # window_shape
        # [1, pd, ph, pw, 1]
        kwargs['kernel_shape'] = [1] + list(coreml_node_pool.kernelSize) + [1]




        assign_IRnode_values(IR_node, kwargs)




    def rename_Softmax(self, coreml_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        CoremlParser._copy_and_repo(coreml_node, IR_node, 'Softmax')

        # input edge
        self.convert_inedge(coreml_node, IR_node)

    def rename_Concatenate(self, source_node):
        IR_node = self._convert_merge(source_node, 'Concat')


    def rename_Flatten(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        CoremlParser._copy_and_repo(source_node, IR_node, 'Flatten')

        # input edge
        self.convert_inedge(source_node, IR_node)


    def rename_innerProduct(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        CoremlParser._copy_and_repo(source_node, IR_node, "FullyConnected")

        # input edge
        self.convert_inedge(source_node, IR_node)

        source_node_layer = source_node.layer
        source_node_inner = source_node_layer.innerProduct


        # units
        IR_node.attr['units'].i = source_node_inner.outputChannels

        # use_bias
        IR_node.attr['use_bias'].b = source_node_inner.hasBias

        # weights
        self.set_weight(source_node_layer.name, 'weights', np.array(source_node_inner.weights.floatValue).reshape( source_node_inner.outputChannels,  source_node_inner.inputChannels).transpose() )
        if IR_node.attr['use_bias'].b:
            self.set_weight(source_node_layer.name, 'bias', np.array(source_node_inner.bias.floatValue) )


    def rename_Padding(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._convert_padding_api(source_node, IR_node)

