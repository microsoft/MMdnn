
import os
import caffe
import numpy as np

from mmdnn.conversion.common.utils import *
from mmdnn.conversion.darknet.prototxt import *
from mmdnn.conversion.darknet.darknet_utils import *

from mmdnn.conversion.darknet.darknet_graph import DarknetGraph
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.DataStructure.parser import Parser
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType

class DarknetParser(Parser):

    dtype_map = {
        0 : graph_pb2.DT_UNDEFINED,
        np.float32 : graph_pb2.DT_FLOAT32,
        np.float64 : graph_pb2.DT_FLOAT64,
        3 : graph_pb2.DT_INT32,
        4 : graph_pb2.DT_UINT8,
        5 : graph_pb2.DT_INT16,
        6 : graph_pb2.DT_INT8,
        7 : graph_pb2.DT_STRING,
        9 : graph_pb2.DT_INT64
    }

    @property
    def src_graph(self):
        return self.dk_graph

    @staticmethod
    def load_weights(self, model, weightfile):
        # def load_conv_bn()
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        buf = np.fromfile(fp, dtype = np.float32)
        fp.close()
        layer_id = 1
        start = 0
        self.set_weight('test', 'weights', np.array([1,2,3,4,5]))
        # print(self.layer_map[''])
        # for block in model:
        #     print(block)
        #     if start >= buf.size:
        #         break
        #     if block['type'] == 'net':
        #         continue
        #     elif block['type'] == 'convolutional':
        #         batch_normalize = int(block['batch_normalize'])
        #         print(batch_normalize)
        #         # assert False
        #         if block.has_key('name'):
        #             conv_layer_name = block['name']
        #             bn_layer_name = '%s-bn' % block['name']
        #             scale_layer_name = '%s-scale' % block['name']
        #         else:
        #             conv_layer_name = 'layer%d-conv' % layer_id
        #             bn_layer_name = 'layer%d-bn' % layer_id
        #             scale_layer_name = 'layer%d-scale' % layer_id

        #         if batch_normalize:
        #             print(start)
        #             start = load_conv_bn(buf, start, )
        #             print(start)
        #             assert False
        #         else:
        #             start = load_conv(buf, start, )
        #         layer_id = layer_id+1
        #     elif block['type'] == 'connected':
        #         if block.has_key('name'):
        #             fc_layer_name = block['name']
        #         else:
        #             fc_layer_name = 'layer%d-fc' % layer_id
        #         start = load_fc2caffe(buf, start, params[fc_layer_name])
        #         layer_id = layer_id+1
        #     elif block['type'] == 'maxpool':
        #         layer_id = layer_id+1
        #     elif block['type'] == 'avgpool':
        #         layer_id = layer_id+1
        #     elif block['type'] == 'region':
        #         layer_id = layer_id + 1
        #     elif block['type'] == 'route':
        #         layer_id = layer_id + 1
        #     elif block['type'] == 'shortcut':
        #         layer_id = layer_id + 1
        #     elif block['type'] == 'softmax':
        #         layer_id = layer_id + 1
        #     elif block['type'] == 'cost':
        #         layer_id = layer_id + 1
        #     else:
        #         print('unknow layer type %s ' % block['type'])
        #         layer_id = layer_id + 1

    def __init__(self, model_config, weightfile):
        super(DarknetParser, self).__init__()

        if not os.path.exists(model_config):
            raise ValueError('Darknet model config [{}] can not be found!'.format(model_config))
        # model = _cntk.Function.load(model)
        # print(model_config)
        if weightfile:
            # print(weight)
            self.weight_loaded = True

        # net_info = cfg2prototxt(model_config)
        # print(net_info)
        # save_prototxt(net_info , 'resnet50.prototxt', region=False)
        # net = caffe.Net('resnet50.prototxt', caffe.TEST)
        # params = net.params
        # print(params)

        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.buf = np.fromfile(fp, dtype = np.float32)
        print(self.buf.size)

        fp.close()
        self.start = 1

        model = parse_cfg(model_config)
        # print(model)
        self.dk_graph = DarknetGraph(model)
        self.dk_graph.build()
        # print(self.dk_graph.get_node('layer87-concat'))
        # self.load_weights(self, model, weightfile)
        # print(self.weights)
        # assert False

    def gen_IR(self):
        # for layer in self.dk_graph.topological_sort:
        # for layer in self.dk_graph.layer_map:

        # load weight by original order
        for layer in self.dk_graph.original_list:
            # layer_map
            # print(layer)

            current_node = self.dk_graph.get_node(layer)
            node_type = current_node.type
            print(node_type)
            # print(current_node.layer)
            # continue
            if hasattr(self, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                func(current_node)
            else:
                self.rename_UNKNOWN(current_node)

    @staticmethod
    def _copy_and_reop(source_node, IR_node, new_op = None):
        if new_op == None: new_op = source_node.type
        IR_node.name = source_node.name
        IR_node.op = new_op

        # print(source_node.layer['attr'].keys())
        # assert False
        if '_output_shape' in source_node.layer['attr'].keys():
            # print("**********")
            output_list = source_node.layer['attr']['_output_shape']
            shape = graph_pb2.TensorShape()
            for dim in output_list:
                new_dim = shape.dim.add()
                if dim == None:
                    new_dim.size = -1
                else:
                    new_dim.size = dim

            IR_node.attr["_output_shape"].list.shape.extend([shape])

        if 'shape' in source_node.layer['attr'].keys():
            shape_list = source_node.layer['attr']['shape']
            if not output_list == None:
                for dim in shape_list:
                    new_dim = IR_node.attr["shape"].shape.dim.add()
                    if dim == None:
                        new_dim.size = -1
                    else:
                        new_dim.size = dim
            else:
                IR_node.attr["shape"].shape.unknown_rank = True


    def _convert_inedge(self, source_node, IR_node, start_idx = 0, end_idx = None):
        if end_idx == None: end_idx = len(source_node.in_edges)
        for idx in range(start_idx, end_idx):
            IR_node.input.append(self.src_graph.get_node(source_node.in_edges[idx]).real_name)

    def _convert_identity_operation(self, source_node, start_idx = 0, end_idx = None, new_op = None):
        IR_node = self.IR_graph.node.add()
        DarknetParser._copy_and_reop(source_node, IR_node, new_op)
        self._convert_inedge(source_node, IR_node, start_idx, end_idx)
        return IR_node

    def rename_UNKNOWN(self, source_node):
        print(source_node.layer)
        # print(source_node.layer['attr']['shape'])
        # print(source_node.in_edges)
        print("Darknet has not supported operator [%s] with name [%s]."
              % (source_node.type, source_node.name))
        assert False

    def rename_DataInput(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='DataInput')
        # print(IR_node)
        # assert False

    def rename_Conv(self, source_node):
        """
        weights: name_weights, name_bias
        """
        print(source_node.layer)
        print("//////////////",self.start)
        IR_node = self._convert_identity_operation(source_node, new_op='Conv')
        # print(IR_node)
        # assert False
        kwargs = {}

        # strides
        stride = source_node.get_attr('stride')
        kwargs['strides'] = [1, stride, stride, 1]

        innode = self.dk_graph.get_node(source_node.in_edges[0])
        input_shape = innode.get_attr('_output_shape')
        # print(input_shape)
        # kernel = source_node.get_attr('kernel')
        # print("kkkkkkkkk", kernel)

        # # assert False
        # kwargs['kernel_shape'] = kernel[-2:] + [kernel[1]] + [kernel[0]]
        kwargs['kernel_shape'] = source_node.get_attr('kernel')

        # padding
        if source_node.get_attr('pad'):
            kwargs['auto_pad'] = "SAME"
            padding = source_node.get_attr('padding')
            kwargs['pads'] = [0, padding, padding, 0, 0, padding, padding, 0]
        else:
            kwargs['auto_pad'] = "VALID"

        # only load weight conv

        print( "-------------------", source_node.get_attr('bias_term'))
        if source_node.get_attr('bias_term') == 'true':
            kwargs['use_bias'] = True
            print(kwargs['kernel_shape'])
            print(source_node.layer)

            kernel = kwargs['kernel_shape']
            kernel = np.zeros([kernel[-1], kernel[-2], kernel[0], kernel[1]])
            k_bias = np.zeros(kwargs['kernel_shape'][-1])
            print(kernel.shape)
            print(k_bias.shape)

            conv_name = source_node.name
            print(conv_name)
            b = np.reshape(self.buf[self.start:self.start+k_bias.size], k_bias.shape)
            self.start = self.start + k_bias.size
            self.set_weight(conv_name, 'bias', b)

            W = np.reshape(self.buf[self.start:self.start+kernel.size], kernel.shape)
            self.start = self.start + kernel.size
            W = np.transpose(W, (2, 3, 1, 0))
            self.set_weight(conv_name, 'weights', W)
        else:
            kwargs['use_bias'] = False


        assign_IRnode_values(IR_node, kwargs)
        # print(IR_node)
        # assert False
        # output[0] : B
        # self._get_bias(source_node, IR_node)

    def rename_BatchNorm(self, source_node):
        print("************", self.start)
        # print(source_node.layer)

        IR_node = self._convert_identity_operation(source_node, new_op='BatchNorm')
        kwargs = {}
        IR_node.attr['use_global_stats'].b = source_node.get_attr('use_global_stats')
        IR_node.attr['bias'].b = source_node.get_attr('use_global_stats')
        IR_node.attr['scale'].b = source_node.get_attr('use_global_stats')

        assign_IRnode_values(IR_node, kwargs)

        innode = self.dk_graph.get_node(source_node.in_edges[0])
        input_shape = innode.get_attr('_output_shape')
        kernel = innode.get_attr('kernel')
        print(kernel)
        kernel = np.zeros([kernel[-1], kernel[-2], kernel[0], kernel[1]])
        # print(input_shape)
        # print(kernel)
        # assert False

        # buf, start, scale_layer['name'], bn_layer['name'], conv_layer['name']
        bias = np.zeros(input_shape[-1])
        scale = np.zeros(input_shape[-1])
        mean = np.zeros(input_shape[-1])
        var = np.zeros(input_shape[-1])
        # assert False
        # kernel = np.zeros(kernel)
        print(bias.shape)
        print(scale.shape)
        print(mean.shape)
        print(var.shape)
        print(kernel.shape)
        # assert False

        bias_content = np.reshape(self.buf[self.start:self.start+bias.size], bias.shape)
        self.start = self.start + bias.size
        self.set_weight(source_node.name, 'bias', bias_content)

        scale_content = np.reshape(self.buf[self.start:self.start+scale.size], scale.shape)
        self.start = self.start + scale.size
        self.set_weight(source_node.name, 'scale', scale_content)


        mean_content = np.reshape(self.buf[self.start:self.start+mean.size], mean.shape)
        self.start = self.start + mean.size
        self.set_weight(source_node.name, 'mean', mean_content)


        var_content = np.reshape(self.buf[self.start:self.start+var.size], var.shape)
        self.start = self.start + var.size
        self.set_weight(source_node.name, 'var', var_content)


        W = np.reshape(self.buf[self.start:self.start+kernel.size], kernel.shape)
        self.start = self.start + kernel.size
        W = np.transpose(W, (2, 3, 1, 0))
        print(W.shape)
        # assert False
        self.set_weight(innode.name, 'weights', W)

        # print(IR_node)
        # assert False

    # no use
    def rename_ReLU(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='Relu')
        # print(IR_node)
        # assert False

    def rename_leakyReLU(self, source_node):
        # print(source_node.layer)
        kwargs = {}
        kwargs['alpha'] = float(source_node.get_attr('negative_slope'))
        IR_node = self._convert_identity_operation(source_node, new_op='LeakyRelu')
        assign_IRnode_values(IR_node, kwargs)
        # print(IR_node)
        # assert False


    def rename_Pooling(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='Pool')
        print(source_node.layer)
        kwargs = {}
        kernel = source_node.get_attr('kernel_size')
        kwargs['kernel_shape'] = [1, kernel, kernel, 1]
        stride = source_node.get_attr('stride')
        kwargs['strides'] = [1, stride, stride, 1]
        kwargs['pooling_type'] = source_node.get_attr('pool')
        pad = source_node.get_attr('padding')
        IR_node.attr["pads"].list.i.extend(([0]+[pad, pad]+[0])*2)
        assign_IRnode_values(IR_node, kwargs)
        # print(IR_node)
        # assert False

    def rename_yolo(self, source_node):
        print(source_node.layer)
        IR_node = self._convert_identity_operation(source_node, new_op='yolo')
        kwargs = {}
        kwargs['truth_thresh'] = source_node.get_attr('truth_thresh')
        kwargs['random'] = source_node.get_attr('random')
        kwargs['ignore_thresh'] = source_node.get_attr('ignore_thresh')
        kwargs['jitter'] = source_node.get_attr('jitter')
        kwargs['num'] = source_node.get_attr('num')
        kwargs['classes'] = source_node.get_attr('classes')
        kwargs['anchors'] = source_node.get_attr('anchors')
        kwargs['mask'] = source_node.get_attr('mask')
        assign_IRnode_values(IR_node, kwargs)

        # print(IR_node)
        # assert False
        # return

    def rename_Concat(self, source_node):
        # print(source_node.layer)
        IR_node = self._convert_identity_operation(source_node, new_op='Concat')
        IR_node.attr["axis"].i = int(source_node.get_attr("axis", "1"))
        # print(IR_node)
        # assert False
        # return

    def rename_upsample(self, source_node):
        # print(source_node.layer)
        IR_node = self._convert_identity_operation(source_node, new_op='upsample')
        stride = source_node.get_attr('strides')
        # print(stride)
        kwargs = {}
        kwargs['strides'] = stride

        assign_IRnode_values(IR_node, kwargs)

        # print(IR_node)
        # assert False

    def rename_Add(self, source_node):
        # print(source_node.layer)
        IR_node = self._convert_identity_operation(source_node, new_op='Add')
        # print(IR_node)
        # assert False

    def rename_InnerProduct(self, source_node):
        print(source_node.layer)
        assert False

    def rename_Softmax(self, source_node):
        print(source_node.layer)
        assert False

