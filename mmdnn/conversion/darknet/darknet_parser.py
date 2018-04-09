
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

#         save_prototxt(net_info , 'resnet50.prototxt', region=False)
#         # assert False
#         net = caffe.Net('resnet50.prototxt', caffe.TEST)
#         params = net.params

        model = parse_cfg(model_config)
        # print(model)
        self.dk_graph = DarknetGraph(model)
        self.dk_graph.build()

#         fp = open(weightfile, 'rb')
#         header = np.fromfile(fp, count=4, dtype=np.int32)
#         buf = np.fromfile(fp, dtype = np.float32)
#         fp.close()

#         print(buf)
#         print(type(buf))

#         layer_id = 1
#         start = 0
#         for block in blocks:
#             print(block)
#             if start >= buf.size:
#                 break

#             if block['type'] == 'net':
#                 continue
#             elif block['type'] == 'convolutional':
#                 batch_normalize = int(block['batch_normalize'])
#                 if block.has_key('name'):
#                     conv_layer_name = block['name']
#                     bn_layer_name = '%s-bn' % block['name']
#                     scale_layer_name = '%s-scale' % block['name']
#                 else:
#                     conv_layer_name = 'layer%d-conv' % layer_id
#                     bn_layer_name = 'layer%d-bn' % layer_id
#                     scale_layer_name = 'layer%d-scale' % layer_id

#                 if batch_normalize:
#                     start = load_conv_bn2caffe(buf, start, params[conv_layer_name], params[bn_layer_name], params[scale_layer_name])
#                 else:
#                     start = load_conv2caffe(buf, start, params[conv_layer_name])
#                 layer_id = layer_id+1



# def load_conv_bn2caffe(buf, start, conv_param, bn_param, scale_param):
#     print(conv_param[0].data.shape)
#     conv_weight = conv_param[0].data
#     running_mean = bn_param[0].data
#     running_var = bn_param[1].data
#     scale_weight = scale_param[0].data
#     scale_bias = scale_param[1].data

#     scale_param[1].data[...] = np.reshape(buf[start:start+scale_bias.size], scale_bias.shape); start = start + scale_bias.size
#     scale_param[0].data[...] = np.reshape(buf[start:start+scale_weight.size], scale_weight.shape); start = start + scale_weight.size
#     bn_param[0].data[...] = np.reshape(buf[start:start+running_mean.size], running_mean.shape); start = start + running_mean.size
#     bn_param[1].data[...] = np.reshape(buf[start:start+running_var.size], running_var.shape); start = start + running_var.size
#     bn_param[2].data[...] = np.array([1.0])
#     conv_param[0].data[...] = np.reshape(buf[start:start+conv_weight.size], conv_weight.shape); start = start + conv_weight.size
#     assert False
#     return start

