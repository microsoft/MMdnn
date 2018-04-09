#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from collections import OrderedDict
from mmdnn.conversion.common.DataStructure.graph import GraphNode, Graph
# from tensorflow.core.framework.node_def_pb2 import NodeDef
# from tensorflow.core.framework import attr_value_pb2


class DarknetGraphNode(GraphNode):

    def __init__(self, layer):

        super(DarknetGraphNode, self).__init__(layer)


    @property
    def name(self):
        return self.layer['name']


    @property
    def type(self):
        return self.layer['type']


    @property
    def dk_layer(self):
        return self.layer


    def get_attr(self, name, default_value = None):
        if name in self.layer['attr'].keys():
            return self.layer['attr'][name]
        else:
            return default_value


class DarknetGraph(Graph):

    def __init__(self, model):
        # pass

        super(DarknetGraph, self).__init__(model)
        self.layer_num_map = {}
        self.model = model

    @staticmethod
    def dim_str_to_int(input_dim):
        if type(input_dim) == list:
            return [int(i) for i in input_dim]

    @staticmethod
    def conv_output_width(width, pad, kernel_size, stride):
        return (width + 2*pad - kernel_size)/stride + 1

    @staticmethod
    def conv_output_height(height, pad, kernel_size, stride):
        return (height + 2*pad - kernel_size)/stride + 1

    def build(self):
        for i, block in enumerate(self.model):
            # print(i)
            # print(block)
            # print("\n")
            node = OrderedDict()
            if block['type'] == 'net':
                node['name'] = 'dk_Input'
                node['input'] = ['data']
                node['type'] = 'DataInput'
                node['input_dim'] = ['-1']
                # NHWC
                node['input_dim'].append(block['height'])
                node['input_dim'].append(block['width'])
                node['input_dim'].append(block['channels'])
                input_param = OrderedDict()
                input_param['shape'] = self.dim_str_to_int(node['input_dim'])
                input_param['_output_shape'] = self.dim_str_to_int(node['input_dim'])
                node['attr'] = input_param
                self.layer_map[node['name']] = DarknetGraphNode(node)
                self.layer_num_map[i] = node['name']
                pre_node_name = node['name']

            elif block['type'] == 'convolutional':
                conv_layer = OrderedDict()
                conv_layer['input'] = [pre_node_name]

                input_shape = self.layer_map[pre_node_name].get_attr('_output_shape')
                w = input_shape[1]
                h = input_shape[2]
                # assert False

                if block.has_key('name'):
                    conv_layer['name'] = block['name']
                else:
                    conv_layer['name'] = 'layer%d-conv' % i
                conv_layer['type'] = 'Conv'

                convolution_param = OrderedDict()
                convolution_param['num_output'] = int(block['filters'])
                convolution_param['kernel_size'] = int(block['size'])
                if block['pad'] == '1':
                    convolution_param['pad'] = int(convolution_param['kernel_size'])/2
                convolution_param['stride'] = int(block['stride'])
                if block['batch_normalize'] == '1':
                    convolution_param['bias_term'] = 'false'
                else:
                    convolution_param['bias_term'] = 'true'
                output_w = self.conv_output_width(w ,convolution_param['pad'], convolution_param['kernel_size'], convolution_param['stride'])
                output_h = self.conv_output_height(h ,convolution_param['pad'], convolution_param['kernel_size'], convolution_param['stride'])
                convolution_param['_output_shape'] = [-1, output_w, output_h, convolution_param['num_output']]
                conv_layer['attr'] = convolution_param
                self.layer_map[conv_layer['name']] = DarknetGraphNode(conv_layer)
                self.layer_num_map[i] = conv_layer['name']
                pre_node_name = conv_layer['name']
                # print("====", pre_node_name)
                # print( self.layer_map[conv_layer['name']].layer)

                if block['batch_normalize'] == '1':
                    bn_layer = OrderedDict()
                    # print("====", pre_node_name)
                    bn_layer['input'] = [pre_node_name]


                    input_shape = self.layer_map[pre_node_name].get_attr('_output_shape')
                    if block.has_key('name'):
                        bn_layer['name'] = '%s-bn' % block['name']
                    else:
                        bn_layer['name'] = 'layer%d-bn' % i
                    bn_layer['type'] = 'BatchNorm'
                    batch_norm_param = OrderedDict()
                    batch_norm_param['use_global_stats'] = 'true'
                    batch_norm_param['_output_shape'] = convolution_param['_output_shape']
                    bn_layer['attr'] = batch_norm_param
                    self.layer_map[bn_layer['name']] = DarknetGraphNode(bn_layer)
                    # print( self.layer_map[bn_layer['name']].layer)
                    # print("*************")
                    pre_node_name = bn_layer['name']

                    scale_layer = OrderedDict()
                    scale_layer['input'] = [pre_node_name]
                    if block.has_key('name'):
                        scale_layer['name'] = '%s-scale' % block['name']
                    else:
                        scale_layer['name'] = 'layer%d-scale' % i
                    scale_layer['type'] = 'Scale'
                    scale_param = OrderedDict()
                    scale_param['bias_term'] = 'true'
                    scale_param['_output_shape'] = convolution_param['_output_shape']
                    scale_layer['attr'] = scale_param
                    self.layer_map[scale_layer['name']] = DarknetGraphNode(scale_layer)
                    pre_node_name = scale_layer['name']

                    # print( self.layer_map['layer1-scale'].layer)

                if block['activation'] != 'linear':
                    relu_layer = OrderedDict()
                    relu_layer['input'] = [pre_node_name]
                    if block.has_key('name'):
                        relu_layer['name'] = '%s-act' % block['name']
                    else:
                        relu_layer['name'] = 'layer%d-act' % i
                    relu_layer['type'] = 'ReLU'
                    relu_param = OrderedDict()
                    if block['activation'] == 'leaky':
                        relu_param['negative_slope'] = '0.1'
                    relu_param['_output_shape'] = input_shape
                    relu_layer['attr'] = relu_param
                    self.layer_map[relu_layer['name']] = DarknetGraphNode(relu_layer)
                    pre_node_name = relu_layer['name']

            elif block['type'] == 'maxpool':
                max_layer = OrderedDict()
                max_layer['input'] = [pre_node_name]
                if block.has_key('name'):
                    max_layer['name'] = block['name']
                else:
                    max_layer['name'] = 'layer%d-maxpool' % i
                max_layer['type'] = 'Pooling'
                pooling_param = OrderedDict()
                pooling_param['kernel_size'] = int(block['size'])
                pooling_param['stride'] = int(block['stride'])
                pooling_param['pool'] = 'MAX'
                if block.has_key('pad') and int(block['pad']) == 1:
                    pooling_param['pad'] = (int(block['size'])-1)/2

                input_shape = self.layer_map[pre_node_name].get_attr('_output_shape')
                w = input_shape[1]
                h = input_shape[2]
                output_w = (w + 2*pooling_param['pad'])/pooling_param['stride']
                output_h = (h + 2*pooling_param['pad'])/pooling_param['stride']

                pooling_param['_output_shape'] = [-1, output_w, output_h, input_shape[-1]]
                max_layer['attr'] = pooling_param
                self.layer_map[max_layer['name']] = DarknetGraphNode(max_layer)
                self.layer_num_map[i] = max_layer['name']
                pre_node_name = max_layer['name']
                # assert False

            elif block['type'] == 'avgpool':
                avg_layer = OrderedDict()

                avg_layer['input'] = [pre_node_name]
                if block.has_key('name'):
                    avg_layer['name'] = block['name']
                else:
                    avg_layer['name'] = 'layer%d-avgpool' % i
                avg_layer['type'] = 'Pooling'
                pooling_param = OrderedDict()
                input_shape = self.layer_map[pre_node_name].get_attr('_output_shape')
                pooling_param['_output_shape'] = [-1, 1, 1, input_shape[-1]]
                pooling_param['pool'] = 'AVE'
                avg_layer['attr'] = pooling_param
                self.layer_map[avg_layer['name']] = DarknetGraphNode(avg_layer)
                self.layer_num_map[i] = avg_layer['name']
                pre_node_name = avg_layer['name']

            elif block['type'] == 'route':
                prev = block['layers'].split(',') #[-1,61]
                # print(prev)
                prev_layer_id = i + int(prev[0])
                route_layer = OrderedDict()
                # route_layer['input'] = self.layer_num_map[prev_layer_id]
                # route_layer['name'] = self.layer_num_map[prev_layer_id]
                # route_layer['type'] = 'Identity'
                # self.layer_map[route_layer['name']] = DarknetGraphNode(route_layer)
                self.layer_num_map[i] = self.layer_num_map[prev_layer_id]
                pre_node_name = self.layer_num_map[i]

            elif block['type'] == 'shortcut':
                prev_layer_id1 = i + int(block['from'])
                prev_layer_id2 = i - 1
                bottom1 = self.layer_num_map[prev_layer_id1]
                bottom2 = self.layer_num_map[prev_layer_id2]
                input_shape = self.layer_map[bottom2].get_attr('_output_shape')
                shortcut_layer = OrderedDict()
                shortcut_layer['input'] = [bottom1, bottom2]
                # print(shortcut_layer['input'] )
                if block.has_key('name'):
                    shortcut_layer['name'] = block['name']
                else:
                    shortcut_layer['name'] = 'layer%d-shortcut' % i
                shortcut_layer['type'] = 'Eltwise'
                eltwise_param = OrderedDict()
                eltwise_param['operation'] = 'SUM'
                eltwise_param['_output_shape'] = input_shape
                shortcut_layer['attr'] = eltwise_param


                self.layer_map[shortcut_layer['name']] = DarknetGraphNode(shortcut_layer)
                self.layer_num_map[i] = shortcut_layer['name']
                pre_node_name = shortcut_layer['name']

                # bottom = shortcut_layer['top']

                if block['activation'] != 'linear':
                    relu_layer = OrderedDict()
                    relu_layer['input'] = pre_node_name
                    if block.has_key('name'):
                        relu_layer['name'] = '%s-act' % block['name']
                    else:
                        relu_layer['name'] = 'layer%d-act' % i
                    relu_layer['type'] = 'ReLU'
                    relu_param = OrderedDict()
                    relu_param['_output_shape'] = input_shape
                    if block['activation'] == 'leaky':

                        relu_param['negative_slope'] = '0.1'

                    relu_layer['attr'] = relu_param
                    self.layer_map[relu_layer['name']] = DarknetGraphNode(relu_layer)
                    pre_node_name = relu_layer['name']

            elif block['type'] == 'connected':
                fc_layer = OrderedDict()
                fc_layer['input'] = [pre_node_name]
                if block.has_key('name'):
                    fc_layer['name'] = block['name']
                else:
                    fc_layer['name'] = 'layer%d-fc' % i
                fc_layer['type'] = 'InnerProduct'
                fc_param = OrderedDict()
                fc_param['num_output'] = int(block['output'])
                input_shape = self.layer_map[pre_node_name].get_attr('_output_shape')
                fc_param['_output_shape'] = input_shape[:-1] + [fc_param['num_output']]
                fc_layer['attr'] = fc_param
                self.layer_map[fc_layer['name']] = DarknetGraphNode(fc_layer)
                self.layer_num_map[i] = fc_layer['name']
                pre_node_name = fc_layer['name']

                if block['activation'] != 'linear':
                    relu_layer = OrderedDict()
                    relu_layer['input'] = [pre_node_name]
                    if block.has_key('name'):
                        relu_layer['name'] = '%s-act' % block['name']
                    else:
                        relu_layer['name'] = 'layer%d-act' % i
                    relu_layer['type'] = 'ReLU'
                    relu_param = OrderedDict()
                    if block['activation'] == 'leaky':

                        relu_param['negative_slope'] = '0.1'
                    relu_param['_output_shape'] = fc_param['_output_shape']
                    relu_layer['attr'] = relu_param
                    self.layer_map[relu_layer['name']] = DarknetGraphNode(relu_layer)
                    pre_node_name = relu_layer['name']

            elif block['type'] == 'softmax':
                sm_layer = OrderedDict()

                sm_layer['input'] = [pre_node_name]
                if block.has_key('name'):
                    sm_layer['name'] = block['name']
                else:
                    sm_layer['name'] = 'layer%d-softmax' % i
                sm_layer['type'] = 'Softmax'
                softmax_param = OrderedDict()
                input_shape = self.layer_map[pre_node_name].get_attr('_output_shape')
                softmax_param['_output_shape'] = input_shape
                sm_layer['attr'] = softmax_param
                self.layer_map[sm_layer['name']] = DarknetGraphNode(sm_layer)
                self.layer_num_map[i] = sm_layer['name']
                pre_node_name = sm_layer['name']

            else:
                print('unknow layer type %s ' % block['type'])
                print(block,"\n")
                # assert False



        for layer in self.layer_map:
            # print(i)
            print(layer)
            print(self.layer_map[layer].layer['input'])
            # self.layer_map[layer.name] = DarknetGraphNode(layer)
            # self.layer_name_map[layer.name] = layer.name
            for pred in self.layer_map[layer].layer['input']:
                if pred not in self.layer_map.keys():
                    print(pred)
                    print("::::::::::::::::::::::::::")
                    assert False

                self._make_connection(pred, layer)

        super(DarknetGraph, self).build()

