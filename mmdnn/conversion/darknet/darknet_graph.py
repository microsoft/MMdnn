#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
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
        self.weights = {}
        self.original_list = OrderedDict()

    @staticmethod
    def dim_str_to_int(input_dim):
        if type(input_dim) == list:
            return [int(i) for i in input_dim]

    @staticmethod
    def conv_output_width(width, padding, kernel_size, stride):
        return (width + 2*padding - kernel_size)/stride + 1

    @staticmethod
    def conv_output_height(height, padding, kernel_size, stride):
        return (height + 2*padding - kernel_size)/stride + 1

    def build(self):

        for i, block in enumerate(self.model):
            # print("\n")
            # print(i)
            # print(block)

            # continue
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
                self.original_list[node['name']] = DarknetGraphNode(node)
                self.layer_num_map[i] = node['name']
                pre_node_name = node['name']

            elif block['type'] == 'convolutional':
                conv_layer = OrderedDict()
                conv_layer['input'] = [pre_node_name]

                input_shape = self.layer_map[pre_node_name].get_attr('_output_shape')
                w = input_shape[1]
                h = input_shape[2]
                channels = input_shape[3]
                # assert False

                if 'name' in block.keys():
                    conv_layer['name'] = block['name']
                else:
                    conv_layer['name'] = 'layer%d-conv' % i
                conv_layer['type'] = 'Conv'

                convolution_param = OrderedDict()
                convolution_param['num_output'] = int(block['filters'])
                convolution_param['kernel_size'] = int(block['size'])
                convolution_param['kernel'] = [int(block['size']), int(block['size']), channels, int(block['filters'])]
                convolution_param['pad'] = int(block['pad'])

                if block['pad'] == '1':
                    convolution_param['padding'] = int(convolution_param['kernel_size'])//2
                convolution_param['stride'] = int(block['stride'])
                if block['batch_normalize'] == '1':
                    convolution_param['bias_term'] = 'false'
                else:
                    convolution_param['bias_term'] = 'true'
                output_w = self.conv_output_width(w ,convolution_param['padding'], convolution_param['kernel_size'], convolution_param['stride'])
                output_h = self.conv_output_height(h ,convolution_param['padding'], convolution_param['kernel_size'], convolution_param['stride'])
                convolution_param['_output_shape'] = [-1, output_w, output_h, convolution_param['num_output']]
                conv_layer['attr'] = convolution_param
                self.layer_map[conv_layer['name']] = DarknetGraphNode(conv_layer)
                self.original_list[conv_layer['name']] = DarknetGraphNode(conv_layer)
                pre_node_name = conv_layer['name']

                if block['batch_normalize'] == '1':
                    bn_layer = OrderedDict()
                    bn_layer['input'] = [pre_node_name]


                    input_shape = self.layer_map[pre_node_name].get_attr('_output_shape')
                    if 'name' in block.keys():
                        bn_layer['name'] = '%s-bn' % block['name']
                    else:
                        bn_layer['name'] = 'layer%d-bn' % i
                    bn_layer['type'] = 'BatchNorm'
                    batch_norm_param = OrderedDict()
                    batch_norm_param['use_global_stats'] = True
                    batch_norm_param['_output_shape'] = convolution_param['_output_shape']
                    batch_norm_param['bias_term'] = True
                    batch_norm_param['scale'] = True
                    bn_layer['attr'] = batch_norm_param


                    self.layer_map[bn_layer['name']] = DarknetGraphNode(bn_layer)
                    self.original_list[bn_layer['name']] = DarknetGraphNode(bn_layer)

                    pre_node_name = bn_layer['name']


                if block['activation'] != 'linear':
                    relu_layer = OrderedDict()
                    relu_layer['input'] = [pre_node_name]
                    if 'name' in block.keys():
                        relu_layer['name'] = '%s-act' % block['name']
                    else:
                        relu_layer['name'] = 'layer%d-act' % i
                    relu_layer['type'] = 'ReLU'
                    relu_param = OrderedDict()
                    if block['activation'] == 'leaky':
                        relu_layer['type'] = 'leakyReLU'
                        relu_param['negative_slope'] = '0.1'
                    relu_param['_output_shape'] = input_shape
                    relu_layer['attr'] = relu_param
                    self.layer_map[relu_layer['name']] = DarknetGraphNode(relu_layer)
                    self.layer_num_map[i] = relu_layer['name']
                    self.original_list[relu_layer['name']] = DarknetGraphNode(relu_layer)
                    pre_node_name = relu_layer['name']

                else:
                    self.layer_num_map[i] = bn_layer['name']


            elif block['type'] == 'maxpool':
                max_layer = OrderedDict()
                max_layer['input'] = [pre_node_name]
                if 'name' in block.keys():
                    max_layer['name'] = block['name']
                else:
                    max_layer['name'] = 'layer%d-maxpool' % i
                max_layer['type'] = 'Pooling'
                pooling_param = OrderedDict()
                pooling_param['kernel_size'] = int(block['size'])
                pooling_param['stride'] = int(block['stride'])
                pooling_param['pool'] = 'MAX'
                pooling_param['padding'] = 0
                if 'pad' in block.keys() and int(block['pad']) == 1:
                    pooling_param['padding'] = (int(block['size'])-1)/2

                input_shape = self.layer_map[pre_node_name].get_attr('_output_shape')
                w = input_shape[1]
                h = input_shape[2]
                output_w = (w + 2*pooling_param['padding'])/pooling_param['stride']
                output_h = (h + 2*pooling_param['padding'])/pooling_param['stride']

                pooling_param['_output_shape'] = [-1, output_w, output_h, input_shape[-1]]
                max_layer['attr'] = pooling_param
                self.layer_map[max_layer['name']] = DarknetGraphNode(max_layer)
                self.original_list[max_layer['name']] = DarknetGraphNode(max_layer)
                self.layer_num_map[i] = max_layer['name']
                pre_node_name = max_layer['name']

            elif block['type'] == 'avgpool':
                avg_layer = OrderedDict()

                avg_layer['input'] = [pre_node_name]
                if 'name' in block.keys():
                    avg_layer['name'] = block['name']
                else:
                    avg_layer['name'] = 'layer%d-avgpool' % i
                avg_layer['type'] = 'Pooling'
                pooling_param = OrderedDict()
                input_shape = self.layer_map[pre_node_name].get_attr('_output_shape')
                pooling_param['_output_shape'] = [-1, 1, 1, input_shape[-1]]
                pooling_param['pool'] = 'AVG'
                avg_layer['attr'] = pooling_param
                self.layer_map[avg_layer['name']] = DarknetGraphNode(avg_layer)
                self.original_list[avg_layer['name']] = DarknetGraphNode(avg_layer)
                self.layer_num_map[i] = avg_layer['name']
                pre_node_name = avg_layer['name']

            elif block['type'] == 'route':
                prev = block['layers'].split(',') #[-1,61]
                if len(prev) == 1:
                    prev_layer_id = i + int(prev[0])
                    self.layer_num_map[i] = self.layer_num_map[prev_layer_id]
                    pre_node_name = self.layer_num_map[i]
                elif len(prev) == 2:
                    input_list = []
                    input_shape = []
                    route_layer = OrderedDict()
                    for p in prev:
                        if int(p)>0:

                            input_name = self.layer_num_map[int(p)+1]
                            input_list.append(input_name)
                            input_shape.append(self.layer_map[input_name].get_attr('_output_shape'))

                        else:
                            prev_layer_id = i + int(p)
                            input_name = self.layer_num_map[prev_layer_id]
                            input_shape.append(self.layer_map[input_name].get_attr('_output_shape'))
                            input_list.append(input_name)
                    route_param = OrderedDict()


                    shape_ = 0
                    for shape in input_shape:
                        shape_ += shape[-1]
                    route_param['axis'] = 3
                    route_param['_output_shape'] = input_shape[0][:-1] + [shape_]
                    route_layer['input'] = input_list

                    if 'name' in block.keys():
                        route_layer['name'] = block['name']
                    else:
                        route_layer['name'] = 'layer%d-concat' % i

                    route_layer['type'] = 'Concat'
                    route_layer['attr'] = route_param

                    self.layer_map[route_layer['name']] = DarknetGraphNode(route_layer)
                    self.original_list[route_layer['name']] = DarknetGraphNode(route_layer)
                    self.layer_num_map[i] = route_layer['name']
                    pre_node_name = route_layer['name']

            elif block['type'] == 'shortcut':
                prev_layer_id1 = i + int(block['from'])
                prev_layer_id2 = i - 1
                bottom1 = self.layer_num_map[prev_layer_id1]
                bottom2 = self.layer_num_map[prev_layer_id2]
                input_shape = self.layer_map[bottom2].get_attr('_output_shape')
                shortcut_layer = OrderedDict()
                shortcut_layer['input'] = [bottom1, bottom2]
                # print(shortcut_layer['input'] )
                if 'name' in block.keys():
                    shortcut_layer['name'] = block['name']
                else:
                    shortcut_layer['name'] = 'layer%d-shortcut' % i
                shortcut_layer['type'] = 'Add'
                eltwise_param = OrderedDict()
                eltwise_param['operation'] = 'SUM'
                eltwise_param['_output_shape'] = input_shape
                shortcut_layer['attr'] = eltwise_param


                self.layer_map[shortcut_layer['name']] = DarknetGraphNode(shortcut_layer)
                self.original_list[shortcut_layer['name']] = DarknetGraphNode(shortcut_layer)
                self.layer_num_map[i] = shortcut_layer['name']
                pre_node_name = shortcut_layer['name']

                if block['activation'] != 'linear':
                    relu_layer = OrderedDict()
                    relu_layer['input'] = [pre_node_name]
                    if 'name' in block.keys():
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
                    self.original_list[relu_layer['name']] = DarknetGraphNode(relu_layer)
                    pre_node_name = relu_layer['name']

            elif block['type'] == 'connected':
                fc_layer = OrderedDict()
                fc_layer['input'] = [pre_node_name]
                if 'name' in block.keys():
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
                self.original_list[fc_layer['name']] = DarknetGraphNode(fc_layer)
                self.layer_num_map[i] = fc_layer['name']
                pre_node_name = fc_layer['name']

                if block['activation'] != 'linear':
                    relu_layer = OrderedDict()
                    relu_layer['input'] = [pre_node_name]
                    if 'name' in block.keys():
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
                    self.original_list[relu_layer['name']] = DarknetGraphNode(relu_layer)
                    pre_node_name = relu_layer['name']

            elif block['type'] == 'softmax':
                sm_layer = OrderedDict()

                sm_layer['input'] = [pre_node_name]
                if 'name' in block.keys():
                    sm_layer['name'] = block['name']
                else:
                    sm_layer['name'] = 'layer%d-softmax' % i
                sm_layer['type'] = 'Softmax'
                softmax_param = OrderedDict()
                input_shape = self.layer_map[pre_node_name].get_attr('_output_shape')
                softmax_param['_output_shape'] = input_shape
                sm_layer['attr'] = softmax_param
                self.layer_map[sm_layer['name']] = DarknetGraphNode(sm_layer)
                self.original_list[sm_layer['name']] = DarknetGraphNode(sm_layer)
                self.layer_num_map[i] = sm_layer['name']
                pre_node_name = sm_layer['name']

            elif block['type'] == 'yolo':

                yolo_layer = OrderedDict()
                yolo_layer['input'] = [pre_node_name]
                if 'name' in block.keys():
                    yolo_layer['name'] = block['name']
                else:
                    yolo_layer['name'] = 'layer%d-yolo' % i
                yolo_layer['type'] = 'yolo'
                yolo_param = OrderedDict()
                yolo_param['truth_thresh'] = float(block['truth_thresh'])
                yolo_param['random'] = float(block['random'])
                yolo_param['ignore_thresh'] = float(block['ignore_thresh'])
                yolo_param['jitter'] = float(block['jitter'])
                yolo_param['num'] = int(block['num'])
                yolo_param['classes'] = int(block['classes'])
                anchors = [int(t) for t in block['anchors'].split(',')]
                yolo_param['anchors'] = anchors
                mask = [int(t) for t in block['mask'].split(',')]
                yolo_param['mask'] = mask

                yolo_layer['attr'] = yolo_param
                self.layer_map[yolo_layer['name']] = DarknetGraphNode(yolo_layer)
                self.original_list[yolo_layer['name']] = DarknetGraphNode(yolo_layer)
                self.layer_num_map[i] = yolo_layer['name']

            elif block['type'] == 'upsample':

                input_shape = self.layer_map[pre_node_name].get_attr('_output_shape')
                upsample_layer = OrderedDict()
                upsample_layer['input'] = [pre_node_name]
                if 'name' in block.keys():
                    upsample_layer['name'] = block['name']
                else:
                    upsample_layer['name'] = 'layer%d-upsample' % i
                upsample_layer['type'] = 'upsample'
                upsample_param = OrderedDict()
                stride = block['stride']
                upsample_param['strides'] = int(stride)
                upsample_param['_output_shape'] = [input_shape[0]] + [q*int(stride) for q in input_shape[1:3]] + [input_shape[-1]]
                upsample_layer['attr'] = upsample_param
                self.layer_map[upsample_layer['name']] = DarknetGraphNode(upsample_layer)
                self.original_list[upsample_layer['name']] = DarknetGraphNode(upsample_layer)
                self.layer_num_map[i] = upsample_layer['name']
                pre_node_name = upsample_layer['name']

            elif block['type'] == 'cost':
                continue

            # spacetodepth
            elif block['type'] == 'reorg':
                input_shape = self.layer_map[pre_node_name].get_attr('_output_shape')
                reorg_layer = OrderedDict()
                reorg_layer['input'] = [pre_node_name]
                if 'name' in block.keys():
                    reorg_layer['name'] = block['name']
                else:
                    reorg_layer['name'] = 'layer%d-reorg' % i

                reorg_layer['type'] = 'SpaceToDepth'
                reorg_param = OrderedDict()
                stride = int(block['stride'])
                reorg_param['strides'] = stride
                reorg_param['_output_shape'] = [-1, input_shape[1]/stride, input_shape[2]/stride, input_shape[3]*stride*stride]
                reorg_layer['attr'] = reorg_param

                self.layer_map[reorg_layer['name']] = DarknetGraphNode(reorg_layer)
                self.original_list[reorg_layer['name']] = DarknetGraphNode(reorg_layer)
                self.layer_num_map[i] = reorg_layer['name']
                pre_node_name = reorg_layer['name']


            elif block['type'] == 'region':
                # print(block)
                region_layer = OrderedDict()
                region_layer['input'] = [pre_node_name]
                if 'name' in block.keys():
                    region_layer['name'] = block['name']
                else:
                    region_layer['name'] = 'layer%d-region' % i
                region_layer['type'] = 'region'
                region_param = OrderedDict()
                region_param['softmax'] = int(block['softmax'])
                region_param['thresh'] = float(block['thresh'])
                region_param['random'] = float(block['random'])
                region_param['jitter'] = float(block['jitter'])
                region_param['num'] = int(block['num'])
                region_param['classes'] = int(block['classes'])
                region_param['coords'] = int(block['coords'])
                region_param['rescore'] = int(block['rescore'])
                region_param['object_scale'] = int(block['object_scale'])

                region_param['noobject_scale'] = int(block['noobject_scale'])
                region_param['class_scale'] = int(block['class_scale'])
                region_param['coord_scale'] = int(block['coord_scale'])

                region_param['bias_match'] = int(block['bias_match'])
                region_param['absolute'] = int(block['absolute'])

                anchors = [float(t) for t in block['anchors'].split(',')]
                region_param['anchors'] = anchors

                region_layer['attr'] = region_param
                # print(region_layer)
                self.layer_map[region_layer['name']] = DarknetGraphNode(region_layer)
                self.original_list[region_layer['name']] = DarknetGraphNode(region_layer)
                self.layer_num_map[i] = region_layer['name']
                # assert False


            else:
                print('unknown layer type %s ' % block['type'])
                print(block,"\n")
                assert False



        for layer in self.layer_map:
            for pred in self.layer_map[layer].layer['input']:
                if pred not in self.layer_map.keys() and pred != 'data':
                    print(pred)
                    print("::::::::::::: unknown input :::::::::::::")
                    assert False

                self._make_connection(pred, layer)

        super(DarknetGraph, self).build()

