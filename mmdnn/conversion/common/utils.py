#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import division
import os
import sys
import numpy as np
from six import text_type, binary_type, integer_types
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.IR_graph import IRGraphNode


__all__ = ["assign_IRnode_values", "convert_onnx_pad_to_tf", 'convert_tf_pad_to_onnx',
           'compute_tf_same_padding', 'is_valid_padding', 'download_file',
           'shape_to_list', 'list_to_shape', 'refine_IR_graph_format']


def assign_attr_value(attr, val):
    from mmdnn.conversion.common.IR.graph_pb2 import TensorShape
    '''Assign value to AttrValue proto according to data type.'''
    if isinstance(val, bool):
        attr.b = val
    elif isinstance(val, integer_types):
        attr.i = val
    elif isinstance(val, float):
        attr.f = val
    elif isinstance(val, binary_type) or isinstance(val, text_type):
        if hasattr(val, 'encode'):
            val = val.encode()
        attr.s = val
    elif isinstance(val, TensorShape):
        attr.shape.MergeFromString(val.SerializeToString())
    elif isinstance(val, list):
        if not val: return
        if isinstance(val[0], integer_types):
            attr.list.i.extend(val)
        elif isinstance(val[0], TensorShape):
            attr.list.shape.extend(val)
        elif isinstance(val[0], float):
            attr.list.f.extend(val)
        else:
            raise NotImplementedError('AttrValue cannot be of list[{}].'.format(val[0]))
    elif isinstance(val, np.ndarray):
        assign_attr_value(attr, val.tolist())
    else:
        raise NotImplementedError('AttrValue cannot be of %s' % type(val))


def assign_IRnode_values(IR_node, val_dict):
    for name, val in val_dict.items():
        assign_attr_value(IR_node.attr[name], val)


# For padding
def convert_tf_pad_to_onnx(pads):
    pads = np.reshape(pads, -1).tolist()
    dims = len(pads)
    assert dims % 2 == 0
    ret = []
    for idx in range(0, dims, 2):
        ret.append(pads[idx])
    for idx in range(1, dims, 2):
        ret.append(pads[idx])
    return ret


def convert_onnx_pad_to_tf(pads):
    return np.transpose(np.array(pads).reshape([2, -1])).reshape(-1, 2).tolist()


def is_valid_padding(pads):
    return sum(np.reshape(pads, -1)) == 0


def shape_to_list(shape):
    return [dim.size for dim in shape.dim]


def list_to_shape(shape):
    ret = graph_pb2.TensorShape()
    for dim in shape:
        new_dim = ret.dim.add()
        new_dim.size = dim
    return ret


def compute_tf_same_padding(input_shape, kernel_shape, strides, data_format='NHWC'):
    """ Convert [SAME] padding in tensorflow, keras to onnx pads,
        i.e. [x1_begin, x2_begin...x1_end, x2_end,...] """
    # print (input_shape)
    # print (kernel_shape)
    # print (strides)
    if data_format.startswith('NC'):
        # Not tested
        input_shape = input_shape[2:]
        remove_dim = len(strides) - len(input_shape)
        if remove_dim > 0:
            strides = strides[remove_dim::]

    else:
        input_shape = input_shape[1:-1]
        remove_dim = len(input_shape) - len(strides) + 1
        if remove_dim < 0:
            strides = strides[1:remove_dim]

    # print (input_shape)
    # print (kernel_shape)
    # print (strides)

    up_list = [0]
    down_list = [0]

    for idx in range(0, len(input_shape)):
        # kernel_shape[idx] = (kernel_shape[idx] - 1) * dilation_rate + 1
        output_shape = (input_shape[idx] + strides[idx] - 1) // strides[idx]
        this_padding = (output_shape - 1) * strides[idx] + kernel_shape[idx] - input_shape[idx]
        this_padding = max(0, this_padding)
        up_list.append(this_padding // 2)
        down_list.append(this_padding - this_padding // 2)

    # print ([0] + up_list + [0] + down_list if data_format.startswith('NC') else up_list + [0] + down_list + [0])
    # print ('-----------------------------------------------------')
    return [0] + up_list + [0] + down_list if data_format.startswith('NC') else up_list + [0] + down_list + [0]



# network library
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def _progress_check(count, block_size, total_size):
    read_size = count * block_size
    read_size_str = sizeof_fmt(read_size)
    if total_size > 0:
        percent = int(count * block_size * 100 / total_size)
        percent = min(percent, 100)
        sys.stdout.write("\rprogress: {} downloaded, {}%.".format(read_size_str, percent))
        if read_size >= total_size:
            sys.stdout.write("\n")
    else:
        sys.stdout.write("\rprogress: {} downloaded.".format(read_size_str))
    sys.stdout.flush()


def _single_thread_download(url, file_name):
    from six.moves import urllib
    result, _ = urllib.request.urlretrieve(url, file_name, _progress_check)
    return result


def _downloader(start, end, url, filename):
    import requests
    headers = {'Range': 'bytes=%d-%d' % (start, end)}
    r = requests.get(url, headers=headers, stream=True)
    with open(filename, "r+b") as fp:
        fp.seek(start)
        var = fp.tell()
        fp.write(r.content)


def _multi_thread_download(url, file_name, file_size, thread_count):
    import threading
    fp = open(file_name, "wb")
    fp.truncate(file_size)
    fp.close()

    part = file_size // thread_count
    for i in range(thread_count):
        start = part * i
        if i == thread_count - 1:
            end = file_size
        else:
            end = start + part

        t = threading.Thread(target=_downloader, kwargs={'start': start, 'end': end, 'url': url, 'filename': file_name})
        t.setDaemon(True)
        t.start()

    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is main_thread:
            continue
        t.join()

    return file_name


def download_file(url, directory='./', local_fname=None, force_write=False, auto_unzip=False, compre_type=''):
    """Download the data from source url, unless it's already here.

    Args:
        filename: string, name of the file in the directory.
        work_directory: string, path to working directory.
        source_url: url to download from if file doesn't exist.

    Returns:
        Path to resulting file.
    """

    if not os.path.isdir(directory):
        os.mkdir(directory)

    if not local_fname:
        k = url.rfind('/')
        local_fname = url[k + 1:]

    local_fname = os.path.join(directory, local_fname)

    if os.path.exists(local_fname) and not force_write:
        print ("File [{}] existed!".format(local_fname))
        return local_fname

    else:
        print ("Downloading file [{}] from [{}]".format(local_fname, url))
        try:
            import wget
            ret = wget.download(url, local_fname)
            print ("")
        except:
            ret = _single_thread_download(url, local_fname)

    if auto_unzip:
        if ret.endswith(".tar.gz") or ret.endswith(".tgz"):
            try:
                import tarfile
                tar = tarfile.open(ret)
                tar.extractall(directory)
                tar.close()
            except:
                print("Unzip file [{}] failed.".format(ret))

        elif ret.endswith('.zip'):
            try:
                import zipfile
                zip_ref = zipfile.ZipFile(ret, 'r')
                zip_ref.extractall(directory)
                zip_ref.close()
            except:
                print("Unzip file [{}] failed.".format(ret))
    return ret
"""
    r = requests.head(url)
    try:
        file_size = int(r.headers['content-length'])
        return _multi_thread_download(url, local_fname, file_size, 5)

    except:
        # not support multi-threads download
        return _single_thread_download(url, local_fname)

    return result
"""

FORMAT_SENSE_OP = {'ConvTranspose', 'Conv', 'BatchNorm', 'Pool', 'DepthwiseConv', 'SeparableConvolution', 'Scale', 'Pad'}

FORMAT_NONSENSE_OP = {'Dropout', 'Exp', 'LeakyRelu', 'Reciprocal', 'Relu', 'Relu6', 'Sigmoid', 'Tanh', 'Elu'}

FORMAT_UNION_OP = FORMAT_NONSENSE_OP.union(FORMAT_SENSE_OP)

LAST_FIRST = 'last2first'
FIRST_LAST = 'first2last'


def refine_IR_graph_format(IR_graph, src_format='channel_last', dst_format='channel_first'):

    if not(src_format=='channel_last' and dst_format=='channel_first'):
        raise NotImplementedError

    def add_transpose_node(in_node, node, node_idx, trans_dir=LAST_FIRST):
        for idx, out_edge in enumerate(in_node.out_edges):
            if out_edge == node.name:

                # get transpose node
                if IR_graph.layer_map.get(in_node.name + '_trans', None) is not None:
                    transpose_node = IR_graph.layer_map[in_node.name + '_trans']
                    transpose_node.in_edges.append(in_node.name)
                    transpose_node.out_edges.append(node.name)
                else:
                    # construct transpose node
                    dim = len(in_node.layer.attr['_output_shapes'].list.shape[0].dim)
                    if trans_dir == LAST_FIRST:
                        perm = [0, dim - 1] + list(range(1, dim - 1))
                    else:
                        perm = [0] + list(range(2, dim)) + [1]
                    transpose_node = IR_graph.model.node.add()
                    transpose_node.name = in_node.name + '_trans'
                    transpose_node.op = 'Transpose'

                    # assign attrs
                    ori_output_shape = [i.size for i in in_node.layer.attr['_output_shapes'].list.shape[0].dim]
                    output_shape = []
                    for i in perm:
                        output_shape.append(ori_output_shape[i])
                    output_shape = list_to_shape(output_shape)
                    kwargs = {'perm': perm, '_output_shapes': [output_shape]}
                    assign_IRnode_values(transpose_node, kwargs)

                    transpose_node = IRGraphNode(transpose_node)
                    transpose_node.format = 'channel_last'
                    transpose_node.in_edges.append(in_node.name)
                    transpose_node.out_edges.append(node.name)

                    # add transpose node into layer map
                    IR_graph.layer_map[transpose_node.name] = transpose_node
                    IR_graph.layer_name_map[transpose_node.name] = transpose_node.name

                # insert transpose node into in_node and node
                in_node.out_edges[idx] = transpose_node.name
                node.in_edges[node_idx] = transpose_node.name

    topo_sort = IR_graph.topological_sort
    for i in topo_sort:
        node = IR_graph.get_node(i)
        if node.type not in FORMAT_UNION_OP:
            node.format = 'channel_last'
            for idx, in_edge in enumerate(node.in_edges):
                in_node = IR_graph.get_node(in_edge)
                if in_node.format == 'channel_first':
                    add_transpose_node(in_node, node, idx, FIRST_LAST)
        elif node.type in FORMAT_NONSENSE_OP:
            node.format = IR_graph.get_parent(node.name, [0]).format
        elif node.type in FORMAT_SENSE_OP:
            node.format = 'channel_first'
            # refine the sense op format, making it really channel_first
            ori_output_shape = [i.size for i in node.layer.attr['_output_shapes'].list.shape[0].dim]
            dim = len(ori_output_shape)
            output_shape = []
            for i in [0, dim - 1] + list(range(1, dim - 1)):
                output_shape.append(ori_output_shape[i])
            # output_shape = list_to_shape(output_shape)
            # kwargs = {'_output_shapes': [output_shape]}
            # assign_IRnode_values(node.layer, kwargs)

            for idx, i in enumerate(node.layer.attr['_output_shapes'].list.shape[0].dim):
                i.size = output_shape[idx]

            for idx, in_edge in enumerate(node.in_edges):
                in_node = IR_graph.get_node(in_edge)
                if in_node.format == 'channel_last':
                    add_transpose_node(in_node, node, idx, LAST_FIRST)
        else:
            raise ValueError

    IR_graph.rebuild()
