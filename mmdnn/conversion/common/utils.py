#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import division
import os
import math
import urllib
import sys
import numpy as np

__all__ = ["assign_IRnode_values", "convert_onnx_pad_to_tf", 'convert_tf_pad_to_onnx', 'compute_tf_same_padding', 'is_valid_padding', 'download_file']

def assign_attr_value(attr, val):
    from mmdnn.conversion.common.IR.graph_pb2 import TensorShape
    '''Assign value to AttrValue proto according to data type.'''
    if isinstance(val, bool):
        attr.b = val
    elif isinstance(val, int):
        attr.i = val
    elif isinstance(val, float):
        attr.f = val
    elif isinstance(val, str):
        attr.s = val.encode('utf-8')
    elif isinstance(val, bytes):
        attr.s = val
    elif isinstance(val, TensorShape):
        attr.shape.MergeFromString(val.SerializeToString())
    elif isinstance(val, list):
        if not val:
            return
        if isinstance(val[0], int):
            attr.list.i.extend(val)
        elif isinstance(val[0], TensorShape):
            attr.list.shape.extend(val)
        else:
            raise NotImplementedError('AttrValue cannot be of %s %s' % (type(val), type(val[0])))
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


def _progress_check(count, block_size, total_size):
    progress_size = int(count * block_size)
    percent = int(count * block_size * 100 / total_size)
    percent = min(percent, 100)
    sys.stdout.write ("\r...%d%%, %d MB" %
                    (percent, progress_size / (1024 * 1024)))
    sys.stdout.flush()


def download_file(url, directory='./', local_fname=None, force_write=False):
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

    if local_fname is None:
        local_fname = url.split('/')[-1]
    local_fname = os.path.join(directory, local_fname)

    if os.path.exists(local_fname) and not force_write:
        print ("File [{}] existed!".format(local_fname))
        return local_fname

    print ("Downloading symbol file [{}] from [{}]".format(local_fname, url))
    result, _ = urllib.request.urlretrieve(url, local_fname, _progress_check)
    return result
