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


__all__ = ["assign_IRnode_values", "convert_onnx_pad_to_tf", 'convert_tf_pad_to_onnx',
           'compute_tf_same_padding', 'is_valid_padding', 'download_file',
           'shape_to_list', 'list_to_shape']


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
                for name in tar.getnames():
                    if not (os.path.realpath(os.path.join(directory, name))+ os.sep).startswith(os.path.realpath(directory) + os.sep):
                        raise ValueError('The decompression path does not match the current path. For more info: https://docs.python.org/3/library/tarfile.html#tarfile.TarFile.extractall')
                tar.extractall(directory)
                tar.close()
            except ValueError:
                raise
            except:
                print("Unzip file [{}] failed.".format(ret))

        elif ret.endswith('.zip'):
            try:
                import zipfile
                zip_ref = zipfile.ZipFile(ret, 'r')
                for name in zip_ref.namelist():
                    if not (os.path.realpath(os.path.join(directory, name))+ os.sep).startswith(os.path.realpath(directory) + os.sep):
                        raise ValueError('The decompression path does not match the current path. For more info: https://docs.python.org/3/library/zipfile.html?highlight=zipfile#zipfile.ZipFile.extractall')
                zip_ref.extractall(directory)
                zip_ref.close()
            except ValueError:
                raise
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
