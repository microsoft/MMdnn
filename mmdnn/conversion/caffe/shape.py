from collections import namedtuple
import math

TensorShape = namedtuple('TensorShape', ['batch_size', 'channels', 'height', 'width'])


def get_kernel_extents(params, dilation):
    ko_h = dilation * (int(params.k_h) - 1) + 1
    ko_w = dilation * (int(params.k_w) - 1) + 1
    return ko_h, ko_w

def get_filter_output_shape(i_h, i_w, dilation, params, round_func):
    ko_h, ko_w = get_kernel_extents(params, dilation)

    o_h = (i_h + 2 * params.p_h - ko_h) / float(params.s_h) + 1
    o_w = (i_w + 2 * params.p_w - ko_w) / float(params.s_w) + 1
    return (int(round_func(o_h)), int(round_func(o_w)))


def get_strided_kernel_output_shape(node, round_func):
    assert node.layer is not None
    input_shape = node.get_only_parent()[0].output_shape
    params = node.kernel_parameters
    dilation = node.parameters.dilation[0] if hasattr(node.parameters, 'dilation') and node.parameters.dilation else 1

    o_h, o_w = get_filter_output_shape(input_shape.height, input_shape.width,
                                       dilation, params, round_func)
    params = node.parameters
    has_c_o = hasattr(params, 'num_output')
    c = params.num_output if has_c_o else input_shape.channels
    return TensorShape(input_shape.batch_size, c, o_h, o_w)


def shape_not_implemented(node):
    raise NotImplementedError

def shape_deconvolution(node):
    input_shape = node.get_only_parent()[0].output_shape
    params = node.kernel_parameters
    dilation = 1 if len(node.parameters.dilation) == 0 else node.parameters.dilation[0]

    ko_h, ko_w = get_kernel_extents(params, dilation)
    o_h = int(params.s_h) * (input_shape.height - 1) + ko_h - 2 * int(params.p_h)
    o_w = int(params.s_w) * (input_shape.width - 1) + ko_w - 2 * int(params.p_w)

    has_c_o = hasattr(node.parameters, 'num_output')
    c = node.parameters.num_output if has_c_o else input_shape.channels
    return TensorShape(input_shape.batch_size, c, o_h, o_w)

def shape_identity(node):
    assert len(node.parents) > 0
    return node.parents[0][0].output_shape


def shape_scalar(node):
    return TensorShape(1, 1, 1, 1)

def shape_reshape(node):
    last_shape = node.get_only_parent()[0].output_shape
    shapes = []
    for idx, shape in enumerate(node.layer.reshape_param.shape.dim):
        shapes.append(shape if shape != 0 else last_shape[idx])

    if len(shapes) == 1 and shapes[0]==-1:
        total_dim = 1
        for i in last_shape:
            total_dim *= i
        return TensorShape(1, 1, 1, total_dim) # return NHWC format

    elif len(shapes) == 4:
        return TensorShape(shapes[0], shapes[1], shapes[2], shapes[3])

    else:
        raise NotImplementedError

def shape_data(node):
    if node.output_shape:
        # Old-style input specification
        return node.output_shape
    try:
        # New-style input specification
        return tuple(map(int, node.parameters.shape[0].dim))
    except:
        # We most likely have a data layer on our hands. The problem is,
        # Caffe infers the dimensions of the data from the source (eg: LMDB).
        # We want to avoid reading datasets here. Fail for now.
        # This can be temporarily fixed by transforming the data layer to
        # Caffe's "input" layer (as is usually used in the "deploy" version).
        # TODO: Find a better solution for this.
        pass


def shape_mem_data(node):
    params = node.parameters
    return TensorShape(params.batch_size, params.channels, params.height, params.width)


def shape_concat(node):
    axis = node.parameters.axis
    output_shape = None
    for parent, idx in node.parents:
        if output_shape is None:
            output_shape = list(parent.output_shape)
        else:
            output_shape[axis] += parent.output_shape[axis]
    return tuple(output_shape)


def shape_convolution(node):
    return get_strided_kernel_output_shape(node, math.floor)


def shape_pool(node):
    if node.parameters.global_pooling:
        return shape_global_pooling(node)
    return get_strided_kernel_output_shape(node, math.ceil)

def shape_unpool(node):
    return get_strided_kernel_output_shape(node, math.ceil)

def shape_inner_product(node):
    input_shape = node.get_only_parent()[0].output_shape
    return TensorShape(input_shape.batch_size, node.parameters.num_output, 1, 1)


def shape_global_pooling(node):
    input_shape = node.get_only_parent()[0].output_shape
    params = node.kernel_parameters
    has_c_o = hasattr(params, 'num_output')
    c = params.num_output if has_c_o else input_shape.channels
    return TensorShape(input_shape.batch_size, c, 1, 1)  # Output height and width is 1 when global_pooling


def shape_split(node):
    input_shape = node.get_only_parent()[0].output_shape
    return TensorShape(input_shape.batch_size, input_shape.channels, input_shape.height, input_shape.width)


def shape_flatten(node):
    input_shape = node.get_only_parent()[0].output_shape
    return TensorShape(input_shape.batch_size, input_shape.channels * input_shape.height * input_shape.width, 1, 1)

