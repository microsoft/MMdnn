
def Add(shapeA, shapeB, axis = None, broadcast = None):
    #do not deal
    return shapeA

def AveragePool(shape, auto_pad = None, kernelShape = None, pads = None, strides = None):
    #I don't want to deal with auto_pad
   
    if kernelShape is None:
        kernelShape = [2 for _ in range(2)]

    dim = len(kernelShape)

    if pads is None:
        pads = [0 for _ in range(dim * 2)]
    if strides is None:
        strides = [1 for _ in range(dim)]
 
    retShape = shape[:-dim]
    dimIdx = 0
    for dimSize in shape[-dim:]:
        padUpper = pads[dimIdx * 2]
        padLower = pads[dimIdx * 2 + 1]
        stride = strides[dimIdx]
        kernelDimSize = kernelShape[dimIdx]
        retShape.append((dimSize + padUpper + padLower - kernelDimSize) // stride + 1)
        dimIdx = dimIdx + 1

    return retShape


def BatchNormalization(shape, scale = None, B = None, mean = None, var = None):
    return shape

def Concat(shapeList, axis):
    newDimSize = sum([x[axis] for x in shapeList])
    newShape = shapeList[0]  
    newShape[axis] = newDimSize
    return newShape

def Conv(shapeX, shapeW, auto_pad = None, dilations = None, group = None, kernel_shape = None, pads = None, strides = None):
    #Don't support auto_pad current!
    #                             2018-02-28
    #if group is None:
    #    group = 1
    #  group is not support yet too.
    kernelDim = len(shapeX) - 2
    if kernel_shape is None:
        kernel_shape = shapeW[2:] #[[1 for _ in range(kernelDimSize)] for _ in range(kernelDimSize)]
    if pads is None:
        [0 for _ in range(kernelDim * 2)]
    if strides is None:
        [1 for _ in range(kernelDim)]
    if pads is None:
        pads = [0 for _ in range(kernelDim * 2)]
    if strides is None:
        strides = [1 for _ in range(kernelDim)]
    if dilations is None:
        dilations = [1 for _ in range(kernelDim)]
 
    retShape = [shapeX[0], shapeW[0]]  
    dimIdx = 0
    for dimSize in shapeX[2:]:
        padUpper = pads[dimIdx * 2]
        padLower = pads[dimIdx * 2 + 1]
        stride = strides[dimIdx]
        dilation = dilations[dimIdx]
        kernelDimSize = (kernel_shape[dimIdx] - 1) // 2 * dilation * 2 + 1
        retShape.append((dimSize + padUpper + padLower - kernelDimSize) // stride + 1)
        dimIdx = dimIdx + 1
    return retShape

def GlobalAveragePool(shapeX):
    return shapeX[:2] + [1, 1]

def MaxPool(shape, auto_pad = None, kernelShape = None, pads = None, strides = None):
    return AveragePool(shape, auto_pad, kernelShape, pads, strides)

def Mul(shapeX, shapeW, axis = None, broadcast = None):
    return shapeX

def Relu(shape):
    return shape

def FC(shapeX, shapeW, shapeB = None, axis = None, axis_w = None):
    if axis is None:
        axis = 1
    if axis_w is None:
        axis_w = 1
    return [shapeX[0], shapeW[1]]

inference_shape = {
    'Add' : Add,
    'AveragePool' : AveragePool,
    'BatchNormalization' : BatchNormalization,
    'Concat' : Concat,
    'Conv' : Conv,
    'GlobalAveragePool' : GlobalAveragePool,
    'MaxPool' : MaxPool,
    'Mul' : Mul,
    'Relu' : Relu,
    'FC' : FC
}


if __name__ == '__main__':

    shape = [1, 9, 9, 9]

    print('input shape is  : ', shape)
    print('output shape is : ', AveragePool(shape, pads=[1,1,1,1], strides=[2,2]))
    print(inference_shape['AveragePool'](shape, pads=[1,1,1,1], kernelShape=[2,2], strides=[2,2]));

    print('input shape is  : ', shape)
    print('output shape is : ', AveragePool(shape, pads=[0,0,0,0], kernelShape=[3,3], strides=[3,3]))

    shape = [3, 9, 9]
    print('input shape is  : ', shape)
    print('output shape is : ', AveragePool(shape))


    x = [1, 1, 5, 5]
    W = [1, 1, 3, 3]
    print('input shapeX is :', x, 'input shapeW is :', W)
    print('output shape is :', Conv(x, W), "without pads")

    W = [2, 1, 3, 3]
    print('input shapeX is :', x, 'input shapeW is :', W)
    print('output shape is :', Conv(x, W, pads=[1,1,1,1]), 'pads is [1, 1, 1, 1]')


    shape1 = [1, 1, 3, 3]
    shape2 = [1, 3, 3, 3]
    shape3 = [1, 5, 3, 3]
    print('output shape is :', Concat([shape1, shape2, shape3], 1))

    shape = [5, 5, 5, 5]
    print("output shape is :", GlobalAveragePool(shape))
