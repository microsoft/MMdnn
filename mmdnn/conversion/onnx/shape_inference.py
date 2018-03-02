
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

def Flatten(shapeT, axis = None):
    if axis is None:
        axis = 1

    firstDim = 1
    secondDim = 1
    for i in range(len(shapeT)):
        if i < axis:
            firstDim *= shapeT[i]
        else:
            secondDim *= shapeT[i]
    
    if (axis > 0):
        return [firstDim, secondDim]
    else:
        return [secondDim]

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
    'FC' : FC,
    'Flatten' : Flatten
}

def testByLeNet(image_shape):
    print('\nLeNet output shape test:')
    print('input_image_shape is : ', image_shape)
    convLay1 = [5, 5]
    WLay1 = [6, -1, 5, 5]
    outputLay1 = inference_shape['Conv'](image_shape, WLay1, kernel_shape = convLay1)
    print('1st Lay output shape is : ', outputLay1)  

    poolLay2 = [2, 2]
    stridesLay2 = [2, 2]
    outputLay2 = inference_shape['AveragePool'](outputLay1, strides = stridesLay2)
    print('2nd Lay output shape is : ', outputLay2)

    convLay3 = [5, 5]
    WLay3 = [16, -1, 5, 5]
    outputLay3 = inference_shape['Conv'](outputLay2, WLay3, kernel_shape = convLay3)
    print('3rd Lay output shape is : ', outputLay3)

    poolLay4 = [2, 2]
    stridesLay4 = [2, 2]
    outputLay4 = inference_shape['AveragePool'](outputLay3, strides = stridesLay4)
    print('4th Lay output shape is : ', outputLay4)

    convLay5 = [5, 5]
    WLay5 = [120, -1, 5, 5]
    outputLay5 = inference_shape['Conv'](outputLay4, WLay5)
    print('5th Lay output shape is : ', outputLay5)
    
    outputLay5Flatten = inference_shape['Flatten'](outputLay5)
    WLay6 = [-1, 84]
    outputLay6 = inference_shape['FC'](outputLay5Flatten, WLay6)
    print('6th Lay output shape is : ', outputLay6)

    WLay7 = [-1, 10]
    outputLay7 = inference_shape['FC'](outputLay6, WLay7)
    print('7th Lay output shape is : ', outputLay7)
    return outputLay7


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
    
    print('LeNet-5 output shape is : ', testByLeNet(image_shape = [-1, 1, 32, 32]))
    print('LeNet-5 output shape is : ', testByLeNet(image_shape = [5, 1, 32, 32]))



