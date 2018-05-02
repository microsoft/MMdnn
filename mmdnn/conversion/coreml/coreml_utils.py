#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from coremltools.models import datatypes

def _infer_coreml_input_shape(IR_shape, if_convert=True):
    """Infer CoreML input shape from IR shape.
    """
    if len(IR_shape) == 0:
        # the end of the tensorflow_resnet_v2_50's squeeze shape is [unknown_rank: true] with len 0
        # 1001 means the 1001 classes for tensorflow_resnet_v2_50
        # !Alert! TODO
        # Future implement can be changed to the last two layer
        shape = [1001,1,1]
    elif len(IR_shape) == 1:
        # TODO - remove style transfer 1D hack
        # Input is 1D but it goes to the width dimension: (1,1,W)
        shape = [1, 1, IR_shape[0]]  #(C,H,W)
    elif len(IR_shape) == 2:
        # assume (Batch, Channels) - Batch dimension should be dropped
        shape = [IR_shape[1]]
    elif len(IR_shape) == 3:
        # assume (Batch, Sequence-Length, channels)
        shape = [IR_shape[2], 1, IR_shape[1]]
    elif len(IR_shape) == 4:   #(B,H,W,C) --> (C,H,W)
        shape = [IR_shape[3], IR_shape[1], IR_shape[2]] #(C,H,W)
    else:
        raise ValueError('Unrecognized IR input shape {}'.format(shape))
    if if_convert:
        shape = datatypes.Array(*shape)
    return shape
