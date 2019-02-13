from mmdnn.conversion.rewriter.graph_matcher import *
from mmdnn.conversion.tensorflow.tensorflow_graph import *
import numpy as np


'''batch size pattern in tensorflow. Note: do not include _number in name'''
static_rnn_batch_size_pattern = OpTypePattern('ExpandDims', name='static_rnn_batch_size', inputs=[
    OpTypePattern('StridedSlice', inputs=[
        OpTypePattern('Shape', inputs=[
            OpTypePattern('*', name='input')
        ]),
        OpTypePattern('Const'),
        OpTypePattern('Const'),
        OpTypePattern('Const')
    ]),
    OpTypePattern('Const')
])

'''rnn h zero pattern in tensorflow.'''
static_rnn_h_zero_pattern = OpTypePattern('Fill', name='h_zero', inputs=[
    OpTypePattern('ConcatV2|Concat', inputs=[
        OpTypePattern('*', name='input'),
        OpTypePattern('Const', name='fill_size'),
        OpTypePattern('Const')
    ]),
    OpTypePattern('Const', name='fill_value')
])

''''split pattern in gru cell in tensorflow'''
gru_xc_pattern = OpTypePattern('Split', name='xc', inputs=[
    OpTypePattern("Const"), # axis for split
    OpTypePattern("Sigmoid", inputs=[
        OpTypePattern("BiasAdd", name="bias_add", inputs=[
            OpTypePattern("MatMul", inputs=[
                OpTypePattern("ConcatV2|Concat", name="xh"),
                OpTypePattern("Identity", name='cell_kernel')
            ]),
        OpTypePattern("Identity", name='cell_bias')
    ])]),
])

'''split pattern in lstm cell in tensorflow'''
lstm_xc_pattern = OpTypePattern('Split', inputs=[
    OpTypePattern("Const"), # axis for split
    OpTypePattern("BiasAdd", name="bias_add", inputs=[
        OpTypePattern("MatMul", inputs=[
            OpTypePattern("ConcatV2|Concat", name="xh"),
            OpTypePattern("*", name="cell_kernel"),
        ]),
        OpTypePattern("*", name="cell_bias"),
    ]),
])

''''gru cell pattern in tensorflow'''
grucell_pattern = \
    OpTypePattern('Add', name='gru_cell', inputs=[
        OpTypePattern('Mul', inputs=[
            gru_xc_pattern,
            OpTypePattern('*', name='input')
        ]),
        OpTypePattern('Mul', inputs=[
            OpTypePattern('Sub', inputs=[
                OpTypePattern('Const'),
                gru_xc_pattern
            ]),
            OpTypePattern('Tanh', inputs=[
                OpTypePattern('BiasAdd', inputs=[
                    OpTypePattern('MatMul', name='FullyConnect', inputs=[
                        OpTypePattern('Concat|ConcatV2', inputs=[
                            OpTypePattern('*', name='input'),
                            OpTypePattern('Mul', inputs=[
                                gru_xc_pattern,
                                OpTypePattern('*', name='input')
                            ]),
                            OpTypePattern('Const'),
                        ]),

                        OpTypePattern('Identity', name='candidate_kernel')
                    ]),
                    OpTypePattern('Identity', name='candidate_bias')
                ])
            ])
        ])
    ])


''''lstm cell pattern in tensorflow'''
lstmcell_pattern = \
    OpTypePattern('Mul', name='lstm_cell', inputs=[
        OpTypePattern("Sigmoid", name="ot", inputs=[lstm_xc_pattern]),
        OpTypePattern('Tanh', inputs=[
            OpTypePattern("Add", name="ct", inputs=[
                OpTypePattern("Mul", inputs=[
                    OpTypePattern("Sigmoid", name="ft", inputs=[
                        OpTypePattern("Add", inputs=[
                            lstm_xc_pattern,
                            OpTypePattern("*", name="ft_bias"),
                        ]),
                    ]),
                    OpTypePattern("*", name='input'),
                ]),
                OpTypePattern("Mul", inputs=[
                    OpTypePattern("Sigmoid", name="it", inputs=[lstm_xc_pattern]),
                    OpTypePattern("Tanh", name="gt", inputs=[lstm_xc_pattern]),
                ]),
            ]),
        ]),
    ])


rnn_patterns = {
    'tensorflow': {
        'gru_cell': grucell_pattern,
        'lstm_cell': lstmcell_pattern,
        'h_zero': static_rnn_h_zero_pattern,
        'static_rnn_batch_size': static_rnn_batch_size_pattern
    }
    # TODO: pytorch, mxnet, keras, cntk
}