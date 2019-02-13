import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

def create_symbol(X, num_classes=0, is_training=False, CUDNN=False, 
                  maxf=30000, edim=125, nhid=100, batchs=64):
    word_vectors = tf.contrib.layers.embed_sequence(X, vocab_size=maxf, embed_dim=edim)
    

    word_list = tf.unstack(word_vectors, axis=1)
    
    if not CUDNN:
        cell1 = tf.contrib.rnn.LSTMCell(nhid)
        cell2 = tf.contrib.rnn.GRUCell(nhid)
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
        outputs, states = tf.nn.static_rnn(stacked_cell, word_list, dtype=tf.float32)
        logits = tf.layers.dense(outputs[-1], 2, activation=None, name='output')
    else:
        # Using cuDNN since vanilla RNN
        from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
        cudnn_cell = cudnn_rnn_ops.CudnnGRU(num_layers=1, 
                                            num_units=nhid, 
                                            input_size=edim, 
                                            input_mode='linear_input')
        params_size_t = cudnn_cell.params_size()
        params = tf.Variable(tf.random_uniform([params_size_t], -0.1, 0.1), validate_shape=False)   
        input_h = tf.Variable(tf.zeros([1, batchs, nhid]))
        outputs, states = cudnn_cell(input_data=word_list,
                                     input_h=input_h,
                                     params=params)
        logits = tf.layers.dense(outputs[-1], 2, activation=None, name='output')
    
    return logits, logits

def dummy_arg_scope():
    with slim.arg_scope([]) as sc:
        return sc