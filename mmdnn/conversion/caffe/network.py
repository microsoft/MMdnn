import numpy as np

DEFAULT_PADDING = 'SAME'


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        assert len(args) >= 1
        if len(args) == 1:
            layer_inputs = args[0]
        else:
            layer_inputs = list(args) 
        layer_output = op(self, layer_inputs, **kwargs)
        # print('op: %s   shape: %s' % (op, layer_output._keras_shape))
        # print('op: %s   shape: %s' % (op, layer_output.get_shape().as_list()))
        # Add to layer LUT.
        self.layers[name] = layer_output
        self.output = layer_output
        return layer_output

    return layer_decorated


class Network(object):

    def __init__(self, trainable=False):
        self.output = None
        self.layers = {}
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be implemented by the subclass')

    def load(self, data_path, session, ignore_missing=False):
        raise NotImplementedError('Must be implemented by the subclass')
    
    def input(self, shape, name):
        raise NotImplementedError('Must be implemented by the subclass')

    def get_output(self):
        raise NotImplementedError('Must be implemented by the subclass')

    def get_unique_name(self, prefix):
        raise NotImplementedError('Must be implemented by the subclass')
    
    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, p_h, p_w, name, group=1, biased=True):
        raise NotImplementedError('Must be implemented by the subclass')
    
    @layer
    def deconv(self, input, c_o, k_h, k_w, s_h, s_w, p_h, p_w, name):
        raise NotImplementedError('Must be implemented by the subclass')

    @layer
    def relu(self, input, name):
        raise NotImplementedError('Must be implemented by the subclass')

    @layer
    def sigmoid(self, input, name):
        raise NotImplementedError('Must be implemented by the subclass')

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, p_h, p_w, name):
        raise NotImplementedError('Must be implemented by the subclass')

    @layer
    def max_unpool(self, input, k_h, k_w, s_h, s_w, p_h, p_w, name):
        raise NotImplementedError('Must be implemented by the subclass')

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, p_h, p_w, name):
        raise NotImplementedError('Must be implemented by the subclass')

    @layer
    def lrn(self, input, local_size, alpha, beta, name, bias=1):
        raise NotImplementedError('Must be implemented by the subclass')

    @layer
    def concat(self, inputs, axis, name):
        raise NotImplementedError('Must be implemented by the subclass')
    
    @layer
    def add(self, inputs, name):
        raise NotImplementedError('Must be implemented by the subclass')

    @layer
    def fc(self, input, num_out, name):
        raise NotImplementedError('Must be implemented by the subclass')

    @layer
    def softmax(self, input, name):
        raise NotImplementedError('Must be implemented by the subclass')

    @layer
    def batch_normalization(self, input, name, epsilon=0.00001, scale_offset=True):
        raise NotImplementedError('Must be implemented by the subclass')

    @layer
    def dropout(self, input, keep_prob, name):
        raise NotImplementedError('Must be implemented by the subclass')
    
    @layer
    def crop(self, inputs, offset, name):
        raise NotImplementedError('Must be implemented by the subclass')