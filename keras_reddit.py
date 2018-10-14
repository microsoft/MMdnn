import keras
from keras.models import Model
from keras import layers
import keras.backend as K
import numpy as np
import keras.layers.core as core


def load_weights_from_file(weight_file):
    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


def set_layer_weights(model, weights_dict):
    for layer in model.layers:
        if layer.name in weights_dict:
            cur_dict = weights_dict[layer.name]
            current_layer_parameters = list()
            if layer.__class__.__name__ == "BatchNormalization":
                if 'scale' in cur_dict:
                    current_layer_parameters.append(cur_dict['scale'])
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
                current_layer_parameters.extend([cur_dict['mean'], cur_dict['var']])
            elif layer.__class__.__name__ == "Scale":
                if 'scale' in cur_dict:
                    current_layer_parameters.append(cur_dict['scale'])
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
            elif layer.__class__.__name__ == "SeparableConv2D":
                current_layer_parameters = [cur_dict['depthwise_filter'], cur_dict['pointwise_filter']]
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
            else:
                # rot weights
                current_layer_parameters = [cur_dict['weights']]
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
            model.get_layer(layer.name).set_weights(current_layer_parameters)

    return model


def KitModel(weight_file = None):
    weights_dict = load_weights_from_file(weight_file) if not weight_file == None else None
        
    Placeholder     = layers.Input(name = 'Placeholder', shape = (40, 40,) , dtype = 'int32')
    rnnlm_1_concat  = layers.concatenate(name = 'rnnlm_1/concat', inputs = [])
    rnnlm_embedding_lookup = layers.Embedding(input_dim = 128, output_dim = 6144, mask_zero = False)(Placeholder)
    model           = Model(inputs = [Placeholder, rnnlm_1_concat], outputs = [rnnlm_1_transpose])
    set_layer_weights(model, weights_dict)
    return model
