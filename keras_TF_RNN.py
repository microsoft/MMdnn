import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.embedding_1 = self.__embedding('embedding_1', num_embeddings=30000, embedding_dim=125)
        self.dense_1 = self.__dense(name = 'dense_1', in_features = 100, out_features = 2, bias = True)

    def forward(self, x):
        embedding_1     = self.embedding_1(torch.LongTensor(np.array(x)))
        gru_1_activation = F.tanh(gru_1)
        dense_1         = self.dense_1(gru_1_activation)
        dense_1_activation = F.softmax(dense_1)
        return dense_1_activation


    @staticmethod
    def __embedding(name, **kwargs):
        layer = nn.Embedding(**kwargs) #shape
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        return layer
        

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer
