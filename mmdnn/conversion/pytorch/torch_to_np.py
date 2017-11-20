import torchfile
import numpy as np

model = torchfile.load('kit.model')

weights = dict()

params = ['weight', 'bias', 'running_mean', 'running_var']

recursive = ['conv_nets']

def save_weight(name, node, level):
    weights[name] = dict()
    current_layer = weights[name]
    for p in params:
        if hasattr(node, p):
            func = getattr(node, p)
            arr = np.array(func)
            if arr.ndim >= 1:
                current_layer[p] = arr
                print ("    " * level + "{}.{} shape {} {}".format(name, p, current_layer[p].shape, current_layer[p].dtype))
    
    for p in recursive:
        if hasattr(node, p):
            func = getattr(node, p)
            if func != None:
                for idx, subnode in enumerate(func):
                    new_name = name + ":{}:{}".format(p, idx)
                    save_weight(new_name, subnode, level + 1)


for idx, data in enumerate(model.modules):
    if data != None:
        print ("Find layer #{} : {}".format(idx, data._typename))
        if hasattr(data, 'search_flag'):
            print ("    name = {}".format(data.search_flag))    
        if data.modules != None:
            print ("    submodule = {}#".format(len(data.modules)))
            for idx_j, sub in enumerate(data.modules):
                print ("        layer [{}]".format(sub._typename))
                name = data.search_flag + ":" + str(idx_j)
                save_weight(name, sub, 2)
                print ("\n")
        else:
            pass
            #print (dir(data))

        print ("\n")

with open("stylebank.npy", 'wb') as of:
    np.save(of, weights)

print ("-------------------------------------------------")

load_weight = np.load('stylebank.npy').item()
for i in load_weight:
    #print (i)
    for j in load_weight[i]:
        pass
        #print ("    {} with shape {}".format(j, load_weight[i][j].shape))