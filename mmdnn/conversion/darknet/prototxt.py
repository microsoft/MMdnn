from collections import OrderedDict

def parse_prototxt(protofile):
    def line_type(line):
        if line.find(':') >= 0:
            return 0
        elif line.find('{') >= 0:
            return 1
        return -1

    def parse_block(fp):
        block = OrderedDict()
        line = fp.readline().strip()
        while line != '}':
            ltype = line_type(line)
            if ltype == 0: # key: value
                #print line
                line = line.split('#')[0]
                key, value = line.split(':')
                key = key.strip()
                value = value.strip().strip('"')
                if key in block.keys():
                    if type(block[key]) == list:
                        block[key].append(value)
                    else:
                        block[key] = [block[key], value]
                else:
                    block[key] = value
            elif ltype == 1: # blockname {
                key = line.split('{')[0].strip()
                sub_block = parse_block(fp)
                block[key] = sub_block
            line = fp.readline().strip()
            line = line.split('#')[0]
        return block

    fp = open(protofile, 'r')
    props = OrderedDict()
    layers = []
    line = fp.readline()
    while line != '':
        line = line.strip().split('#')[0]
        if line == '':
            line = fp.readline()
            continue
        ltype = line_type(line)
        if ltype == 0: # key: value
            key, value = line.split(':')
            key = key.strip()
            value = value.strip().strip('"')
            if key in props.keys():
               if type(props[key]) == list:
                   props[key].append(value)
               else:
                   props[key] = [props[key], value]
            else:
                props[key] = value
        elif ltype == 1: # blockname {
            key = line.split('{')[0].strip()
            if key == 'layer':
                layer = parse_block(fp)
                layers.append(layer)
            else:
                props[key] = parse_block(fp)
        line = fp.readline()

    if len(layers) > 0:
        net_info = OrderedDict()
        net_info['props'] = props
        net_info['layers'] = layers
        return net_info
    else:
        return props

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
