#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import TensorShape
from mmdnn.conversion.common.DataStructure.graph import Graph, GraphNode

def load_protobuf_from_file(container, filename):
    with open(filename, 'rb') as fin:
        file_content = fin.read()
    
    # First try to read it as a binary file.    
    try:
        container.ParseFromString(file_content)
        print("Parse file [%s] with binary format successfully." % (filename))
        return container
    except Exception as e:  # pylint: disable=broad-except
        print ("Info: Trying to parse file [%s] with binary format but failed with error [%s]." % (filename, str(e)))

    # Next try to read it as a text file.
    try:
        from google.protobuf import text_format
        text_format.Parse(file_content.decode('UTF-8'), container, allow_unknown_extension=True)
        print("Parse file [%s] with text format successfully." % (filename))
    except text_format.ParseError as e:
        raise IOError("Cannot parse file %s: %s." % (filename, str(e)))

    return container


class IRGraphNode(GraphNode):

    @staticmethod
    def replace_scope(name):
        return name.replace('/', '_')

    @property
    def IR_layer(self):
        return self.layer

    @property
    def name(self):
        return self.layer.name

    @property
    def type(self):
        return self.layer.op

    def set_attrs(self, attrs):
        for name, val in attrs.items():            
            if isinstance(val, bool):
                self.attr[name] = val
            elif isinstance(val, int):
                self.attr[name].i = val
            elif isinstance(val, float):
                self.attr[name].f = val
            elif isinstance(val, str):
                self.attr[name].s = val.encode('utf-8')
            elif isinstance(val, TensorShape):
                self.attr[name].shape.MergeFromString(val.SerializeToString())
            elif isinstance(val, list):
                if len(val) == 0: return

                if isinstance(val[0], int):
                    self.attr[name].list.i.extend(val)
                elif isinstance(val[0], TensorShape):
                    self.attr[name].list.shape.extend(val)
                else:
                    raise NotImplementedError('AttrValue cannot be list of %s' % type(val[0]))
            else:
                raise NotImplementedError('AttrValue cannot be of %s' % type(val))


    def get_attr(self, name, default_value = None):
        if name in self.layer.attr:
            attr = self.layer.attr[name]
            field = attr.WhichOneof('value')
            val = getattr(attr, field) if field else default_value
            return val.decode('utf-8') if isinstance(val, bytes) else val            
        else:
            return default_value


class IRGraph(Graph):
  
    @staticmethod
    def shapeToStr(tensor_shape, keep_minus_one = False):
        ret = ""
        first = True
        for e in tensor_shape.dim:
            if e.size != -1 or keep_minus_one:
                if first == False:
                    ret += ", "
                ret += str(e.size)
                first = False
        return ret

        
    def __init__(self, filename):
        model = graph_pb2.GraphDef()
        load_protobuf_from_file(model, filename)
        super(IRGraph, self).__init__(model)

      
    def filter_node(self):
        for name, layer in self.layer_map.items():
            if len(layer.in_edges) == 0 and len(layer.out_edges) == 0:
                self.layer_map.pop(name)
    
    
    def build(self):        
        for layer in self.model.node:
            self.layer_map[layer.name] = IRGraphNode(layer)
            self.layer_name_map[layer.name] = layer.name
        
        for i, layer in enumerate(self.model.node):
            for pred in layer.input:
                self._make_connection(pred, layer.name)

        self.filter_node()
        super(IRGraph, self).build()