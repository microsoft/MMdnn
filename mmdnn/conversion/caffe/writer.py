import base64
from google.protobuf import json_format
from importlib import import_module
import json
import numpy as np
import os
import sys

from mmdnn.conversion.caffe.errors import ConversionError
from mmdnn.conversion.caffe.common_graph import fetch_attr_value
from mmdnn.conversion.caffe.utils import get_lower_case, get_upper_case, get_real_name


class JsonFormatter(object):
    '''Dumpt a DL graph into a Json file.'''

    def __init__(self, graph):
        self.graph_def = graph.as_graph_def()

    def dump(self, json_path):
        json_txt = json_format.MessageToJson(self.graph_def)
        parsed = json.loads(json_txt)
        formatted = json.dumps(parsed, indent=4, sort_keys=True)
        with open(json_path, 'w') as f:
            f.write(formatted)

    
class PyWriter(object):
    '''Dumpt a DL graph into a Python script.'''

    def __init__(self, graph, data, target):
        self.graph = graph
        self.data = data
        self.tab = ' ' * 4
        self.prefix = ''
        target = target.lower()
        if target == 'tensorflow':
            self.target = target
            self.net = 'TensorFlowNetwork'
        elif target == 'keras':
            self.target = target
            self.net = 'KerasNetwork'
        elif target == 'caffe':
            self.target = target
            self.net = 'CaffeNetwork'
        else:
            raise ConversionError('Target %s is not supported yet.' % target)

    def indent(self):
        self.prefix += self.tab

    def outdent(self):
        self.prefix = self.prefix[:-len(self.tab)]

    def statement(self, s):
        return self.prefix + s + '\n'

    def emit_imports(self):
        return self.statement('from dlconv.%s import %s\n' % (self.target, self.net))

    def emit_class_def(self, name):
        return self.statement('class %s(%s):' % (name, self.net))

    def emit_setup_def(self):
        return self.statement('def setup(self):')

    def emit_node(self, node):
        '''Emits the Python source for this node.'''
        
        def pair(key, value):
            return '%s=%s' % (key, value)
        args = []
        for input in node.input:
            input = input.strip().split(':')
            name = ''.join(input[:-1])
            idx = int(input[-1])
            assert name in self.graph.node_dict
            parent = self.graph.get_node(name)
            args.append(parent.output[idx])
        #FIXME:
        output = [node.output[0]]
        # output = node.output
        for k, v in node.attr:
            if k == 'cell_type':
                args.append(pair(k, "'" + fetch_attr_value(v) + "'"))
            else:
                args.append(pair(k, fetch_attr_value(v)))
        args.append(pair('name', "'" + node.name + "'")) # Set the node name
        args = ', '.join(args)
        return self.statement('%s = self.%s(%s)' % (', '.join(output), node.op, args))

    def dump(self, code_output_dir):
        if not os.path.exists(code_output_dir):
            os.makedirs(code_output_dir)
        file_name = get_lower_case(self.graph.name)
        code_output_path = os.path.join(code_output_dir, file_name + '.py')
        data_output_path = os.path.join(code_output_dir, file_name + '.npy')
        with open(code_output_path, 'w') as f:
            f.write(self.emit())
        with open(data_output_path, 'wb') as f:
            np.save(f, self.data)
        return code_output_path, data_output_path

    def emit(self):
        # Decompose DAG into chains
        chains = []
        for node in self.graph.topologically_sorted():
            attach_to_chain = None
            if len(node.input) == 1:
                parent = get_real_name(node.input[0])
                for chain in chains:
                    if chain[-1].name == parent: # Node is part of an existing chain.
                        attach_to_chain = chain
                        break
            if attach_to_chain is None: # Start a new chain for this node.
                attach_to_chain = []
                chains.append(attach_to_chain)
            attach_to_chain.append(node)
            
        # Generate Python code line by line
        source = self.emit_imports()
        source += self.emit_class_def(self.graph.name)
        self.indent()
        source += self.emit_setup_def()
        self.indent()
        blocks = []
        for chain in chains:
            b = ''
            for node in chain:
                b += self.emit_node(node)
            blocks.append(b[:-1])
        source += '\n\n'.join(blocks)
        return source


class ModelSaver(object):

    def __init__(self, code_output_path, data_output_path):
        self.code_output_path = code_output_path
        self.data_output_path = data_output_path

    def dump(self, model_output_dir):
        '''Return the file path containing graph in generated model files.'''
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        sys.path.append(os.path.dirname(self.code_output_path))
        file_name = os.path.splitext(os.path.basename(self.code_output_path))[0]
        module = import_module(file_name)
        class_name = get_upper_case(file_name)
        net = getattr(module, class_name)
        return net.dump(self.data_output_path, model_output_dir)


class GraphDrawer(object):

    def __init__(self, toolkit, meta_path):
        self.toolkit = toolkit.lower()
        self.meta_path = meta_path

    def dump(self, graph_path):
        if self.toolkit == 'tensorflow':
            from dlconv.tensorflow.visualizer import TensorFlowVisualizer
            if self._is_web_page(graph_path):
                TensorFlowVisualizer(self.meta_path).dump_html(graph_path)
            else:
                raise NotImplementedError('Image format or %s is unsupported!' % graph_path)
        elif self.toolkit == 'keras':
            from dlconv.keras.visualizer import KerasVisualizer
            png_path, html_path = (None, None)
            if graph_path.endswith('.png'):
                png_path = graph_path
            elif self._is_web_page(graph_path):
                png_path = graph_path + ".png"
                html_path = graph_path
            else:
                raise NotImplementedError('Image format or %s is unsupported!' % graph_path)
            KerasVisualizer(self.meta_path).dump_png(png_path)
            if html_path:
                self._png_to_html(png_path, html_path)
                os.remove(png_path)
        else:
            raise NotImplementedError('Visualization of %s is unsupported!' % self.toolkit)

    def _is_web_page(self, path):
        return path.split('.')[-1] in ('html', 'htm')

    def _png_to_html(self, png_path, html_path):
        with open(png_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        source = """<!DOCTYPE>
<html>
    <head>
        <meta charset="utf-8">
        <title>Keras</title>
    </head>
    <body>
        <img alt="Model Graph" src="data:image/png;base64,{base64_str}" />
    </body>
</html>""".format(base64_str=encoded)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(source)