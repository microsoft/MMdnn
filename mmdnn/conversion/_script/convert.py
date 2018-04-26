import sys as _sys
import argparse
import mmdnn.conversion._script.convertToIR as convertToIR
import mmdnn.conversion._script.IRToCode as IRToCode
import mmdnn.conversion._script.IRToModel as IRToModel
from six import text_type as _text_type
import uuid
import os


def _get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--srcFramework', '-sf',
        type=_text_type,
        choices=["caffe", "caffe2", "cntk", "mxnet", "keras", "tensorflow", 'tf', 'pytorch'],
        help="Source toolkit name of the model to be converted.")
    parser.add_argument(
        '--inputWeight', '-iw',
        type=_text_type,
        default=None,
        help='Path to the model weights file of the external tool (e.g caffe weights proto binary, keras h5 binary')
    parser.add_argument(
        '--inputNetwork', '-in',
        type=_text_type,
        default=None,
        help='Path to the model network file of the external tool (e.g caffe prototxt, keras json')
    parser.add_argument(
        '--dstFramework', '-df',
        type=_text_type,
        choices=['caffe', 'caffe2', 'cntk', 'mxnet', 'keras', 'tensorflow', 'coreml', 'pytorch', 'onnx'],
        required=True,
        help='Format of model at srcModelPath (default is to auto-detect).')
    parser.add_argument(
        '--outputModel', '-om',
        type=_text_type,
        required=True,
        help='Path to save the destination model')
    return parser


def _extract_ir_args(args, unknown_args, temp_filename):
    unknown_args.extend(['--srcFramework', args.srcFramework])
    if args.inputWeight is not None:
        unknown_args.extend(['--weights', args.inputWeight])
    if args.inputNetwork is not None:
        unknown_args.extend(['--network', args.inputNetwork])
    unknown_args.extend(['--dstPath', temp_filename])

    ir_parser = convertToIR._get_parser()
    return ir_parser.parse_known_args(unknown_args)


def _extract_code_args(args, unknown_args, temp_filename,network_filename):
    unknown_args.extend(['--dstFramework', args.dstFramework])
    unknown_args.extend(['--IRModelPath', temp_filename + '.pb'])
    unknown_args.extend(['--IRWeightPath', temp_filename + '.npy'])
    unknown_args.extend(['--dstModelPath', network_filename + '.py'])
    unknown_args.extend(['--dstWeightPath', temp_filename + '.npy'])
    code_parser = IRToCode._get_parser()
    return code_parser.parse_known_args(unknown_args)


def _extract_model_args(args, unknown_args, temp_filename):
    unknown_args.extend(['--framework', args.dstFramework])
    unknown_args.extend(['--inputNetwork', temp_filename + '.pb'])
    unknown_args.extend(['--inputWeight', temp_filename + '.npy'])
    unknown_args.extend(['--output', args.outputModel])
    model_parser = IRToModel._get_parser()
    return model_parser.parse_known_args(unknown_args)


def remove_temp_files(temp_filename, verbose=False):
    exts = ['.json', '.pb', '.npy', '.py']
    # exts = ['.pb', '.npy', '.py']
    for ext in exts:
        temp_file = temp_filename + ext
        if os.path.isfile(temp_file):
            os.remove(temp_file)
            if verbose:
                print('temporary file [{}] has been removed.'.format(temp_file))

def get_network_filename(framework, temp_filename, output_model_filename):
    if framework in ['pytorch']:
        return os.path.join(os.path.dirname(output_model_filename), os.path.basename(output_model_filename).split('.')[0])
    return temp_filename


def _main():
    parser = _get_parser()
    args, unknown_args = parser.parse_known_args()
    temp_filename = uuid.uuid4().hex
    ir_args, unknown_args = _extract_ir_args(args, unknown_args, temp_filename)
    ret = convertToIR._convert(ir_args)
    if int(ret) != 0:
        _sys.exit(int(ret))
    if args.dstFramework != 'coreml':
        network_filename = get_network_filename(args.dstFramework, temp_filename, args.outputModel)
        code_args, unknown_args = _extract_code_args(args, unknown_args, temp_filename, network_filename)
        ret = IRToCode._convert(code_args)
        if int(ret) != 0:
            _sys.exit(int(ret))
        from mmdnn.conversion._script.dump_code import dump_code
        dump_code(args.dstFramework, network_filename + '.py', temp_filename + '.npy', args.outputModel)
        remove_temp_files(temp_filename)

    else:
        model_args, unknown_args = _extract_model_args(args, unknown_args, temp_filename)
        ret = IRToModel._convert(model_args)
        remove_temp_files(temp_filename)
        _sys.exit(int(ret))


if __name__ == '__main__':
    _main()
