import sys as _sys
from six import text_type as _text_type
import sys
import os.path


def dump_code(framework, network_filepath, weight_filepath, dump_filepath):
    if network_filepath.endswith('.py'):
        network_filepath = network_filepath[:-3]
    sys.path.insert(0, os.path.dirname(os.path.abspath(network_filepath)))
    MainModel = __import__(network_filepath)
    if framework == 'caffe':
        from mmdnn.conversion.caffe.saver import save_model
    elif framework == 'cntk':
        from mmdnn.conversion.cntk.saver import save_model
    elif framework == 'keras':
        from mmdnn.conversion.keras.saver import save_model
    elif framework == 'mxnet':
        from mmdnn.conversion.mxnet.saver import save_model
    elif framework == 'pytorch':
        from mmdnn.conversion.pytorch.saver import save_model
    elif framework == 'tensorflow':
        from mmdnn.conversion.tensorflow.saver import save_model
    elif framework == 'onnx':
        from mmdnn.conversion.onnx.saver import save_model
    else:
        raise NotImplementedError("{} saver is not finished yet.".format(framework))
    save_model(MainModel, network_filepath, weight_filepath, dump_filepath)

    return 0


def _get_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Dump the model code into target model.')

    parser.add_argument(
        '-f', '--framework', type=_text_type, choices=["caffe", "cntk", "mxnet", "keras", "tensorflow", 'torch', 'onnx'],
        required=True,
        help='Format of model at srcModelPath (default is to auto-detect).'
    )

    parser.add_argument(
        '-in', '--inputNetwork',
        type=_text_type,
        required=True,
        help='Path to the model network architecture file.')

    parser.add_argument(
        '-iw', '--inputWeight',
        type=_text_type,
        required=True,
        help='Path to the model network weight file.')

    parser.add_argument(
        '-o', '-om', '--outputModel',
        type=_text_type,
        required=True,
        help='Path to save the target model')
    return parser


def _main():
    parser = _get_parser()
    args = parser.parse_args()
    ret = dump_code(args.framework, args.inputNetwork, args.inputWeight, args.outputModel)
    _sys.exit(int(ret))


if __name__ == '__main__':
    _main()
