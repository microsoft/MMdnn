import sys as _sys
import google.protobuf.text_format as text_format
from six import text_type as _text_type



def _convert(args):
    if args.dstFramework == 'caffe':
        from mmdnn.conversion.caffe.caffe_emitter import CaffeEmitter
        if args.IRWeightPath is None:
            emitter = CaffeEmitter(args.IRModelPath)
        else:
            assert args.dstWeightPath
            emitter = CaffeEmitter((args.IRModelPath, args.IRWeightPath))

    elif args.dstFramework == 'keras':
        from mmdnn.conversion.keras.keras2_emitter import Keras2Emitter
        emitter = Keras2Emitter(args.IRModelPath)

    elif args.dstFramework == 'tensorflow':
        from mmdnn.conversion.tensorflow.tensorflow_emitter import TensorflowEmitter
        if args.IRWeightPath is None:
            # Convert network architecture only
            emitter = TensorflowEmitter(args.IRModelPath)
        else:
            emitter = TensorflowEmitter((args.IRModelPath, args.IRWeightPath))

    elif args.dstFramework == 'cntk':
        from mmdnn.conversion.cntk.cntk_emitter import CntkEmitter
        if args.IRWeightPath is None:
            emitter = CntkEmitter(args.IRModelPath)
        else:
            emitter = CntkEmitter((args.IRModelPath, args.IRWeightPath))

    elif args.dstFramework == 'coreml':
        raise NotImplementedError("CoreML emitter is not finished yet.")

    elif args.dstFramework == 'pytorch':
        if not args.dstWeightPath or not args.IRWeightPath:
            raise ValueError("Need to set a target weight filename.")
        from mmdnn.conversion.pytorch.pytorch_emitter import PytorchEmitter
        emitter = PytorchEmitter((args.IRModelPath, args.IRWeightPath))

    elif args.dstFramework == 'mxnet':
        from mmdnn.conversion.mxnet.mxnet_emitter import MXNetEmitter
        if args.IRWeightPath is None:
            emitter = MXNetEmitter(args.IRModelPath)
        else:
            if args.dstWeightPath is None:
                raise ValueError("MXNet emitter needs argument [dstWeightPath(dw)], like -dw mxnet_converted-0000.param")
            emitter = MXNetEmitter((args.IRModelPath, args.IRWeightPath, args.dstWeightPath))
    elif args.dstFramework == 'onnx':
        from mmdnn.conversion.onnx.onnx_emitter import OnnxEmitter
        if args.IRWeightPath is None:
            raise NotImplementedError("ONNX emitter needs IR weight file")
        else:
            emitter = OnnxEmitter(args.IRModelPath, args.IRWeightPath)
    else:
        assert False

    emitter.run(args.dstModelPath, args.dstWeightPath, args.phase)

    return 0


def _get_parser():
    import argparse

    parser = argparse.ArgumentParser(description = 'Convert IR model file formats to other format.')

    parser.add_argument(
        '--phase',
        type=_text_type,
        choices=['train', 'test'],
        default='test',
        help='Convert phase (train/test) for destination toolkits.'
    )

    parser.add_argument(
        '--dstFramework', '-f',
        type=_text_type,
        choices=['caffe', 'caffe2', 'cntk', 'mxnet', 'keras', 'tensorflow', 'coreml', 'pytorch', 'onnx'],
        required=True,
        help='Format of model at srcModelPath (default is to auto-detect).')

    parser.add_argument(
        '--IRModelPath', '-n', '-in',
        type=_text_type,
        required=True,
        help='Path to the IR network structure file.')

    parser.add_argument(
        '--IRWeightPath', '-w', '-iw',
        type=_text_type,
        required=False,
        default=None,
        help = 'Path to the IR network structure file.')

    parser.add_argument(
        '--dstModelPath', '-d',
        type = _text_type,
        required = True,
        help = 'Path to save the destination model')

    # MXNet
    parser.add_argument(
        '--dstWeightPath', '-dw', '-ow',
        type=_text_type,
        default=None,
        help='[MXNet] Path to save the destination weight.')
    return parser


def _main():
    parser=_get_parser()
    args = parser.parse_args()
    ret = _convert(args)
    _sys.exit(int(ret)) # cast to int or else the exit code is always 1


if __name__ == '__main__':
    _main()