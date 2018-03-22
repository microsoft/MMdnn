import sys as _sys
import google.protobuf.text_format as text_format
from six import text_type as _text_type



def _convert(args):
    if args.framework == 'caffe':
        raise NotImplementedError("Destination [Caffe] is not implemented yet.")

    elif args.framework == 'keras':
        raise NotImplementedError("Destination [Keras] is not implemented yet.")

    elif args.framework == 'tensorflow':
        raise NotImplementedError("Destination [Tensorflow] is not implemented yet.")

    elif args.framework == 'cntk':
        raise NotImplementedError("Destination [Tensorflow] is not implemented yet.")

    elif args.framework == 'coreml':
        from mmdnn.conversion.coreml.coreml_emitter import CoreMLEmitter
        assert args.inputNetwork is not None
        assert args.inputWeight is not None
        emitter = CoreMLEmitter(args.inputNetwork, args.inputWeight)
        model, in_, out_ = emitter.gen_model(
            args.inputNames,
            args.outputNames,
            image_input_names = set(args.imageInputNames) if args.imageInputNames else None,
            is_bgr = args.isBGR,
            red_bias = args.redBias,
            blue_bias = args.blueBias,
            green_bias = args.greenBias,
            gray_bias = args.grayBias,
            image_scale = args.scale,
            class_labels = args.classInputPath if args.classInputPath else None,
            predicted_feature_name = args.predictedFeatureName)

        """
        from google.protobuf import text_format
        with open(args.output+'.txt', 'w') as f:
            f.write(text_format.MessageToString(model))
        """

        with open(args.output, 'wb') as f:
            model = model.SerializeToString()
            f.write(model)


        return 0

    elif args.framework == 'pytorch':
        if not args.dstWeightPath or not args.IRWeightPath:
            raise ValueError("Need to set a target weight filename.")
        from mmdnn.conversion.pytorch.pytorch_emitter import PytorchEmitter
        emitter = PytorchEmitter((args.IRModelPath, args.IRWeightPath))

    elif args.framework == 'mxnet':
        from mmdnn.conversion.mxnet.mxnet_emitter import MXNetEmitter
        if args.IRWeightPath == None:
            emitter = MXNetEmitter(args.IRModelPath)
        else:
            emitter = MXNetEmitter((args.IRModelPath, args.IRWeightPath, args.inputShape, args.dstWeightPath))

    else:
        assert False

    emitter.run(args.output)

    return 0


def _get_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Convert IR model file formats to other format.')

    parser.add_argument(
        '-f', '--framework', type=_text_type, choices=['coreml'], required=True,
        help='Format of model at srcModelPath (default is to auto-detect).'
    )

    parser.add_argument(
        '-in', '--inputNetwork',
        type=_text_type,
        required=True,
        help='Path of the IR network architecture file.')

    parser.add_argument(
        '-iw', '--inputWeight',
        type=_text_type,
        required=True,
        help='Path to the IR network weight file.')

    parser.add_argument(
        '-o', '--output',
        type=_text_type,
        required=True,
        help='Path to save the destination model')

    # Caffe
    parser.add_argument(
        '--phase', type=_text_type, choices=['train', 'test'], default='test',
        help='[Caffe] Convert phase (train/test) for destination toolkits.'
    )

    # For CoreML
    parser.add_argument('--inputNames', type=_text_type, nargs='*', help='Names of the feature (input) columns, in order (required for keras models).')
    parser.add_argument('--outputNames', type=_text_type, nargs='*', help='Names of the target (output) columns, in order (required for keras models).')
    parser.add_argument('--imageInputNames', type=_text_type, default=[], action='append', help='Label the named input as an image. Can be specified more than once for multiple image inputs.')
    parser.add_argument('--isBGR', action='store_true', default=False, help='True if the image data in BGR order (RGB default)')
    parser.add_argument('--redBias', type=float, default=0.0, help='Bias value to be added to the red channel (optional, default 0.0)')
    parser.add_argument('--blueBias', type=float, default=0.0, help='Bias value to be added to the blue channel (optional, default 0.0)')
    parser.add_argument('--greenBias', type=float, default=0.0, help='Bias value to be added to the green channel (optional, default 0.0)')
    parser.add_argument('--grayBias', type=float, default=0.0, help='Bias value to be added to the gray channel for Grayscale images (optional, default 0.0)')
    parser.add_argument('--scale', type=float, default=1.0, help='Value by which the image data must be scaled (optional, default 1.0)')
    parser.add_argument('--classInputPath', type=_text_type, default='', help='Path to class labels (ordered new line separated) for treating the neural network as a classifier')
    parser.add_argument('--predictedFeatureName', type=_text_type, default='class_output', help='Name of the output feature that captures the class name (for classifiers models).')
    return parser


def _main():
    parser=_get_parser()
    args = parser.parse_args()
    ret = _convert(args)
    _sys.exit(int(ret)) # cast to int or else the exit code is always 1


if __name__ == '__main__':
    _main()
