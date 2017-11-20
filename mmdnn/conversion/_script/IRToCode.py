import sys as _sys
import google.protobuf.text_format as text_format
from six import text_type as _text_type



def _convert(args):
    if args.dstModelFormat == 'caffe':
        raise NotImplementedError("Destination [Caffe] is not implemented yet.")

    elif args.dstModelFormat == 'keras':
        from mmdnn.conversion.keras.keras2_emitter import Keras2Emitter
        emitter = Keras2Emitter(args.IRModelPath)

    elif args.dstModelFormat == 'tensorflow':
        from mmdnn.conversion.tensorflow.tensorflow_emitter import TensorflowEmitter
        if args.IRWeightPath == None:        
            emitter = TensorflowEmitter(args.IRModelPath)
        else:
            emitter = TensorflowEmitter((args.IRModelPath, args.IRWeightPath))
    
    elif args.dstModelFormat == 'cntk':
        from mmdnn.conversion.cntk.cntk_emitter import CntkEmitter
        if args.IRWeightPath == None:
            emitter = CntkEmitter(args.IRModelPath)
        else:
            emitter = CntkEmitter((args.IRModelPath, args.IRWeightPath))

    elif args.dstModelFormat == 'coreml':
        raise NotImplementedError("CoreML emitter is not finished yet.")
        assert args.IRWeightPath != None
        from mmdnn.conversion.coreml.coreml_emitter import CoreMLEmitter
        emitter = CoreMLEmitter((args.IRModelPath, args.IRWeightPath))
        model = emitter.gen_model()
        print ("Saving the CoreML model [{}].".format(args.dstModelPath + '.mlmodel'))
        model.save(args.dstModelPath + '.mlmodel')
        print ("The converted CoreML model saved as [{}].".format(args.dstModelPath + '.mlmodel'))
        return 0
    
    elif args.dstModelFormat == 'pytorch':
        if not args.dstWeightPath or not args.IRWeightPath:
            raise ValueError("Need to set a target weight filename.")
        from mmdnn.conversion.pytorch.pytorch_emitter import PytorchEmitter
        emitter = PytorchEmitter((args.IRModelPath, args.IRWeightPath))        

    elif args.dstModelFormat == 'mxnet':
        from mmdnn.conversion.mxnet.mxnet_emitter import MXNetEmitter
        if args.IRWeightPath == None:
            emitter = MXNetEmitter(args.IRModelPath)
        else:
            emitter = MXNetEmitter((args.IRModelPath, args.IRWeightPath, args.inputShape, args.dstWeightPath))
        
    else:
        assert False
    
    emitter.run(args.dstModelPath, args.dstWeightPath, args.phase)

    return 0


def _main():
    import argparse

    parser = argparse.ArgumentParser(description = 'Convert IR model file formats to other format.')
    
    parser.add_argument(
        '--phase',
        type = _text_type,
        choices = ['train', 'test'],
        default = 'test',
        help = 'Convert phase (train/test) for destination toolkits.'
    )
    
    parser.add_argument(
        '--dstModelFormat', '-f',
        type = _text_type,
        choices = ['caffe', 'caffe2', 'cntk', 'mxnet', 'keras', 'tensorflow', 'coreml', 'pytorch'], 
        required = True,
        help = 'Format of model at srcModelPath (default is to auto-detect).')

    parser.add_argument(
        '--IRModelPath', '-n',
        type = _text_type,
        required = True, 
        help = 'Path to the IR network structure file.')

    parser.add_argument(
        '--IRWeightPath', '-w',
        type = _text_type,
        required = False,
        default = None,
        help = 'Path to the IR network structure file.')

    parser.add_argument(
        '--dstModelPath', '-d',
        type = _text_type,
        required = True, 
        help = 'Path to save the destination model')

    # MXNet
    parser.add_argument(
        '--dstWeightPath', '-dw',
        type = _text_type,
        default = None,
        help = '[MXNet] Path to save the destination weight.')

    parser.add_argument(
        '--inputShape',
        nargs = '+',
        type = int,
        default = None,
        help = '[MXNet] Input shape of model (batch, channel, height, width).')

    args = parser.parse_args()
    ret = _convert(args)
    _sys.exit(int(ret)) # cast to int or else the exit code is always 1


if __name__ == '__main__':
    _main()