#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from six import text_type as _text_type


def extract_model(args):
    if args.framework == 'caffe':
        from mmdnn.conversion.examples.caffe.extractor import caffe_extractor
        extractor = caffe_extractor()

    elif args.framework == 'keras':
        from mmdnn.conversion.examples.keras.extractor import keras_extractor
        extractor = keras_extractor()

    elif args.framework == 'tensorflow' or args.framework == 'tf':
        from mmdnn.conversion.examples.tensorflow.extractor import tensorflow_extractor
        extractor = tensorflow_extractor()

    elif args.framework == 'mxnet':
        from mmdnn.conversion.examples.mxnet.extractor import mxnet_extractor
        extractor = mxnet_extractor()

    elif args.framework == 'cntk':
        from mmdnn.conversion.examples.cntk.extractor import cntk_extractor
        extractor = cntk_extractor()

    elif args.framework == 'pytorch':
        from mmdnn.conversion.examples.pytorch.extractor import pytorch_extractor
        extractor = pytorch_extractor()

    else:
        raise ValueError("Unknown framework [{}].".format(args.framework))

    files = extractor.download(args.network, args.path)

    if files and args.image:
        predict = extractor.inference(args.network, files, args.path, args.image)
        top_indices = predict.argsort()[-5:][::-1]
        result = [(i, predict[i]) for i in top_indices]
        print(result)


def _main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract pre-trained models for frameworks.')

    parser.add_argument(
        '--framework', '-f',
        type=_text_type,
        required=True,
        choices=["caffe", "cntk", "mxnet", "keras", "tensorflow", 'tf', 'pytorch'],
        help="Framework name")

    parser.add_argument(
        '--network', '-n',
        type=_text_type,
        default=None,
        help='Path to the model network file of the external tool (e.g caffe prototxt, keras json')

    parser.add_argument(
        '-i', '--image',
        type=_text_type, help='Test Image Path')

    parser.add_argument(
        '--path', '-p', '-o',
        type=_text_type,
        default='./',
        help='Path to save the pre-trained model files (e.g keras h5)')

    args = parser.parse_args()
    extract_model(args)


if __name__ == '__main__':
    _main()
