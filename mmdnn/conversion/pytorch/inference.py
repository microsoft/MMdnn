# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import argparse
import numpy as np
import sys
import os
import caffe

def inference(args):
    net = caffe.Net(args.network, args.weights, caffe.TEST)

    from tensorflow.contrib.keras.python.keras.preprocessing import image
    image_path = 'mmdnn/conversion/examples/data/seagull.jpg'

    if args.preprocess == 'vgg':
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        # Zero-center by mean pixel
        x = x[..., ::-1]
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

        x = np.transpose(x, [2, 0, 1])

    elif args.preprocess == 'resnet' or args.preprocess == 'inception':
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x /= 255.0
        x -= 0.5
        x *= 2.0
        x = np.transpose(x, [2, 0, 1])

    else:
        assert False

    x = np.expand_dims(x, 0)
    net.blobs['data'].data[...] = x
    predict = np.squeeze(net.forward()['prob'][0])

    test = 'pool1/norm1'
    immediate_data = net.blobs[test].data[0]
    print (immediate_data)
    print (immediate_data.shape)
    print ("%.30f" % np.sum(np.array(immediate_data)))

    top_indices = predict.argsort()[-5:][::-1]
    result = [(i, predict[i]) for i in top_indices]
    print (result)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--preprocess',
                        type=str, choices = ["vgg", "resnet", "inception"], help='Model Preprocess Type', required=False, default='vgg')

    parser.add_argument('-n', '--network',
                        type=str, required=True)

    parser.add_argument('-w', '--weights',
                        type=str, required=True)


    args = parser.parse_args()

    inference(args)
