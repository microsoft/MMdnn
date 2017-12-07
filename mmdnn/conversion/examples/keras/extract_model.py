#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import argparse
from six import text_type as _text_type
import keras
from mmdnn.conversion.examples.imagenet_test import TestKit

networks_map = {
    'inception_v3'      : lambda : keras.applications.inception_v3.InceptionV3(input_shape=(299, 299, 3)),
    'vgg16'             : lambda : keras.applications.vgg16.VGG16(),
    'vgg19'             : lambda : keras.applications.vgg19.VGG19(),
    'resnet'            : lambda : keras.applications.resnet50.ResNet50(),
    'mobilenet'         : lambda : keras.applications.mobilenet.MobileNet(),
    'xception'          : lambda : keras.applications.xception.Xception(input_shape=(299, 299, 3)),
    'inception_resnet'  : lambda : keras.applications.inception_resnet_v2.InceptionResNetV2()
}

image_size = {
    'inception_v3'      : 299,
    'vgg16'             : 224,
    'vgg19'             : 224,
    'resnet'            : 224,
    'mobilenet'         : 224,
    'xception'          : 299,
    'inception_resnet'  : 299
}

def _main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--network',
                        type=_text_type, help='Model Type', required=True,
                        choices=networks_map.keys())

    parser.add_argument('-i', '--image',
                        type=_text_type, help='Test Image Path')

    args = parser.parse_args()

    model = networks_map.get(args.network)
    if model is None:
        raise NotImplementedError("Unknown keras application [{}]".format(args.network))

    model = model()
    # save network structure as JSON
    json_string = model.to_json()
    with open("imagenet_{}.json".format(args.network), "w") as of:
        of.write(json_string)

    print("Network structure is saved as [imagenet_{}.json].".format(args.network))

    model.save_weights('imagenet_{}.h5'.format(args.network))

    print("Network weights are saved as [imagenet_{}.h5].".format(args.network))

    if args.image:
        import numpy as np
        func = TestKit.preprocess_func['keras'][args.network]
        img = func(args.image)
        img = np.expand_dims(img, axis=0)
        predict = model.predict(img)
        predict = np.squeeze(predict)
        top_indices = predict.argsort()[-5:][::-1]
        result = [(i, predict[i]) for i in top_indices]
        print(result)

        # layer_name = 'block2_pool'
        # intermediate_layer_model = keras.Model(inputs=model.input,
        #                                  outputs=model.get_layer(layer_name).output)
        # intermediate_output = intermediate_layer_model.predict(img)
        # print (intermediate_output)
        # print (intermediate_output.shape)
        # print ("%.30f" % np.sum(intermediate_output))


if __name__ == '__main__':
    _main()
