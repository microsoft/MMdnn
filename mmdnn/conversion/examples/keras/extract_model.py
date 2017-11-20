#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import argparse
from six import text_type as _text_type

def _main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-n', '--network',
        type = _text_type, choices = ['vgg16', 'vgg19', 'inception_v3', 'resnet50', 'mobilenet', 'xception'], help='Model Type', required = True)

    args = parser.parse_args()
    
    import keras
    choices = {
        'inception_v3' : lambda : keras.applications.inception_v3.InceptionV3(input_shape=(299,299,3)),
        'vgg16'        : lambda : keras.applications.vgg16.VGG16(),
        'vgg19'        : lambda : keras.applications.vgg19.VGG19(),
        'resnet50'     : lambda : keras.applications.resnet50.ResNet50(),
        'mobilenet'    : lambda : keras.applications.mobilenet.MobileNet(),
        'xception'     : lambda : keras.applications.xception.Xception()
    }    
    model = choices.get(args.network)
    if model == None:
        raise NotImplementedError("Unknown keras application [{}]".format(args.network))
    model = model()

    # save network structure as JSON
    json_string = model.to_json()
    with open("imagenet_{}.json".format(args.network), "w") as of:
        of.write(json_string)

    print ("Network structure is saved as [imagenet_{}.json].".format(args.network))

    model.save_weights('imagenet_{}.h5'.format(args.network))
    
    print ("Network weights are saved as [imagenet_{}.h5].".format(args.network))


if __name__=='__main__':
    _main()