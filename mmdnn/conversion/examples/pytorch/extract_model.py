#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import argparse
import os
from six import text_type as _text_type
from mmdnn.conversion.examples.imagenet_test import TestKit
import torch
import torchvision.models as models


NETWORKS_MAP = {
    'inception_v3'      : lambda : models.inception_v3(pretrained=True),
    'vgg16'             : lambda : models.vgg16(pretrained=True),
    'vgg19'             : lambda : models.vgg19(pretrained=True),
    'resnet152'         : lambda : models.resnet152(pretrained=True),
    'densenet'          : lambda : models.densenet201(pretrained=True),
    'squeezenet'        : lambda : models.squeezenet1_1(pretrained=True)
}


def _main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--network',
                        type=_text_type, help='Model Type', required=True,
                        choices=NETWORKS_MAP.keys())

    parser.add_argument('-i', '--image', type=_text_type, help='Test Image Path')

    args = parser.parse_args()

    file_name = "imagenet_{}.pth".format(args.network)
    if not os.path.exists(file_name):
        model = NETWORKS_MAP.get(args.network)
        model = model()
        torch.save(model, file_name)
        print("PyTorch pretrained model is saved as [{}].".format(file_name))
    else:
        print("File [{}] existed!".format(file_name))
        model = torch.load(file_name)

    if args.image:
        import numpy as np
        func = TestKit.preprocess_func['pytorch'][args.network]
        img = func(args.image)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0).copy()
        data = torch.from_numpy(img)
        data = torch.autograd.Variable(data, requires_grad=False)

        model.eval()
        predict = model(data).data.numpy()
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
