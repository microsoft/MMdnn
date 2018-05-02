#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import argparse
import numpy as np
import sys
import os
from mmdnn.conversion.examples.imagenet_test import TestKit

import colorsys
from keras import backend as K
from PIL import Image, ImageFont, ImageDraw
from mmdnn.conversion.examples.keras.utils import yolo_eval


class TestKeras(TestKit):

    def __init__(self):

        # self.anchors = np.array([[10,13], [16,30],[33,23],[30,61],[62,45], [59,119],[116,90],[156,198],[373,326]])
        self.class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        super(TestKeras, self).__init__()
        self.model = self.MainModel.KitModel(self.args.w)


    def preprocess(self, image_path):
        x = super(TestKeras, self).preprocess(image_path)
        self.data = np.expand_dims(x, 0)

    def print_result(self):
        predict = self.model.predict(self.data)
        super(TestKeras, self).print_result(predict)

    def generate(self):
        self.input_image_shape = K.placeholder(shape=(2, ))
        output = self.model.output
        output.sort(key=lambda x: int(x.shape[1]))
        # print(output)

        boxes, scores, classes = yolo_eval(output, self.anchors,
                    len(self.class_names), self.input_image_shape,
                    score_threshold=self.score_threshold, iou_threshold=self.iou_threshold)
        return boxes, scores, classes

    def yolo_result(self, path):
        image = Image.fromarray(np.uint8(np.squeeze(self.data)))

        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.model.input: self.data/255.,
                self.input_image_shape: [416, 416],
                K.learning_phase(): 0
            })
        # print(out_boxes, out_scores, out_classes)
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # get random colors
            self.colors = []
            C = list(np.random.random_integers(255, size=(len(self.class_names),3)))
            for i in C:
                self.colors.append(tuple(i))

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(tuple(text_origin), label, fill=(0, 0, 0))
            del draw
        image.save("{}.jpg".format(path), "JPEG")


    def print_intermediate_result(self, layer_name, if_transpose = False):
        from keras.models import Model
        intermediate_layer_model = Model(inputs = self.model.input,
                                         outputs = self.model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(self.data)
        super(TestKeras, self).print_intermediate_result(intermediate_output, if_transpose)


    def inference(self, image_path):
        self.preprocess(image_path)

        print(self.data.shape)
        # self.print_intermediate_result('conv1_7x7_s2_1', True)

        self.print_result()

        self.test_truth()

    def dump(self, path = None):
        if path is None: path = self.args.dump
        self.model.save(path)
        print ('Keras model file is saved as [{}], generated by [{}.py] and [{}].'.format(
            path, self.args.n, self.args.w))

    def detect(self, image_path, path = None):
        self.yolo_parameter = self.MainModel.yolo_parameter()
        # yolov3 80 classes
        assert self.yolo_parameter[1] == 80
        self.anchors = []
        for i in range(len(self.yolo_parameter[0])):
            if i%2:
                tmp = [self.yolo_parameter[0][i-1], self.yolo_parameter[0][i]]
                self.anchors.append(tmp)
        self.anchors = np.array(self.anchors)
        self.score_threshold = self.yolo_parameter[2]
        self.iou_threshold = self.yolo_parameter[3]

        self.preprocess(image_path)

        self.yolo_result(path)

        print ('Keras yolo model result file is saved as [{}.jpg], generated by [{}.py] and [{}].'.format(
            path, self.args.n, self.args.w))


if __name__=='__main__':
    tester = TestKeras()
    if tester.args.dump:
        tester.dump()
    elif tester.args.detect:
        tester.detect(tester.args.image, tester.args.detect)
    else:
        tester.inference(tester.args.image)
