import tensorflow as tf
from keras import backend as K
from PIL import Image

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w * min(w/image_w, h/image_h))
    new_h = int(image_h * min(w/image_w, h/image_h))
    resized_image = image.resize((new_w,new_h), Image.BICUBIC)

    boxed_image = Image.new('RGB', size, (128,128,128))
    boxed_image.paste(resized_image, ((w-new_w)//2,(h-new_h)//2))
    return boxed_image

def yolo_head(feats, anchors, num_classes, input_shape):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    conv_dims = K.shape(feats)[1:3]
    conv_height_index = K.arange(0, stop=conv_dims[1])
    conv_width_index = K.arange(0, stop=conv_dims[0])
    conv_height_index = K.tile(conv_height_index, [conv_dims[0]])

    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[1], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    conv_dims = K.cast(conv_dims[::-1], K.dtype(feats))

    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # Adjust preditions to each spatial grid point and anchor size.
    # Note: YOLO iterates over height index before width index.
    # TODO: It works with +1, don't know why.
    box_xy = (box_xy + conv_index + 1) / conv_dims
    box_wh = box_wh * anchors_tensor / K.cast(input_shape[::-1], K.dtype(box_wh))

    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    # print("feats,anchors, num_classes, input_shape", feats, anchors, num_classes, input_shape)
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    # print(box_xy, box_wh, box_confidence, box_class_probs)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
        anchors,
        num_classes,
        image_shape,
        max_boxes=20,
        score_threshold=.6,
        iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    # yolo_outputs order 13,26,52

    input_shape = K.shape(yolo_outputs[0])[1:3] * 32

    for i in range(0,3):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[i],
            anchors[6-3*i:9-3*i], num_classes, input_shape, image_shape)
        if i==0:
            boxes, box_scores = _boxes, _box_scores
        else:
            boxes = K.concatenate([boxes,_boxes], axis=0)
            box_scores = K.concatenate([box_scores,_box_scores], axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    for i in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, i])
        class_box_scores = tf.boolean_mask(box_scores[:, i], mask[:, i])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * i
        if i==0:
            boxes_, scores_, classes_ = class_boxes, class_box_scores, classes
        else:
            boxes_ = K.concatenate([boxes_,class_boxes], axis=0)
            scores_ = K.concatenate([scores_,class_box_scores], axis=0)
            classes_ = K.concatenate([classes_,classes], axis=0)
    return boxes_, scores_, classes_