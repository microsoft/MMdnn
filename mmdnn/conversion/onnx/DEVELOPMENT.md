# Some guidance and tips on developing ONNX emitter

## Why do we need ONNX **master** branch

ONNX [versioned each operator it supported](https://github.com/onnx/onnx/blob/master/docs/Operators.md).
We found that ONNX operator API changed a lot in the latest version 6, and is more compatible with the IR MMdnn defined, so we decided to build the ONNX emitter against ONNX operator API version 6.

Unluckily, the operator version of the latest released ONNX ([v1.1.2](https://github.com/onnx/onnx/releases/tag/v1.1.2)) is 2, rather than the version 6 that the master branch uses.
That's why we need the ONNX **master** branch.

## Why do we need ONNX-TensorFlow

ONNX is defined as "a open format to represent deep learning models", for the ease of developing ONNX emitter, we need to find a backend that runs ONNX directly.
There are currently two ONNX backend implementation that we are able to use: the Caffe2 backend ([ONNX-Caffe2](https://github.com/caffe2/caffe2/tree/master/caffe2/python/onnx)) and the TensorFlow backend ((ONNX-TensorFlow)[https://github.com/onnx/onnx-tensorflow]).

Due to the fact that ONNX-TensorFlow supports more ONNX operators than ONNX-Caffe2 especially for version 6, so we choose ONNX-TensorFlow as the backend to run ONNX emitter tests.

## How can I find the output of intermediate layers

We can use the code [here](https://github.com/onnx/onnx-tensorflow/blob/master/example/test_model_large_stepping.py) to extract the output of the intermediate layers (see [here](https://github.com/onnx/onnx-tensorflow/issues/88) for some more discussion).

Here below is an example to extract the output of each intermediate layers of a Keras-converted ONNX model.

```python
from keras.preprocessing import image
import numpy as np
from keras.applications.mobilenet import preprocess_input, decode_predictions
import onnx
from onnx_tf.backend import prepare
from onnx import helper, TensorProto

model = onnx.load('mobilenet.onnx')

more_outputs = []
output_to_check = []
for node in model.graph.node:
    more_outputs.append(helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, (224, 224)))
    output_to_check.append(node.output[0])
model.graph.output.extend(more_outputs)

tf_rep = prepare(model)

img = image.load_img('elephant.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

predictions = tf_rep.run(x)
preds = predictions[0].reshape((1, 1000))

print('Predicted:', decode_predictions(preds, top=3)[0])

for op in output_to_check:
    np.savetxt(op.replace('/', '__') + '.tf', predictions[op].flatten(), delimiter='\t')
    print(op, predictions[op].shape)
```

_Tips: set a berakpoint to find some more information about the model._

## Some concerns regarding NCHW and NHWC

Notice that the IR MMdnn defined is NHWC (channel last), but the IR ONNX defined is NCHW (channel first).

It brings some difficulty while developing ONNX emitter since channel converting is needed.
We take the strategy that add a Transpose operator at the beginning to change the input data from NHWC to NCHW, and adjust the parameters of channel-related operators (such as `Conv`) from NHWC to NCHW, to solve the channel problem.

We didn't take the strategy that keep the whole data in the original NHWC format but add transpose layers before and after each channel-related operator, since it may bring deviation into the result.

It seems that ONNX is considering to support NHWC ([discussion](https://github.com/onnx/onnx/issues/369)), and MMdnn is considering to support NCHW too.
We are looking forward to that.

## Models that ONNX emitter currently not supported

- Keras `xception`: `SeparableConv` is not supported.
- Keras `nasnet`:  ValueError: You are trying to load a weight file containing 533 layers into a model with 527 layers.
- Caffe `voc-fcn8s`: `Crop` is not supported.
- Caffe `voc-fcn16s`: `Crop` is not supported.
- Caffe `voc-fcn32s`: `Crop` is not supported.

## Operators that have potential problems

### `Slice`

The `strides` attribute is ignored currently due that `strides` is not supported natively in ONNX `Slice`.
