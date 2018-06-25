# Paddle README

Paddle parser (Paddle -> IR part) is an experimental product and Paddle Emit is coming soon. The parser version is chosen as paddle v2 model instead of fuild version.

We tested the [paddle model zoo pre-trained models](https://github.com/PaddlePaddle/models/blob/develop/image_classification/models/model_download.sh#L28-L38) and the parser works. Any contribution is welcome.

|    Models    | Caffe | CoreML | CNTK | Keras | MXNet | PyTorch | TensorFlow | ONNX |
| :----------: | :---: | :----: | :--: | :---: | :---: | :-----: | :--------: | :--: |
|     Alexnet  |   √   |    √   |   √  |   √   |   √   |    √    |      √     |   √  |
|     Vgg19    |   √   |    √   |   √  |   √   |   √   |    √    |      √     |   √  |
|     ResNet   |   √   |    √   |   √  |   √   |   √   |    √    |      √     |   √  |


**√** - Correctness tested

**o** - Some difference after conversion

**space** - not tested


## build the graph

We use the parent information of the node to build the graph. The [code](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/paddle/paddle_graph.py#L52-L62) here builds the build layerwise.

## build the parser

Since the paddlepaddle node only contains the frontend to the user. The backend information can be got from the [`parse_network`](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/paddle/paddle_parser.py#L97-L105). The `self.spec_dict` is the dict, whose key is the name of the node. This will show the information spec of neural layer.

Since the shape in Paddlepaddle is always one dim, we need to infer the shape of every layer. In fact, the data is channel-first, and flatten to one dim. The inference is done by brute force. Say, for the layer of conv, we get the channel, width and height, and then concatenate them into a list.

Currently, the main pillars for the neural network are contained in the paddlepaddle parser.

Some tips for the future contributor on `padding defuse`.
- The conv padding is now parsed using the explicit way instead of 'SAME' or 'VALID'. For paddle2tf example, a conv with 7x7 kernel, stride by 2 and (3, 3) padding from 224x224 to 112x112 , if converted using 'SAME' padding, then the conv in tensorflow will using the `BOTTOM_RIGHT_HEAVY` mode with (2, 3) padding. That will lead to a big difference.
- The pool padding is using both explicit way and inexplicit way. For example, when converting paddle to tensorflow, a pool with kernel of 3x3, stride by 2 and (0,0) padding from 112x112 to 56x56, if converted using padding (0, 0) padding, the output of this pool will give the shape of 55x55. In this way, using the 'SAME' padding can solve the problem. Therefore, we use some special cases in the parser to solve this problem at least partially.

## Things needed
- the api for the paddle is needed for the pytest or conversion
- The emit of paddle is yet to be implemented.

## Things yet to check
I find it puzzled between `img_x == width, img_y == height` and `img_y == width, img_x == height`. It might not arouse error because usually the width is equal to the height.

## Usage

Usage page is under construction.


## Develop version

MacOS 10.13.3 with

- Paddle 0.11.0

@ 2018/06/21

### Attention
The installation of paddle can be tricky at first. One has to change the `libpython` in order to Install Paddlepaddle. The problem is that the Paddlepaddle might link to the Anaconda Python library. One can refer to [Paddle Technique](https://github.com/PaddlePaddle/Paddle/issues/5401) for more details.


## Limitation

- Currently no RNN related operations support.

## Link
- [fuild-onnx](https://github.com/PaddlePaddle/paddle-onnx)
- [caffe-paddle](https://github.com/PaddlePaddle/models/tree/develop/image_classification/caffe2paddle)
- [tf-paddle](https://github.com/PaddlePaddle/models/tree/develop/image_classification/tf2paddle)
- [paddle model zoo](https://github.com/PaddlePaddle/models)
- issue about the pretrained model: [paddle](https://github.com/PaddlePaddle/Paddle/issues/11650), [paddle model](https://github.com/PaddlePaddle/models/issues/1001)
- [dump the model](http://www.paddlepaddle.org/docs/develop/documentation/zh/howto/capi/workflow_of_capi_cn.html)