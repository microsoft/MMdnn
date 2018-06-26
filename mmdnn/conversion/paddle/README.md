# Paddle README

Paddle parser (Paddle -> IR part) is an experimental product and Paddle Emit is coming soon. The parser version is chosen as paddle v2 model instead of fuild version.

We tested the [paddle model zoo pre-trained models](https://github.com/PaddlePaddle/models/blob/develop/image_classification/models/model_download.sh#L28-L38) and the parser works. Any contribution is welcome.

|    Models    | Caffe | CoreML | CNTK | Keras | MXNet | PyTorch | TensorFlow | ONNX |
| :----------: | :---: | :----: | :--: | :---: | :---: | :-----: | :--------: | :--: |
|     Alexnet  |       |    √   |   √  |   √   |   √   |    √    |      √     |   √  |
|     Vgg16    |       |    √   |   √  |   √   |   √   |    √    |      √     |   √  |
|     ResNet50 |       |    √   |   √  |   √   |   √   |    √    |      √     |   √  |
|    ResNet101 |       |    √   |   √  |   √   |   √   |    √    |      √     |   √  |


**√** - Correctness tested

**o** - Some difference after conversion

**space** - not tested



## Dump the protobuf of the network

In order to be user-friendly, the paddle parser used `**.bin` for the conversion. 

One might follow this to dump his/her network.
```python
from paddle.utils.dump_v2_config import dump_v2_config
from mnist_v2 import network

predict = network(is_infer=True)
dump_v2_config(predict, "trainer_config.bin", True)
```

`***.bin` is actually the protobuf of the network, which contains the structure of network and layer information in the network. 

The current method treats the network protobuf and parameters differently. That means the user has to give the path of his or her network (`**.bin`) and the parameters (`**.tar.gz`).

Another approach is to combine the network and parameters in one file. This is made possible by [`merge_v2_model`](http://www.paddlepaddle.org/docs/develop/documentation/zh/howto/capi/workflow_of_capi_cn.html).

### Build the graph

The protobuf information in the `**.bin` is used to build the graph. 

### Build the parser
The information spec of neural layer is also contained in the `**.bin`. It is extracted to set the parameters for each layer. The weights for such layers like `conv`, `fc` or `bn` can be obtained from  `**.tar.gz`. 


## Tips or trouble-shooting
Currently, the main pillars for the neural network are contained in the paddlepaddle parser.

Some tips for the shape inference in paddle parser:

- Since the shape in Paddlepaddle is always one dim, we need to infer the shape of every layer. In fact, the data is channel-first, and flatten to one dim. The inference is done by brute force. Say, for the layer of conv, we get the channel, width and height, and then concatenate them into a list.


Some tips for the future contributor on `padding defuse`:
- The conv padding is now parsed using the explicit way instead of 'SAME' or 'VALID'. For paddle2tf example, a conv with 7x7 kernel, stride by 2 and (3, 3) padding from 224x224 to 112x112 , if converted using 'SAME' padding, then the conv in tensorflow will using the `BOTTOM_RIGHT_HEAVY` mode with (2, 3) padding. That will lead to a big difference.
- The pool padding is using both explicit way and inexplicit way. For example, when converting paddle to tensorflow, a pool with kernel of 3x3, stride by 2 and (0,0) padding from 112x112 to 56x56, if converted using padding (0, 0) padding, the output of this pool will give the shape of 55x55. In this way, using the 'SAME' padding can solve the problem. Therefore, we use some special cases in the parser to solve this problem at least partially.

Other tricks involved: 
- [`reset_parser()`](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/tests/test_rnn_layer.py#L35) is used at the beginning to get rid of existing layers.
- [`paddle.init(use_gpu=False, trainer_count=1)`](https://github.com/PaddlePaddle/Paddle/issues/3533) is used to get rid of the crash.
- `import tensorflow as tf` in the middle might lead to `floating point exception`.
- [`parse_network()`](https://github.com/lcy-seso/paddle_example/blob/master/seq_slice_demo/test_seq_slice.py#L55) is used to get the protobuf of network.
- `class_dim` is set to be `1000` or `1001`. That is the label of imagenet classification. For the vgg16 conversion from paddlepaddle to tensorflow, the class_dim is chosen to be `1001`. The error reaches to `1e-5`. However, inferring from the last `fullyconnected` layer's dim, the last layer's dim should be `1000`. But when the class_dim is set to be `1000`, the error is bigger. The question is that if the `fc` layer's units does not match the size of weight `w`, the framework (like `pytorch`) does not work, though tensorflow works. 



## Things needed
- ~~the api for the paddle is needed for the pytest or conversion~~
- The emit of paddle is yet to be implemented.

## Things yet to check
I find it puzzled between `img_x == width, img_y == height` and `img_y == width, img_x == height`. It might not arouse error currently because usually the width is equal to the height.

## Usage

Usage page is under construction.


## Develop version

MacOS 10.13.3 with

- Paddle 0.11.0

@ 2018/06/21

### Attention
The installation of paddle can be tricky at first. One has to change the `libpython` in order to Install Paddlepaddle. The problem is that the Paddlepaddle might link to the Anaconda Python library. One can refer to [Paddle Technique](https://github.com/PaddlePaddle/Paddle/issues/5401) for more details.


## Limitation

- Currently, no RNN related operations support.

## Link
- [fuild-onnx](https://github.com/PaddlePaddle/paddle-onnx)
- [caffe-paddle](https://github.com/PaddlePaddle/models/tree/develop/image_classification/caffe2paddle)
- [tf-paddle](https://github.com/PaddlePaddle/models/tree/develop/image_classification/tf2paddle)
- [paddle model zoo](https://github.com/PaddlePaddle/models)
- issue about the pretrained model: [paddle](https://github.com/PaddlePaddle/Paddle/issues/11650), [paddle model](https://github.com/PaddlePaddle/models/issues/1001)
- [dump the model](http://www.paddlepaddle.org/docs/develop/documentation/zh/howto/capi/workflow_of_capi_cn.html)
- [trouble shooting](http://www.cnblogs.com/charlotte77/p/8270710.html)
- [paddle tutorial](https://github.com/PaddlePaddle/board/wiki/PaddlePaddle-vs-Tensorflow-api-and-concepts)
