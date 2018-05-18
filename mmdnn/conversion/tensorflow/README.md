# TensorFlow README

TensorFlow parser (Tensorflow -> IR part) is an experimental product, since the granularity of TensorFlow checkpoint graph is much finer than other platform. We have to use *graph matching*-like method to retrieve operators.

We tested the [slim pre-trained models](https://github.com/tensorflow/models/tree/master/research/slim) and the parser works. Any contribution is welcome.

|    Models    | Caffe | CoreML | CNTK | Keras | MXNet | PyTorch | TensorFlow | ONNX |
| :----------: | :---: | :----: | :--: | :---: | :---: | :-----: | :--------: | :--: |
|     Vgg19    |   √   |    √   |   √  |   √   |   √   |    √    |      √     |   √  |
| Inception_v1 |   o   |    √   |   o  |   √   |   √   |    √    |      √     |   √  |
| Inception_v3 |   x   |    √   |   √  |   o   |   √   |    √    |      √     |   √  |
|   ResNet V1  |   x   |    √   |   o  |   √   |   √   |    √    |      √     |   √  |
|   ResNet V2  |   x   |    √   |   √  |   √   |   √   |    √    |      √     |   √  |
| MobileNet V1 |   x   |    √   |   o  |   √   |   √   |    √    |      √     |   √  |
|   NasNet-A   |   x   |        |      |       |   √   |    √    |      √     |   √  |

**√** - Correctness tested

**o** - Some difference after conversion

**space** - not tested

## Usage

We provide some tools to help you convert TensorFlow model.

### Download TensorFlow pre-trained model
#### 1.Slim Model

You can refer [Slim Model Extractor](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/examples/tensorflow/extractor.py) to extract your own tensorflow model, which is a sample tool to extract both architecture and weights from slim pre-trained models.

Support frameworks: ['vgg16', 'vgg19', 'inception_v1', 'inception_v1_frozen', 'inception_v3', 'inception_v3_frozen', 'resnet_v1_50', 'resnet_v1_152', 'resnet_v2_50', 'resnet_v2_152', 'resnet_v2_200', 'mobilenet_v1_1.0', 'mobilenet_v1_1.0_frozen', 'mobilenet_v2_1.0_224', 'inception_resnet_v2', 'nasnet-a_large']

Example:

```bash
$ mmdownload -f tensorflow -n resnet_v2_152

Downloading file [./resnet_v2_152_2017_04_14.tar.gz] from [http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz]
100% [......................................................................] 675629399 / 675629399
Model saved in file: ./imagenet_resnet_v2_152.ckpt
```

> [Note!] The extractor create a Squeeze operator named **MMdnn_Output** as the output node of the models.

Then you can see files *imagenet_resnet_v2_152.ckpt.data-00000-of-00001*, *imagenet_resnet_v2_152.ckpt.index* and *imagenet_resnet_v2_152.ckpt.meta*, which can be handled by Tensorflow parser.

Mainly extract code like:

```python
with slim.arg_scope(...):
    data_input = tf.placeholder(name='input', dtype=tf.float32, shape=[...])
    logits = your_own_network_builder(data_input)
    if logits.op.type == 'Squeeze':
        labels = tf.identity(logits, name='MMdnn_Output')
    else:
        labels = tf.squeeze(logits, name='MMdnn_Output')
```

#### 2. Frozen Protobuf Model
There is also another type of pre-trained model(frozen protobuf model file).

Example:

```bash
$ mmdownload -f tensorflow -n inception_v1_frozen

Downloading file [./inception_v1_2016_08_28_frozen.pb.tar.gz] from [https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz]
progress: 24120.0 KB downloaded, 100%

```
The names of input node and destination node are saved in [extractor](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/examples/tensorflow/extractor.py).

---

### Meta File Graph Visualization

When you prepared your checkpoint, you can find the output node name from the graph by Tensorboard.

```bash
$ mmvismeta imagenet_resnet_v2_152.ckpt.meta ./logs/

TensorBoard 0.4.0rc3 at http://kit-station:6006 (Press CTRL+C to quit)
```

![tensorboard](https://nxtb0g.dm2304.livefilestore.com/y4mSQWnEhuXOj67Bsv-nFS7kocOD0JmGRFJsUIrZCDRfO6CIP1-wUBana8wydOM3ZHgoVe_wR_KXq_hX6sCg_D_6H93F3oQMUjfu_VjbYswl_dX2mBolqts1zG9_eA483i_BokvfQknb9JQYvOwcwJvrPVH9GI2L_0GJoxJpYGw0kFDxmzICwjc-j_wHKwdiZUyS32CBCVBS67qZlTgFuPiHA?width=1024&height=676&cropmode=none)


## One-step conversion

Above MMdnn@0.1.4, we provide one command to achieve the conversion

```bash
$  mmconvert -sf tensorflow -in imagenet_inception_v1.ckpt.meta -iw inception_v1.ckpt -df cntk -om tf_inception_v1.dnn --inputShape 3 224 224 --dstNodeName MMdnn_Output
.
.
.
Parse file [imagenet_inception_v1.ckpt.meta] with binary format successfully.
Tensorflow model file [imagenet_inception_v1.ckpt.meta] loaded successfully.
Tensorflow checkpoint file [inception_v1.ckpt] loaded successfully. [231] variables loaded.
IR network structure is saved as [b136dfb62a964486bb2bc2e27cece6e8.json].
IR network structure is saved as [b136dfb62a964486bb2bc2e27cece6e8.pb].
IR weights are saved as [b136dfb62a964486bb2bc2e27cece6e8.npy].
Parse file [b136dfb62a964486bb2bc2e27cece6e8.pb] with binary format successfully.
Target network code snippet is saved as [b136dfb62a964486bb2bc2e27cece6e8.py].
CNTK model file is saved as [tf_inception_v1.dnn], generated by [b136dfb62a964486bb2bc2e27cece6e8.py] and [b136dfb62a964486bb2bc2e27cece6e8.npy].

```

Then you get the CNTK original model *tf_inception_v1.dnn* converted from Tensorflow. Temporal files are removed automatically.

---

## Step-by-step conversion (for debugging)

### Convert only architecture(*.ckpt.meta) from Tensorflow to IR

You can convert only network structure to IR for visualization or training in other frameworks.

We use resnet_v2_152 model as an example.

```bash
$ mmtoir -f tensorflow -d resnet152 -n imagenet_resnet_v2_152.ckpt.meta --dstNodeName MMdnn_Output

Parse file [imagenet_resnet_v2_152.ckpt.meta] with binary format successfully.
Tensorflow model file [imagenet_resnet_v2_152.ckpt.meta] loaded successfully.
IR network structure is saved as [resnet152.json].
IR network structure is saved as [resnet152.pb].
Warning: weights are not loaded.
```

### Convert model including architecture(*.ckpt.meta) and weights(.ckpt) from Tensorflow to IR

You can use following bash command to convert the checkpoint files to IR architecture file [*resnet152.pb*], [*resnet152.json*] and IR weights file [*resnet152.npy*]

```bash
$ mmtoir -f tensorflow -d resnet152 -n imagenet_resnet_v2_152.ckpt.meta -w imagenet_resnet_v2_152.ckpt  --dstNodeName MMdnn_Output

Parse file [imagenet_resnet_v2_152.ckpt.meta] with binary format successfully.
Tensorflow model file [imagenet_resnet_v2_152.ckpt.meta] loaded successfully.
Tensorflow checkpoint file [imagenet_resnet_v2_152.ckpt] loaded successfully. [816] variables loaded.
IR network structure is saved as [resnet152.json].
IR network structure is saved as [resnet152.pb].
IR weights are saved as [resnet152.npy].
```

### Convert frozen protobuf model file from Tensorflow to IR

You can convert frozen protobuf model file to IR for visualization or training in other frameworks.

We use resnet_v2_152 model as an example.

```bash
$ mmtoir -f tensorflow --frozen_pb inception_v1_2016_08_28_frozen.pb -d inception_v1 --dstNodeName InceptionV1/Logits/Predictions/Reshape_1 --inputShape 224 224 3

IR network structure is saved as [inception_v1.json].
IR network structure is saved as [inception_v1.pb].
IR weights are saved as [inception_v1.npy].
```

### Convert models from IR to Tensorflow code snippet

The generated Tensorflow code snippet can restore weights from IR weights file directly, but we need the tensors' shape information to infer some parameters.

```bash
$ mmtocode -f tensorflow --IRModelPath resnet152.pb --IRWeightPath resnet152.npy --dstModelPath tf_resnet152.py

Parse file [resnet152.pb] with binary format successfully.
Target network code snippet is saved as [tf_resnet152.py].
```

You can refer the example tool to test your converted model. In this case we use the Tensorflow -> IR -> Tensorflow resnet_v2_152 model as an example.

```bash
$ python -m mmdnn.conversion.examples.tensorflow.imagenet_test -s tensorflow -p resnet -n tf_resnet152 -w resnet152.npy
.
.
.
[(387, 14.552185), (102, 11.523594), (386, 7.2283654), (500, 4.6292458), (899, 2.8113561)]
Test model [resnet] from [tf] passed.
```

The information shows that the output result of **"Squeeze"** layers from original slim model and converted model are same.

### Convert models from IR to Tensorflow model

After generating the Tensorflow code snippet, you can convert the Tensorflow code snippet and IR weights file to Tensorflow original model for further usage.

```bash
$ python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n tf_resnet152.py -w resnet152.npy --dump tf_resnet152.ckpt
.
.
.
Tensorflow file is saved as [tf_resnet152.ckpt], generated by [tf_resnet152.py] and [resnet152.npy].
```

## Develop version

Ubuntu 16.04 with

- Tensorflow 1.7

@ 2018/05/17

## Limitation

- Currently no RNN related operations support.