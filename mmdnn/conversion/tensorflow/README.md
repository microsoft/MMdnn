# Tensorflow README

## Usage

Tensorflow parser (Tensorflow -> IR part) is an experimental product, since the granularity of tensorflow checkpoint graph is much finer than other platform. We have to use *graph matching*-like method to retrieve operators.

We tested the [slim pre-trained models](https://github.com/tensorflow/models/tree/master/research/slim) and the parser works. Any contribution is welcome.

### Sample Tools

We provide some tools to help you convert tensorflow files.

#### Slim Model Extractor

You can refer [Slim Model Extractor](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/examples/tensorflow/extractor.py) to extract your own tensorflow model, which is a sample tool to extract both architecture and weights from slim pre-trained models.

Support frameworks: ['resnet_v1_152', 'inception_v3', 'resnet_v2_50', 'resnet_v2_200', 'resnet_v1_50', 'inception_resnet_v2', 'resnet_v2_152', 'vgg19', 'mobilenet_v1_1.0', 'inception_v1']

Example:

```bash
$ python -m mmdnn.conversion._script.extractModel -f tensorflow -n resnet_v2_152

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

#### Meta File Graph Visualization

When you prepared your checkpoint, you can find the output node name from the graph by Tensorboard.

```bash
$ python -m mmdnn.conversion.examples.tensorflow.vis_meta imagenet_resnet_v2_152.ckpt.meta ./logs/

TensorBoard 0.4.0rc3 at http://kit-station:6006 (Press CTRL+C to quit)
```

![tensorboard](https://nxtb0g.dm2304.livefilestore.com/y4mSQWnEhuXOj67Bsv-nFS7kocOD0JmGRFJsUIrZCDRfO6CIP1-wUBana8wydOM3ZHgoVe_wR_KXq_hX6sCg_D_6H93F3oQMUjfu_VjbYswl_dX2mBolqts1zG9_eA483i_BokvfQknb9JQYvOwcwJvrPVH9GI2L_0GJoxJpYGw0kFDxmzICwjc-j_wHKwdiZUyS32CBCVBS67qZlTgFuPiHA?width=1024&height=676&cropmode=none)

### Convert only architecture from Tensorflow to IR

You can convert only network structure to IR for visualization or training in other frameworks.

We use resnet_v2_152 model as an example.

```bash
$ python -m mmdnn.conversion._script.convertToIR -f tensorflow -d resnet152 -n imagenet_resnet_v2_152.ckpt.meta --dstNodeName MMdnn_Output

Parse file [imagenet_resnet_v2_152.ckpt.meta] with binary format successfully.
Tensorflow model file [imagenet_resnet_v2_152.ckpt.meta] loaded successfully.
IR network structure is saved as [resnet152.json].
IR network structure is saved as [resnet152.pb].
Warning: weights are not loaded.
```

### Convert model (including architecture and weights) from Tensorflow to IR

You can use following bash command to convert the checkpoint files to IR architecture file [*resnet152.pb*], [*resnet152.json*] and IR weights file [*resnet152.npy*]

```bash
$ python -m mmdnn.conversion._script.convertToIR -f tensorflow -d resnet152 -n imagenet_resnet_v2_152.ckpt.meta -w imagenet_resnet_v2_152.ckpt  --dstNodeName MMdnn_Output

Parse file [imagenet_resnet_v2_152.ckpt.meta] with binary format successfully.
Tensorflow model file [imagenet_resnet_v2_152.ckpt.meta] loaded successfully.
Tensorflow checkpoint file [imagenet_resnet_v2_152.ckpt] loaded successfully. [816] variables loaded.
IR network structure is saved as [resnet152.json].
IR network structure is saved as [resnet152.pb].
IR weights are saved as [resnet152.npy].
```

### Convert models from IR to Tensorflow code snippet

The generated Tensorflow code snippet can restore weights from IR weights file directly, but we need the tensors' shape information to infer some parameters.

```bash
$ python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath resnet152.pb --IRWeightPath resnet152.npy --dstModelPath tf_resnet152.py

Parse file [resnet152.pb] with binary format successfully.
Target network code snippet is saved as [tf_resnet152.py].
```

You can refer the example tool to test your converted model. In this case we use the Tensorflow -> IR -> Tensorflow resnet_v2_152 model as an example.

```bash
$ python -m mmdnn.conversion.examples.tensorflow.imagenet_test -s tf -p resnet -n tf_resnet152 -w resnet152.npy
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

- Tensorflow gpu 1.4.1

@ 11/22/2017

## Limitation

- Currently no RNN related operations support

- Currently no tensorflow frozen graph support.