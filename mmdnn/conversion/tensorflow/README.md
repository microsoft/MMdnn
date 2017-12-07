# Tensorflow README

## Usage

Tensorflow parser (Tensorflow-IR part) is an experimental product, since the granularity of tensorflow checkpoint graph is much finer than other platform. We have to use *graph matching*-like method to retrieve operators.

We tested the [slim pre-trained models](https://github.com/tensorflow/models/tree/master/research/slim) and the parser works. Any contribution is welcome.

### Extract Tensorflow models

You can refer [slim model extractor](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/examples/tensorflow/extract_model.py) to extract your tensorflow model, which is a example tool to extract both architecture and weights from slim pre-trained models.

We will use the **resnet_v2_152** model as an example.

1. Download the pre-trained checkpoint from [slim page](https://github.com/tensorflow/models/tree/master/research/slim).

```bash
$ wget http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz -P examples/tf/
$ tar -xvf examples/tf/resnet_v2_152_2017_04_14.tar.gz
$ rm examples/tf/resnet_v2_152_2017_04_14.tar.gz
$ mv *.ckpt *.graph examples/tf/
```

Checkpoint **weights** file is stored as *examples/tf/resnet_v2_152.ckpt*

2. Use python to extract both network architecture and weights

```bash
$ python -m mmdnn.conversion.examples.tensorflow.extract_model -n resnet152 -ckpt examples/tf/resnet_v2_152.ckpt
.
.
.
Model saved in file: imagenet_resnet152.ckpt
```

Then you can see files *imagenet_resnet152.ckpt.data-00000-of-00001*, *imagenet_resnet152.ckpt.index* and *imagenet_resnet152.ckpt.meta*, which can be parserd by Tensorflow parser.

3. The network architecture graph is saved in *./graphs*, you can use

```bash
$ tensorboard --logdir graphs/
```

to visualize network graph, to get the output node. In this case, the output node name is [**Squeeze**], which is most the same in our slim model extractor.

![tensorboard](https://nxtb0g.dm2304.livefilestore.com/y4mm6MNZXBSSJ80ar7X2y5ZSzTCxZiC9dNDzv67plb4yQutUb-WBQR8bosYLtyepjxH4QE21pNqg3sIviJXEgaMOW0HVwMwMgwU2KAbW6RokO8nS0ZHy82hAivvX8JgU1yEuA-M4gBYyt8egLilIN10IgGBj-5ZMh0s18Dz4iCwKbuDX16DfqpJ-_rV50JzXEhIJkPfaFczLss0P3ItIEwWlw?width=1200&height=449&cropmode=none)

### Convert architecture from Tensorflow to IR

You can convert only network structure to IR for visualization or training in other frameworks.

> Note: it is much better to specify the **output node name** for Tensorflow models

```bash
$ python -m mmdnn.conversion._script.convertToIR -f tensorflow -d resnet152 -n imagenet_resnet152.ckpt.meta --dstNodeName Squeeze

Parse file [imagenet_resnet152.ckpt.meta] with binary format successfully.
Tensorflow model file [imagenet_resnet152.ckpt.meta] loaded successfully.
IR network structure is saved as [resnet152.json].
IR network structure is saved as [resnet152.pb].
Warning: weights are not loaded.
```

### Convert model (including architecture and weights) from Tensorflow to IR

You can use following bash command to convert the checkpoint files to IR architecture file [*resnet152.pb*], [*resnet152.json*] and IR weights file [*resnet152.npy*]

```bash
$ python -m mmdnn.conversion._script.convertToIR -f tensorflow -d resnet152 -n imagenet_resnet152.ckpt.meta -w imagenet_resnet152.ckpt  --dstNodeName Squeeze

Parse file [imagenet_resnet152.ckpt.meta] with binary format successfully.
Tensorflow model file [imagenet_resnet152.ckpt.meta] loaded successfully.
Tensorflow checkpoint file [imagenet_resnet152.ckpt] loaded successfully. [816] variables loaded.
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

- Tensorflow gpu 1.4.0

@ 11/22/2017

## Limitation

- Currently no RNN related operations support
