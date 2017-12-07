# ![MMdnn](https://ndqzpq.dm2304.livefilestore.com/y4mF9ON1vKrSy0ew9dM3Fw6KAvLzQza2nL9JiMSIfgfKLbqJPvuxwOC2VIur_Ycz4TvVpkibMkvKXrX-N9QOkyh0AaUW4qhWDak8cyM0UoLLxc57apyhfDaxflLlZrGqiJgzn1ztsxiaZMzglaIMhoo8kjPuZ5-vY7yoWXqJuhC1BDHOwgNPwIgzpxV1H4k1oQzmewThpAJ_w_fUHzianZtMw?width=35&height=35&cropmode=none) MMdnn

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A comprehensive, cross-framework solution to convert, visualize and diagnosis deep neural network models. The "MM" in MMdnn stands for model management and "dnn" is an acronym for deep neural network.

Basically it converts many DNN models that trained by one framework into others. The major features include:

- **Model File Converter** Converting DNN models between many frameworks
- **Model Code Snippet Generator** Generating training or inference code snippet for any frameworks
- **Model Visualization** Visualizing dnn network structure and parameters for any framework
- **Model compatibility testing** (On-going)

## Features

### Model Conversion

Across the industry and academia, there are a number of existing frameworks available for developers and researchers to design a model, where each framework has its own network structure definition and saving model format. The gaps between frameworks impede the inter-operation of the models.

![Supported](https://mxtw2g.dm2304.livefilestore.com/y4m4pZSqv6iifJyuIpPQ22Z1d4IzQqZYUYRqk418Y9_0s564LrHQH4fhRUnLBjBP_VbrIrgzaXqxIJxm6LymIywnqBNyrU41sDB33lq2pEMb8KC5djkAhVQ3EE7eVM3XPs_XLpNoqNbkUbtKbQxEdx-0O5XOuoOqea_BUK4XL6JWJcSWF2FEB-5U-tHjqLpl5OiztJ_8M8n57ZCjnhBb1wSHA?width=303&height=300&cropmode=none)

We provide a model converter to help developers convert models between frameworks, through an intermediate format.

The intermediate representation will store the network structures as a protobuf binary and pre-trained weights as NumPy native format.

> [Note] Currently the IR weights data is in NHWC (channel last) format.

#### Support frameworks

> [Note] You can click the links to get detail README of each framework

- [Caffe](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/caffe/README.md) (Source only)
- [Keras](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/keras/README.md)
- [MXNet](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/mxnet/README.md)
- [Tensorflow](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/tensorflow/README.md) (Experimental)
- [Microsoft Cognitive Toolkit (CNTK)](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/cntk/README.md) (Destination only)
- [PyTorch](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/pytorch/README.md) (Destination only)

#### Tested models

The model conversion between current supported frameworks is tested on some **ImageNet** models.

Models                                              | Caffe | Keras | Tensorflow | CNTK | MXNet | PyTorch |
:--------------------------------------------------:|:-----:|:-----:|:----------:|:----:|:-----:|:-------:|
[Inception V1](http://arxiv.org/abs/1409.4842v1)    |   √   |   √   |     √      |   √  |   √   | x (No LRN)
[Inception V3](http://arxiv.org/abs/1512.00567)     |   ×   |   √   |     √      |   √  |   √   |    √
[ResNet V1 50](https://arxiv.org/abs/1512.03385)    |   ×   |   √   |     √      |   o  |   √   |    √
[ResNet V2 152](https://arxiv.org/abs/1603.05027)   |   √   |   √   |     √      |   √  |   √   |    √
[VGG 19](http://arxiv.org/abs/1409.1556.pdf)        |   √   |   √   |     √      |   √  |   √   |    √
[MobileNet_v1](https://arxiv.org/pdf/1704.04861.pdf)|   ×   |   √   |     √      | × (No Relu6) | × | ×
[Xception](https://arxiv.org/pdf/1610.02357.pdf)    |   ×   |   √   |     √      |   ×  |   ×   |    ×
[SqueezeNet](https://arxiv.org/pdf/1602.07360)      |       |   √   |     √      |   √  |   √   |    ×

#### On-going frameworks

- [Caffe2](https://caffe2.ai/)
- [CoreML](https://developer.apple.com/documentation/coreml)

#### Installation

You can get stable version of MMdnn by
```bash
pip install https://github.com/Microsoft/MMdnn/releases/download/0.1.1/mmdnn-0.1.1-py2.py3-none-any.whl
```

or you can try newest version by
```bash
pip install -U git+https://github.com/Microsoft/MMdnn.git@master
```

#### Usage

We will use the conversion from [Keras "inception_v3" model](https://github.com/fchollet/deep-learning-models) to CNTK as an example.

Install [Keras](https://keras.io/#installation) and [Tensorflow](https://www.tensorflow.org/install/) in case

```bash
$ pip install keras
$ pip install tensorflow
```

1. The example will download the pre-trained models at first, then use a simple model extractor for [Keras applications](https://keras.io/applications/#applications), you can refer it to extract your Keras model structure and weights.

```bash
$ python -m mmdnn.conversion.examples.keras.extract_model -n inception_v3

Using TensorFlow backend.
Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5
96075776/96112376 [============================>.] - ETA: 0s
.
.
.
Network structure is saved as [imagenet_inception_v3.json].
Network weights are saved as [imagenet_inception_v3.h5].
```

The structure file *imagenet_inception_v3.json* and weights file *imagenet_inception_v3.h5* are downloaded to current working directory.

2. Convert the pre-trained model files to intermediate representation

```bash
$ python -m mmdnn.conversion._script.convertToIR -f keras -d converted -n imagenet_inception_v3.json -w imagenet_inception_v3.h5

Using TensorFlow backend.
.
.
.
Network file [imagenet_inception_v3.json] is loaded successfully.
IR network structure is saved as [converted.json].
IR network structure is saved as [converted.pb].
IR weights are saved as [converted.npy].
```

The Command will take *imagenet_inception_v3.json* as network structure description file, *imagenet_inception_v3.h5* as pre-trained weights, and you will get the intermediate representation files *converted.json* for visualization, *converted.proto* and *converted.npy* for next steps.


3. Convert the IR files to CNTK models

```bash
$ python -m mmdnn.conversion._script.IRToCode -f cntk -d converted_cntk.py -n converted.pb -w converted.npy

Parse file [converted.pb] with binary format successfully.
Target network code snippet is saved as [converted_cntk.py].
```

And you will get a file name *converted_cntk.py*, which contains the **original CNTK** codes to build the *Inception V3* network.

With the three steps, you have already converted the pre-trained Keras Inception_v3 models to CNTK network file *converted_cntk.py* and weight file *converted.npy*. You can use these two files to fine-tune training or inference.

4. Test the converted model

```bash
$ python -m mmdnn.conversion.examples.cntk.imagenet_test -p inception_v3 -s keras -n converted_cntk -w converted.npy
.
.
.
[(386, 0.94166422), (101, 0.029935161), (385, 0.0025184231), (340, 0.0001713269), (684, 0.00014733501)]
Test model [inception_v3] from [keras] passed.
```

The converted model has been tested.

5. Dump the original CNTK model

```bash
$ python -m mmdnn.conversion.examples.cntk.imagenet_test -n converted_cntk -w converted.npy --dump cntk_inception_v3.dnn
.
.
.
CNTK model file is saved as [cntk_inception_v3.dnn], generated by [converted_cntk.py] and [converted.npy].
```
The file *cntk_inception_v3.dnn* can be loaded by CNTK directly.

### Model Visualization

Some tools are provided to visualize the model network structure and settings. Take [Keras "inception_v3" model] as an example again.

1. Download the pre-trained models

```bash
python -m mmdnn.conversion.examples.keras.extract_model -n inception_v3
```

2. Convert the pre-trained model files into intermediate representation

```bash
python3 -m mmdnn.conversion._script.convertToIR -f keras -d converted -n imagenet_inception_v3.json -w imagenet_inception_v3.h5
```

3. Open the simple model visualizater [*visualization/index.html*] and choose file *converted.json*

![Inception_v3](https://opacdq.dm2304.livefilestore.com/y4mNlERtWTEHaNad3F2mbhwFTwHdSI2qXXG4fR-a46E4b0bjCUuXle49NeOuUO3Lntx9FsGq3tBK3krGtUmJsCcpijsNjggjptKlCYewvu-75k0m3UhsPZflWs7ouGrxOEJqq1RqovWM-xm9hOYGoW0FWK18RuBXyRBwGIbj4F-iy8ASLm4qDbS1UPP_VfiPOgKWMXOk6Bw6EhCpQZJvblpsw?width=1024&height=833&cropmode=none)

# Contributing

We are working on other frameworks conversion and visualization, such as PyTorch and CoreML. And more RNN related operators are investigating. Any contributions and suggestions are welcome!

Thanks to [Saumitro Dasgupta](https://github.com/ethereon), the initial codes of *caffe-to-tensorflow* are references to his project [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow).

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.