# ![MMdnn](https://ndqzpq.dm2304.livefilestore.com/y4mF9ON1vKrSy0ew9dM3Fw6KAvLzQza2nL9JiMSIfgfKLbqJPvuxwOC2VIur_Ycz4TvVpkibMkvKXrX-N9QOkyh0AaUW4qhWDak8cyM0UoLLxc57apyhfDaxflLlZrGqiJgzn1ztsxiaZMzglaIMhoo8kjPuZ5-vY7yoWXqJuhC1BDHOwgNPwIgzpxV1H4k1oQzmewThpAJ_w_fUHzianZtMw?width=35&height=35&cropmode=none) MMdnn

[![PyPi Version](https://img.shields.io/pypi/v/mmdnn.svg)](https://pypi.org/project/mmdnn/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Linux](https://travis-ci.org/Microsoft/MMdnn.svg?branch=master)](https://travis-ci.org/Microsoft/MMdnn)

MMdnn is a comprehensive and cross-framework tool to convert, visualize and diagnose deep learning (DL) models.
The "MM" stands for model management, and "dnn" is the acronym of deep neural network.

Major features include:

- <a href="#conversion">**Model Conversion**</a>

  - We implement a universal converter to convert DL models between frameworks, which means you can train a model with one framework and deploy it with another.

- **Model Retraining**

  - During the model conversion, we generate some code snippets to simplify later retraining or inference.

- **Model Search & Visualization**

  - We provide a [model collection](mmdnn/models/README.md) to help you find some popular models.
  - We provide a <a href="#visualization">model visualizer</a> to display the network architecture more intuitively.

- **Model Deployment**

  - We provide some guidelines to help you deploy DL models to another hardware platform.
    - [Android](https://github.com/Microsoft/MMdnn/wiki/Deploy-your-TensorFlow-Lite-Model-in-Android)
    - [Serving](https://github.com/Microsoft/MMdnn/wiki/Tensorflow-Serving-Via-Docker)
    
  - We provide a guide to help you accelerate inference with TensorRT.
    - [TensorRT](https://github.com/Microsoft/MMdnn/wiki/Using-TensorRT-to-Accelerate-Inference)
  

## Related Projects

Targeting at openness and advancing state-of-art technology, [Microsoft Research (MSR)](https://www.microsoft.com/en-us/research/group/systems-and-networking-research-group-asia/) and [Microsoft Software Technology Center (STC)](https://www.microsoft.com/en-us/ard/company/introduction.aspx) had also released few other open source projects:

* [OpenPAI](https://github.com/Microsoft/pai) : an open source platform that provides complete AI model training and resource management capabilities, it is easy to extend and supports on-premise, cloud and hybrid environments in various scale.
* [FrameworkController](https://github.com/Microsoft/frameworkcontroller) : an open source general-purpose Kubernetes Pod Controller that orchestrate all kinds of applications on Kubernetes by a single controller.
* [NNI](https://github.com/Microsoft/nni) : a lightweight but powerful toolkit to help users automate Feature Engineering, Neural Architecture Search, Hyperparameter Tuning and Model Compression.
* [NeuronBlocks](https://github.com/Microsoft/NeuronBlocks) : an NLP deep learning modeling toolkit that helps engineers to build DNN models like playing Lego. The main goal of this toolkit is to minimize developing cost for NLP deep neural network model building, including both training and inference stages.
* [SPTAG](https://github.com/Microsoft/SPTAG) : Space Partition Tree And Graph (SPTAG) is an open source library for large scale vector approximate nearest neighbor search scenario.

We encourage researchers, developers and students to leverage these projects to boost their AI / Deep Learning productivity.

## Installation

### Install manually

You can get a stable version of MMdnn by

```bash
pip install mmdnn
```
And make sure to have [Python](https://www.python.org/) installed
or you can try the newest version by

```bash
pip install -U git+https://github.com/Microsoft/MMdnn.git@master
```

### Install with docker image

MMdnn provides a docker image, which packages MMdnn and Deep Learning frameworks that we support as well as other dependencies.
You can easily try the image with the following steps:

1. Install Docker Community Edition(CE)

    [_Learn more about how to install docker_](https://github.com/Microsoft/MMdnn/blob/master/docs/InstallDockerCE.md)

1. Pull MMdnn docker image
    ```bash
    docker pull mmdnn/mmdnn:cpu.small
    ```

1. Run image in an interactive mode

    ```bash
    docker run -it mmdnn/mmdnn:cpu.small
    ```

## Features

### <a name="conversion">Model Conversion</a>

Across the industry and academia, there are a number of existing frameworks available for developers and researchers to design a model, where each framework has its own network structure definition and saving model format. The gaps between frameworks impede the inter-operation of the models.

<img src="https://raw.githubusercontent.com/Microsoft/MMdnn/master/docs/supported.jpg" width="633" >

We provide a model converter to help developers convert models between frameworks through an intermediate representation format.

#### Support frameworks

> [Note] You can click the links to get detailed README of each framework.

- [Caffe](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/caffe/README.md)
- [Microsoft Cognitive Toolkit (CNTK)](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/cntk/README.md)
- [CoreML](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/coreml/README.md)
- [Keras](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/keras/README.md)
- [MXNet](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/mxnet/README.md)
- [ONNX](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/onnx/README.md) (Destination only)
- [PyTorch](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/pytorch/README.md)
- [TensorFlow](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/tensorflow/README.md) (Experimental) (We highly recommend you read the README of TensorFlow first)
- [DarkNet](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/darknet/README.md) (Source only, Experiment)

#### Tested models

The model conversion between currently supported frameworks is tested on some **ImageNet** models.

Models | Caffe | Keras | TensorFlow | CNTK | MXNet | PyTorch  | CoreML | ONNX
:-----:|:-----:|:-----:|:----------:|:----:|:-----:|:--------:|:------:|:-----:|
[VGG 19](https://arxiv.org/abs/1409.1556.pdf) | √ | √ | √ | √ | √ | √ | √ | √
[Inception V1](https://arxiv.org/abs/1409.4842v1) | √ | √ | √ | √ | √ | √ | √ | √
[Inception V3](https://arxiv.org/abs/1512.00567)  | √ | √ | √ | √ | √ | √ | √ | √
[Inception V4](https://arxiv.org/abs/1512.00567)  | √ | √ | √ | o | √ | √ | √ | √
[ResNet V1](https://arxiv.org/abs/1512.03385)                               |   ×   |   √   |     √      |   o  |   √   |    √ | √ | √
[ResNet V2](https://arxiv.org/abs/1603.05027)                               |   √   |   √   |     √      |   √  |   √   | √ | √ | √
[MobileNet V1](https://arxiv.org/pdf/1704.04861.pdf)                        |   ×   |   √   |     √      |   o  |   √   |    √       | √ | √ | √
[MobileNet V2](https://arxiv.org/pdf/1704.04861.pdf)                        |   ×   |   √   |     √      |   o  |   √   |    √       | √ | √ | √
[Xception](https://arxiv.org/pdf/1610.02357.pdf)                            |   √   |   √   |     √      |   o  |   ×   |    √ | √ | √ | √
[SqueezeNet](https://arxiv.org/pdf/1602.07360)                              |   √   |   √   |     √      |   √  |   √   |    √ | √ | √ | √
[DenseNet](https://arxiv.org/abs/1608.06993)                                |   √   |   √   |     √      |   √  |   √   |    √       | √ | √
[NASNet](https://arxiv.org/abs/1707.07012)                                  |   x   |   √   |     √      |   o  |   √   | √ | √ | x
[ResNext](https://arxiv.org/abs/1611.05431)                                 |   √   |   √   |     √      |   √  |   √   | √ | √ | √ | √ | √
[voc FCN](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) |       |       |     √      |   √  |       |
Yolo3                                                                       |       |   √   |            |   √  |

#### Usage

One command to achieve the conversion. Using TensorFlow **ResNet V2 152** to PyTorch as our example.

```bash
$ mmdownload -f tensorflow -n resnet_v2_152 -o ./
$ mmconvert -sf tensorflow -in imagenet_resnet_v2_152.ckpt.meta -iw imagenet_resnet_v2_152.ckpt --dstNodeName MMdnn_Output -df pytorch -om tf_resnet_to_pth.pth
```

Done.

#### On-going frameworks

- Torch7 (help wanted)
- Chainer (help wanted)

#### On-going Models

- Face Detection
- Semantic Segmentation
- Image Style Transfer
- Object Detection
- RNN

---

### <a name="visualization">Model Visualization</a>

We provide a [local visualizer](mmdnn/visualization) to display the network architecture of a deep learning model.
Please refer to the [instruction](mmdnn/visualization/README.md).

---

## Examples

### Official Tutorial

- [Keras "inception V3" to CNTK](https://github.com/Microsoft/MMdnn/blob/master/docs/keras2cntk.md) and [related issue](https://github.com/Microsoft/MMdnn/issues/19)

- [TensorFlow slim model "ResNet V2 152" to PyTorch](https://github.com/Microsoft/MMdnn/blob/master/docs/tf2pytorch.md)

- [Mxnet model "LResNet50E-IR" to TensorFlow](https://github.com/Microsoft/MMdnn/issues/85) and [related issue](https://github.com/Microsoft/MMdnn/issues/135)

### Users' Examples

- [MXNet "ResNet-152-11k" to PyTorch](https://github.com/Microsoft/MMdnn/issues/6)

- [Another Example of MXNet "ResNet-152-11k" to PyTorch](https://blog.paperspace.com/convert-full-imagenet-pre-trained-model-from-mxnet-to-pytorch/)

- [MXNet "ResNeXt" to Keras](https://github.com/Microsoft/MMdnn/issues/58)

- [TensorFlow "ResNet-101" to PyTorch](https://github.com/Microsoft/MMdnn/issues/22)

- [TensorFlow "mnist mlp model" to CNTK](https://github.com/Microsoft/MMdnn/issues/11)

- [TensorFlow "Inception_v3" to MXNet](https://github.com/Microsoft/MMdnn/issues/30)

- [Caffe "voc-fcn" to TensorFlow](https://github.com/Microsoft/MMdnn/issues/29)

- [Caffe "AlexNet" to TensorFlow](https://github.com/Microsoft/MMdnn/issues/10)

- [Caffe "inception_v4" to TensorFlow](https://github.com/Microsoft/MMdnn/issues/26)

- [Caffe "VGG16_SOD" to TensorFlow](https://github.com/Microsoft/MMdnn/issues/27)

- [Caffe "SqueezeNet v1.1" to CNTK](https://github.com/Microsoft/MMdnn/issues/48)

---

## Contributing

Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Intermediate Representation

The intermediate representation stores the **network architecture** in **protobuf binary** and **pre-trained weights** in **NumPy** native format.

> [Note!] Currently the IR weights data is in NHWC (channel last) format.

Details are in [ops.txt](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/common/IR/ops.pbtxt) and [graph.proto](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/common/IR/graph.proto). New operators and any comments are welcome.

### Frameworks

We are working on other frameworks conversion and visualization, such as PyTorch, CoreML and so on. We're investigating more RNN related operators. Any contributions and suggestions are welcome! Details in [Contribution Guideline](https://github.com/Microsoft/MMdnn/wiki/Contribution-Guideline).

## Authors

Yu Liu (Peking University): Project Developer & Maintainer

Cheng CHEN (Microsoft Research Asia): Caffe, CNTK, CoreML Emitter, Keras, MXNet, TensorFlow

Jiahao YAO (Peking University): CoreML, MXNet Emitter, PyTorch Parser; HomePage

Ru ZHANG (Chinese Academy of Sciences): CoreML Emitter, DarkNet Parser, Keras, TensorFlow frozen graph Parser; Yolo and SSD models; Tests

Yuhao ZHOU (Shanghai Jiao Tong University): MXNet

Tingting QIN (Microsoft Research Asia): Caffe Emitter

Tong ZHAN (Microsoft): ONNX Emitter

Qianwen WANG (Hong Kong University of Science and Technology): Visualization

## Acknowledgements

Thanks to [Saumitro Dasgupta](https://github.com/ethereon), the initial code of *caffe -> IR converting* is references to his project [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow).

## License
Licensed under the [MIT](LICENSE) license.
