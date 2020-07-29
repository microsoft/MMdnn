# CoreML README


We tested the [Awesome-CoreML-Models](https://github.com/likedan/Awesome-CoreML-Models) and the parser works. Any contribution is welcome.

Models                   | Caffe | CoreML | CNTK | Keras | MXNet | PyTorch | TensorFlow| Onnx
:-----------------------:|:-----:|:------:|:----:|:-----:|:-----:|:-------:|:------:|:------:|
alexnet                  |   √   |   √    |      |   √   |   √   |    √    | √       | √
densenet201              |   √   |   √    |      |   √   |   √   |    √    | √       | √
inception_v3             |   √   |   √    |      |   √   |       |    √    | √       | √
vgg19                    |   √   |   √    |      |   √   |   √   |    √    | √       | √
vgg19_bn                 |   √   |   √    |      |   √   |   √   |    √    | √       | √
resnet152                |   √   |   √    |      |   √   |   √   |    √    | √       | √


**√** - Correctness tested

**o** - Some difference after conversion

**space** - not tested

---

## Convert to CoreML

We use a Keras "mobilenet" model to CoreML as an examples.

### Prepare your pre-trained model

In this example, we can use our [Keras pre-trained model extractor](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/examples/keras/extract_model.py) to prepare mobilenet model.

```bash
$ python -m mmdnn.conversion.examples.keras.extract_model -n mobilenet -i mmdnn/conversion/examples/data/seagull.jpg

Using TensorFlow backend.
Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_1_0_224_tf.h5
17227776/17225924 [==============================] - 12s 1us/step
17235968/17225924 [==============================] - 12s 1us/step
Network structure is saved as [imagenet_mobilenet.json].
Network weights are saved as [imagenet_mobilenet.h5].
[(21, 0.84343129), (23, 0.10283408), (146, 0.039170805), (404, 0.0033809284), (144, 0.0026779801)]
```

The Keras model architecture is saved as *imagenet_mobilenet.json*, weights are saved as *imagenet_mobilenet.h5*, and get the original model inference result for our example photo.

Then use keras -> IR parser to convert the original Keras to IR format.

```bash
$ python -m mmdnn.conversion._script.convertToIR -f keras -d keras_mobilenet -n imagenet_mobilenet.json -w imagenet_mobilenet.h5

Using TensorFlow backend.
Network file [imagenet_mobilenet.json] and [imagenet_mobilenet.h5] is loaded successfully.
IR network structure is saved as [keras_mobilenet.json].
IR network structure is saved as [keras_mobilenet.pb].
IR weights are saved as [keras_mobilenet.npy].
```

Then we got the IR format model.

### Slim Model Extractor

You can refer [Slim Model Extractor](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/examples/coreml/extractor.py) to extract your own coreml model, which is a sample tool to extract both architecture and weights from slim pre-trained models.

Supported models: ['inception_v3', 'mobilenet', 'resnet50', 'tinyyolo', 'vgg16']

Example:

```bash
$ mmdownload -f coreml -n mobilenet

Downloading file [./MobileNet.mlmodel] from [https://docs-assets.developer.apple.com/coreml/models/MobileNet.mlmodel]
progress: 16736.0 KB downloaded, 100%
Coreml model mobilenet is saved in [./]
```


## Convert model (including architecture and weights) from Coreml to IR

You can use following bash command to convert the checkpoint files to IR architecture file [*resnet152.pb*], [*resnet152.json*] and IR weights file [*resnet152.npy*]

```bash
$ mmtoir -f coreml -d mobilenet -n MobileNet.mlmodel --dstNodeName MMdnn_Output

IR network structure is saved as [mobilenet.json].
IR network structure is saved as [mobilenet.pb].
IR weights are saved as [mobilenet.npy].
```

### Convert to CoreML

```bash
$ python -m mmdnn.conversion._script.IRToModel -f coreml -in keras_mobilenet.pb -iw keras_mobilenet.npy -o keras_mobilenet.mlmodel --scale 0.00784313725490196 --redBias -1 --greenBias -1 --blueBias -1

Parse file [keras_mobilenet.pb] with binary format successfully.
.
.
.
input {
  name: "input_1"
  type {
    imageType {
      width: 224
      height: 224
      colorSpace: RGB
    }
  }
}
output {
  name: "reshape_2"
  type {
    multiArrayType {
      shape: 1000
      dataType: DOUBLE
    }
  }
}
```

Then the converted CoreML model is saved as *keras_mobilenet.mlmodel*.

> [Note!] The argument *--scale 0.00784313725490196 --redBias -1 --greenBias -1 --blueBias -1* is Keras mobilenet preprocessing.

### Test converted model (Not necessary)

We implemented an sample code for image inference testing. You can refer the [code](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/examples/coreml/imagenet_test.py) to implement **your own testing code**.

```bash
$ python -m mmdnn.conversion.examples.coreml.imagenet_test -input input_1 -output reshape_2 --image mmdnn/conversion/examples/data/seagull.jpg -size 224 -n keras_mobilenet.mlmodel

Loading model [keras_mobilenet.mlmodel].
Model loading success.
[(21, 0.83917254209518433), (23, 0.10752557963132858), (146, 0.038640134036540985), (404, 0.0034028184600174427), (144, 0.0027129633817821741)]
```

The inference result is slightly different from the original keras model. Currently we consider it is acceptable. Any further investigation is welcome.

## Develop version

macOS High Sierra 10.13.3 (17C205)

@ 2018/01/10

## Limitation

- Currently no RNN-related operations support