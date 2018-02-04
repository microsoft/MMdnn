# Caffe README

## Usage

Currently we only implemented the Caffe-IR part. Any contribution to Caffe emitter (IR -> Caffe) is welcome.

### Caffe pre-trained model

We tested [vgg19 model](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md) and [Googlenet model](https://github.com/BVLC/caffe/blob/80f44100e19fd371ff55beb3ec2ad5919fb6ac43/models/bvlc_googlenet/readme.md) from [Caffe2 Model Zoo](https://github.com/caffe2/caffe2/wiki/Model-Zoo).

### Convert model from Caffe to IR

You can use following bash command to convert the network architecture [*VGG_ILSVRC_19_layers_deploy.prototxt*] with weights [*VGG_ILSVRC_19_layers.caffemodel*] to IR architecture file [*vgg19.pb*], [*vgg19.json*] and IR weights file [*vgg19.npy*]

```bash
$ python -m mmdnn.conversion._script.convertToIR -f caffe -d vgg19 -n VGG_ILSVRC_19_layers_deploy.prototxt -w VGG_ILSVRC_19_layers.caffemodel
.
.
.
IR network structure is saved as [vgg19.json].
IR network structure is saved as [vgg19.pb].
IR weights are saved as [vgg19.npy].
```

### Convert model from IR to Caffe code

You can use following bash command to convert the IR architecture file [*vgg19.pb*] and weigths file [*vgg19.npy*] to Caffe Python code file[*caffe_vgg19.py*] and IR weights file suit for caffe model[*caffe_vgg19.npy*]

```bash
$ python -m mmdnn.conversion._script.IRToCode -f caffe -n vgg19.pb -w vgg19.npy -d caffe_vgg19.py -dw caffe_vgg19.npy
.
.
.
Parse file [vgg19.pb] with binary format successfully.
Target network code snippet is saved as [caffe_vgg19.py].
Target weights are saved as [caffe_vgg19.npy].
```

### Generate Caffe model from code

You can use following bash command to generate caffe architecture file [*vgg19.prototxt*] and weights file [*vgg19.caffemodel*] from python code [*caffe_vgg19.py*] and weights file [*caffe_vgg19.npy*]

```bash
$ python caffe_vgg19.py -w caffe_vgg19.npy -p vgg19.prototxt -m vgg19.caffemodel
```

## Limitation

- Currently no RNN related operations support
- Only VGG19 and GoogleNet models tested.

---

## Acknowledgement

Thanks to [Saumitro Dasgupta](https://github.com/ethereon), the initial code of *caffe-to-tensorflow* references to his project [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow).
