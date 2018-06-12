# Models

## ImageNet

| | | | | | | |
|-|-|-|-|-|-|-|
|<b>alexnet</b><br />Framework: caffe<br />Description: <br />Download: [prototxt](https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt) [caffemodel](http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel)<br />Convertor:
<div>
    <select id="framelist">
        <option value="tensorflow">Tensorflow</option>
        <option value="MXNet">MXNet</option>
        <option value="PyTorch">PyTorch</option>
    </select>
    <input type="button" id="convert" value="Convert" onclick="combo('framelist', 'convcmd')"/>
    <div id="convcmd">&nbsp</div>
    <script type="text/javascript">
        function combo(framelist, convcmd) {
            convcmd = document.getElementById(convcmd);
            framelist = document.getElementById(framelist);
            var idx = framelist.selectedIndex;
            var content = framelist.options[idx].innerHTML;
            //convcmd.value = "python -c import " + content;
            convcmd.innerHTML = "python -c import " + content;
        }
    </script>
</div>

|<b>inception_v1</b><br />Framework: caffe<br />Description: <br />Download: [prototxt](https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt) [caffemodel](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel)
|
|<b>vgg16</b><br />Framework: caffe<br />Description: <br />Download: [prototxt](https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/c3ba00e272d9f48594acef1f67e5fd12aff7a806/VGG_ILSVRC_16_layers_deploy.prototxt) [caffemodel](http://data.mxnet.io/models/imagenet/test/caffe/VGG_ILSVRC_16_layers.caffemodel)|
|-|-|-|
|