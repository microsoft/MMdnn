source ~/.bashrc
clear

# cntk
#   resnet18
python3 -m mmdnn.conversion._script.convertToIR -f cntk -n examples/cntk/models/ResNet18_ImageNet_CNTK.model -o kit_imagenet --inputShape 3 224 224 &&

# # mxnet
# #   vgg19
# python3 -m mmdnn.conversion._script.convertToIR -f mxnet -n examples/mxnet/models/vgg19-symbol.json -w examples/mxnet/models/vgg19-0000.params -d kit_imagenet --inputShape 3 224 224 &&

# python3 -m mmdnn.conversion._script.IRToCode -f cntk --IRModelPath kit_imagenet.pb --dstModelPath kit_imagenet.py --IRWeightPath kit_imagenet.npy &&

# python3 -m mmdnn.conversion.examples.cntk.imagenet_test -p vgg19 -s mxnet -w kit_imagenet.npy &&

# #   resnet50
# python3 -m mmdnn.conversion._script.convertToIR -f mxnet -n examples/mxnet/models/resnet-50-symbol.json -w examples/mxnet/models/resnet-50-0000.params -d kit_imagenet --inputShape 3 224 224 &&

# python3 -m mmdnn.conversion._script.IRToCode -f cntk --IRModelPath kit_imagenet.pb --dstModelPath kit_imagenet.py --IRWeightPath kit_imagenet.npy &&

# python3 -m mmdnn.conversion.examples.cntk.imagenet_test -p resnet -s mxnet -w kit_imagenet.npy &&

# #   Inception BN
# python3 -m mmdnn.conversion._script.convertToIR -f mxnet -n examples/mxnet/models/Inception-BN-symbol.json -w examples/mxnet/models/Inception-BN-0126.params -d kit_imagenet --inputShape 3 224 224 &&

# python3 -m mmdnn.conversion._script.IRToCode -f cntk --IRModelPath kit_imagenet.pb --dstModelPath kit_imagenet.py --IRWeightPath kit_imagenet.npy &&

# python3 -m mmdnn.conversion.examples.cntk.imagenet_test -p inception_bn -s mxnet -w kit_imagenet.npy &&

# #   SqueezeNet
# python3 -m mmdnn.conversion._script.convertToIR -f mxnet -n examples/mxnet/models/squeezenet_v1.1-symbol.json -w examples/mxnet/models/squeezenet_v1.1-0000.params -d kit_imagenet --inputShape 3 224 224 &&

# python3 -m mmdnn.conversion._script.IRToCode -f cntk --IRModelPath kit_imagenet.pb --dstModelPath kit_imagenet.py --IRWeightPath kit_imagenet.npy &&

# python3 -m mmdnn.conversion.examples.cntk.imagenet_test -p squeezenet -s mxnet -w kit_imagenet.npy &&

# #   imagenet11k-resnet152-11k
# python3 -m mmdnn.conversion._script.convertToIR -f mxnet -n examples/mxnet/models/resnet-152-symbol.json -w examples/mxnet/models/resnet-152-0000.params -d kit_imagenet --inputShape 3 224 224 &&

# python3 -m mmdnn.conversion._script.IRToCode -f cntk --IRModelPath kit_imagenet.pb --dstModelPath kit_imagenet.py --IRWeightPath kit_imagenet.npy &&

# python3 -m mmdnn.conversion.examples.cntk.imagenet_test -p resnet152-11k -s mxnet -w kit_imagenet.npy &&



# # caffe
# #   vgg19
# python3 -m mmdnn.conversion._script.convertToIR -f caffe -d kit_imagenet -n examples/caffe/models/VGG_ILSVRC_19_layers_deploy.prototxt -w examples/caffe/models/VGG_ILSVRC_19_layers.caffemodel &&

# python3 -m mmdnn.conversion._script.IRToCode --dstModelFormat cntk --IRModelPath kit_imagenet.pb --dstModelPath kit_imagenet.py --IRWeightPath kit_imagenet.npy &&

# python3 -m mmdnn.conversion.examples.cntk.imagenet_test -p vgg19 -s caffe -w kit_imagenet.npy &&

# #   inception v1
# python3 -m mmdnn.conversion._script.convertToIR -f caffe -d kit_imagenet -n examples/caffe/models/bvlc_googlenet.prototxt -w examples/caffe/models/bvlc_googlenet.caffemodel &&

# python3 -m mmdnn.conversion._script.IRToCode --dstModelFormat cntk --IRModelPath kit_imagenet.pb --dstModelPath kit_imagenet.py --IRWeightPath kit_imagenet.npy &&

# python3 -m mmdnn.conversion.examples.cntk.imagenet_test -p inception_v1 -s caffe -w kit_imagenet.npy &&

# #   resnet 152
# python3 -m mmdnn.conversion._script.convertToIR -f caffe -d kit_imagenet -n examples/caffe/models/ResNet-152-deploy.prototxt -w examples/caffe/models/ResNet-152-model.caffemodel &&

# python3 -m mmdnn.conversion._script.IRToCode --dstModelFormat cntk --IRModelPath kit_imagenet.pb --dstModelPath kit_imagenet.py --IRWeightPath kit_imagenet.npy &&

# python3 -m mmdnn.conversion.examples.cntk.imagenet_test -p resnet152 -s caffe -w kit_imagenet.npy &&

# #   squeezenet
# python3 -m mmdnn.conversion._script.convertToIR -f caffe -d kit_imagenet -n examples/caffe/models/squeezenet_v1.1.prototxt -w examples/caffe/models/squeezenet_v1.1.caffemodel &&

# python3 -m mmdnn.conversion._script.IRToCode -f cntk -in kit_imagenet.pb -iw kit_imagenet.npy -d kit_imagenet.py &&

# python3 -m mmdnn.conversion.examples.cntk.imagenet_test -p squeezenet -s caffe -n kit_imagenet.py -w kit_imagenet.npy &&

# # tensorflow
# #   resnet_v2 152
# python3 -m mmdnn.conversion._script.convertToIR -f tensorflow -d kit_imagenet -n examples/tensorflow/models/imagenet_resnet152.ckpt.meta --dstNodeName Squeeze -w examples/tensorflow/models/imagenet_resnet152.ckpt &&

# python3 -m mmdnn.conversion._script.IRToCode -f cntk --IRModelPath kit_imagenet.pb --dstModelPath kit_imagenet.py --IRWeightPath kit_imagenet.npy &&

# python3 -m mmdnn.conversion.examples.cntk.imagenet_test -p resnet -s tensorflow -w kit_imagenet.npy &&

# #   vgg19
# python3 -m mmdnn.conversion._script.convertToIR -f tensorflow -d kit_imagenet -n examples/tensorflow/models/imagenet_vgg19.ckpt.meta -node vgg_19/fc8/squeezed -w examples/tensorflow/models/imagenet_vgg19.ckpt &&

# python3 -m mmdnn.conversion._script.IRToCode -f cntk --IRModelPath kit_imagenet.pb --dstModelPath kit_imagenet.py --IRWeightPath kit_imagenet.npy &&

# python3 -m mmdnn.conversion.examples.cntk.imagenet_test -p vgg19 -s tensorflow -w kit_imagenet.npy &&

# #   inception_v3
# python3 -m mmdnn.conversion._script.convertToIR -f tensorflow -d kit_imagenet -n examples/tensorflow/models/imagenet_inception_v3.ckpt.meta -node InceptionV3/Logits/SpatialSqueeze -w examples/tensorflow/models/imagenet_inception_v3.ckpt &&

# python3 -m mmdnn.conversion._script.IRToCode -f cntk --IRModelPath kit_imagenet.pb --dstModelPath kit_imagenet.py --IRWeightPath kit_imagenet.npy &&

# python3 -m mmdnn.conversion.examples.cntk.imagenet_test -p inception_v3 -s tensorflow -w kit_imagenet.npy &&

# # #    mobilenet
# # python3 -m mmdnn.conversion._script.convertToIR -f tensorflow -d kit_imagenet -n examples/tensorflow/models/mobilenet_v1_1.0_224.ckpt.meta -node MobilenetV1/Logits/SpatialSqueeze -w examples/tensorflow/models/mobilenet_v1_1.0_224.ckpt &&

# # python3 -m mmdnn.conversion._script.IRToCode -f cntk --IRModelPath kit_imagenet.pb --dstModelPath kit_imagenet.py --IRWeightPath kit_imagenet.npy &&

# # python3 -m mmdnn.conversion.examples.cntk.imagenet_test -p mobilenet -s tensorflow -w kit_imagenet.npy &&


# # keras
# #   vgg_19
# python3 -m mmdnn.conversion._script.convertToIR -f keras -d kit_imagenet -n examples/keras/models/imagenet_vgg19.json -w examples/keras/models/imagenet_vgg19.h5 &&

# python3 -m mmdnn.conversion._script.IRToCode --dstModelFormat cntk --IRModelPath kit_imagenet.pb --dstModelPath kit_imagenet.py --IRWeightPath kit_imagenet.npy &&

# python3 -m mmdnn.conversion.examples.cntk.imagenet_test -p vgg19 -s keras -w kit_imagenet.npy &&

# #   inception_v3
# python3 -m mmdnn.conversion._script.convertToIR -f keras -d ./kit_imagenet -n examples/keras/models/imagenet_inception_v3.json -w examples/keras/models/imagenet_inception_v3.h5 &&

# python3 -m mmdnn.conversion._script.IRToCode --dstModelFormat cntk --IRModelPath kit_imagenet.pb --dstModelPath kit_imagenet.py --IRWeightPath kit_imagenet.npy &&

# python3 -m mmdnn.conversion.examples.cntk.imagenet_test -p inception_v3 -s keras -w kit_imagenet.npy &&

# #   resnet50
# python3 -m mmdnn.conversion._script.convertToIR -f keras -d kit_imagenet -n examples/keras/models/imagenet_resnet.json -w examples/keras/models/imagenet_resnet.h5 &&

# python3 -m mmdnn.conversion._script.IRToCode -f cntk --IRModelPath kit_imagenet.pb --dstModelPath kit_imagenet.py --IRWeightPath kit_imagenet.npy &&

# python3 -m mmdnn.conversion.examples.cntk.imagenet_test -p resnet -s keras -w kit_imagenet.npy &&

: