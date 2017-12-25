clear

# python -m mmdnn.conversion._script.IRToModel -f coreml -in examples/caffe/caffe_vgg19.pb -iw examples/caffe/caffe_vgg19.npy -o vgg19.mlmodel --isBGR --blueBias 103.939 --redBias 123.68 --greenBias 116.779 &&

python -m mmdnn.conversion.examples.coreml.imagenet_test -p vgg19 -s caffe --model vgg19.mlmodel -input data -output prob &&

# python -m mmdnn.conversion.examples.coreml.imagenet_test -p vgg19 -s caffe -n tt.mlmodel &&

# python -m mmdnn.conversion._script.IRToModel -f coreml -in keras_resnet.pb -iw keras_resnet.npy -o keras_resnet.mlmodel --isBGR --blueBias 103.939 --redBias 123.68 --greenBias 116.779 &&
# --classInputPath synset_words.txt

# python -m mmdnn.conversion.examples.coreml.imagenet_test -p resnet -s keras --model keras_resnet.mlmodel -input input_1 -output fc1000_activation &&
:
