clear

python -m mmdnn.conversion._script.IRToModel -f coreml -in examples/caffe/caffe_vgg19.pb -iw examples/caffe/caffe_vgg19.npy -o vgg19.mlmodel --classInputPath synset_words.txt &&

python3 -m mmdnn.conversion.examples.coreml.imagenet_test -p vgg19 -s caffe -n vgg19.mlmodel &&

:
