clear

python -m mmdnn.conversion._script.IRToModel -f coreml -in caffe_vgg19.pb -iw caffe_vgg19.npy -o vgg19.mlmodel &&

python -m mmdnn.conversion.examples.coreml.imagenet_test -p vgg19 -s caffe -n vgg19.mlmodel &&

:
