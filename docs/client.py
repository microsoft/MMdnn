'''
Send JPEG image to tensorflow_model_server loaded with GAN model.

Hint: the code has been compiled together with TensorFlow serving
and not locally. The client is called in the TensorFlow Docker container
'''

from __future__ import print_function

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
import tensorflow as tf

# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


# Command line arguments
tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def main(_):
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    # Send request
    image = tf.gfile.FastGFile(FLAGS.image, 'rb').read()
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'tensorflow-serving'
    request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    request.inputs['image'].CopyFrom(tf.contrib.util.make_tensor_proto(image))
    #request.inputs['input'].CopyFrom()

    result = stub.Predict(request, 10.0)  # 10 secs timeout
    print(result)


if __name__ == '__main__':
    tf.app.run()
