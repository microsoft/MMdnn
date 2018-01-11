from __future__ import print_function
import keras
from keras import layers
from keras.datasets import mnist
from keras.models import Model, load_model
from keras import backend as K
import os.path
import numpy as np
import json


def main():
    '''
    a cnn with no specified input shape (None, None, 1, )
    use GlobalPooling2D as sudo-flatten layer
    but of course, pooling has worse performance than flatten 
    '''
    batch_size = 128
    num_classes = 10
    # input image dimensions
    img_rows, img_cols = 28, 28
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    in_layer=layers.Input(shape=(28,28,1,))
    x=layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(in_layer)
    x=layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')(x)
    x=layers.MaxPooling2D(pool_size=(2,2))(x)
    x=layers.Flatten( )(x)
    x=layers.Dropout(rate=0.25)(x)
    x=layers.Dense(units=128, activation="relu")(x)
    x=layers.Dropout(rate=0.5)(x)
    out_layer=layers.Dense(units=10, activation="softmax")(x)
    model = Model(inputs=[in_layer], outputs=[out_layer])
    model.compile(optimizer=keras.optimizers.Adadelta(),
                loss=keras.losses.categorical_crossentropy,
                metrics=['accuracy'],
                loss_weights=None,
                sample_weight_mode=None, )

    if (os.path.exists('mnist_model.h5')):
        model = load_model('mnist_model.h5')
    else:
        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=1,
                verbose=1,
                validation_data=(x_test, y_test))
        model.save_weights('mnist_weights.h5') 
        model.save("mnist_model.h5")
    
    # save data
    sample = x_test[0]
    sample = np.expand_dims(sample, axis=0)
    weights = {}
    outputs = {}
    layer0 = model.layers[0].input
    for layer in model.layers:
        model_inter = Model(inputs=[layer0], outputs=[layer.output])
        pred = model_inter.predict(sample)
        outputs[layer.name]=pred[0].tolist()
        weights[layer.name]=[w.tolist() for w in layer.get_weights()]
        # print(pred[0].tolist())
    with open("outputs_mnist.json","w") as json_file:
        json.dump(outputs, json_file)
    json_file.close()
    with open("weights_mnist.json","w") as json_file:
        json.dump(weights, json_file)
    json_file.close()

if __name__=="__main__":
    main()
    print('finish')