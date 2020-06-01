
# import keras.backend as K

import datetime
import os
import time

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam as Adam
from tensorflow.keras.losses import binary_crossentropy as BC

from load_data import load_data_3D

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# TF 1 code for memory growth
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# K.tensorflow_backend.set_session(tf.Session(config=config))



def plot_results(history):
    loss = history.history['loss']
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']

    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training','Validation'])

    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training','Validation'])

    plt.show()

def build_CNN(input_shape, n_conv_layers=3, n_filters=16, n_dense_layers=0, n_nodes=50, learning_rate=0.01):

    # Setup a sequential model
    model = Sequential()

    # Add first convolutional layer to the model, requires input shape
    model.add(Conv3D(n_filters, kernel_size=(3, 3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2,2,2)))

    # Add remaining convolutional layers to the model, the number of filters should increase a factor 2 for each layer
    temp = n_filters * 2
    for i in range(n_conv_layers-1):
        model.add(Conv3D(temp, kernel_size=(3, 3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(2,2,2)))
        temp = temp * 2

    model.add(Flatten())

    # Add intermediate dense layers
    for i in range(n_dense_layers):
        model.add(Dense(n_nodes,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

    model.add(Dense(1,activation='sigmoid'))

    # Compile model\n",
    model.compile(loss=BC,optimizer=Adam(lr=learning_rate),metrics=['accuracy'])

    return model

## Load data

data = load_data_3D(subfolder='ABIDE_192cubes_500')

number_of_channels = data["nr_of_channels"]
volume_size = data["volume_size"] + (number_of_channels,)
print('volume size: ', volume_size)

controls_volumes = data["controls_volumes"]
asds_volumes = data["asds_volumes"]

# Create array containing all volumes
X = np.vstack((controls_volumes, asds_volumes))

# Create labels
Y = np.concatenate( (np.zeros(len(controls_volumes)), np.ones(len(asds_volumes))), axis=None )

del controls_volumes
del asds_volumes

print('Data:  ', (X.shape,Y.shape))

Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=0.30)
del X, Y

Xtest, Xval, Ytest, Yval = train_test_split(Xval, Yval, test_size=0.50)

print('Train: ', (Xtrain.shape,Ytrain.shape))
print('Valid: ', (Xval.shape,Yval.shape))
print('Test:  ', (Xtest.shape,Ytest.shape))

## Build model

input_shape = volume_size
model = build_CNN(input_shape, 5, 8, 2, 100)

# model.summary()

## Train

batch_size = 8
n_epochs = 20

start_time = time.time()

history = model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=n_epochs, validation_data=(Xval,Yval))

elapsed_time = time.time() - start_time
elapsed_time_string = str(datetime.timedelta(seconds=round(elapsed_time)))
print('Training time: ', elapsed_time_string)

score = model.evaluate(Xtest, Ytest, batch_size=batch_size)
print('Test loss: %.4f' % score[0])
print('Test accuracy: %.4f' % score[1])

plot_results(history)



