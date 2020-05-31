
import os
import time
import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import nibabel as nib

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam as Adam
from tensorflow.keras.losses import binary_crossentropy as BC

from load_data import load_data_3D

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def plot_results(history):
    loss = history.history['loss']
    acc = history.history['accuracy']

    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.legend(['Training','Validation'])

    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(acc)
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

    # Compile model
    model.compile(loss=BC,optimizer=Adam(lr=learning_rate),metrics=['accuracy'])

    return model

## Dataset functions

def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        # 'file_name': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    # Load one example
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    # Turn your saved image string into an array
    image = tf.io.decode_raw(parsed_features['image'], tf.float32)
    image = tf.reshape(image, [192, 192, 192, 1])
    # file_name = parsed_features['file_name']
    label = tf.reshape(parsed_features['label'], [1])
    label = parsed_features['label']

    return image, label

def create_dataset(filepath, batch_size, buffer_size):

    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)

    # This dataset will go on forever
    dataset = dataset.repeat()

    # Set the number of datapoints you want to load and shuffle
    dataset = dataset.shuffle(buffer_size=buffer_size)

    # Set the batchsize
    dataset = dataset.batch(batch_size=batch_size)

    # Prefetch batches for efficiency
    dataset = dataset.prefetch(2)

    return dataset

## Load dataset

data_folder = '/flush/davab27/data/ABIDE_192cubes_500_tf'
fileRecordRoot = os.path.join(data_folder, 'abide500')

nFiles = 32
tfrecord_files = ['{}_{:04d}.tfrecord'.format(fileRecordRoot, i) for i in range(nFiles)]

dataset = create_dataset(tfrecord_files, 64, 8)

## (optional) Check some dataset elements

# iMax = 2
# i = 0
#
# for element in dataset:
#     if i == iMax:
#         break
#     print(element)
#     i += 1

## Build model

input_shape = [192,192,192,1]
model = build_CNN(input_shape, 5, 8, 2, 100)

## Train

n_im = 500

batch_size = 8
n_epochs = 10

start_time = time.time()

history = model.fit(dataset, epochs=10, steps_per_epoch=n_im//batch_size)

elapsed_time = time.time() - start_time
elapsed_time_string = str(datetime.timedelta(seconds=round(elapsed_time)))
print('Training time: ', elapsed_time_string)

plot_results(history)



