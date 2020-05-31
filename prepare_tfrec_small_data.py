# Prepare TFRecord files from small dataset (loads all data into RAM)
# Data is not subdivided into training/validation/test

import numpy as np
import tensorflow as tf

from load_data import load_data_3D

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Don't need GPUs

## Load data

# Use your own code to load the data
data = load_data_3D(subfolder='ABIDE_192cubes_500')

a = data["controls_volumes"]
b = data["asds_volumes"]

# Number of images in each class
nA = len(a)
nB = len(b)
nExamples = nA + nB

## Prepare data

# Input data
X = np.vstack([a,b])

# Target data
Y = np.concatenate( (np.zeros(nA), np.ones(nB)), axis=None ).astype(int)
# Y = np.arange(nExamples)  # Used for debugging, check order of Dataset operations, etc.

## (optional) Permute data

perm = np.random.permutation(np.arange(nExamples))

X = X[perm]
Y = Y[perm]

# To permute arrays: X = X[perm]
# To permute lists:  X = [X[i] for i in perm]

## TF features

# These are standard. No need to touch them.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

## Determine number of TFRecord files

# Number of data Examples (x,y) in each tfrecord file. Last file may have fewer Examples.
maxExamplesPerFile = 16

nFullFiles, remainder = np.divmod(nExamples, maxExamplesPerFile)

nExamplesPerFile = [maxExamplesPerFile] * nFullFiles
if remainder: nExamplesPerFile.append(int(remainder))

nFiles = len(nExamplesPerFile)

print('{:d} files with {:d} examples each'.format(nFiles, maxExamplesPerFile))

## Write TFRecord files

# Define and create output folder
folderOut = 'test'
if not os.path.exists(folderOut):
    os.makedirs(folderOut)

# Files will be named "<fileOutRoot>_0001.tfrecord", etc.
fileOutRoot = os.path.join(folderOut, 'abide500')

exIndex = 0

# Loop through TFRecord files
for fileIndex, nExamplesFile in enumerate(nExamplesPerFile):

    fileName = '{}_{:04d}.tfrecord'.format(fileOutRoot, fileIndex)

    writer = tf.io.TFRecordWriter(fileName)

    # Loop through all indiviual examples in the dataset
    for image, label in zip(X[exIndex:exIndex+nExamplesFile], Y[exIndex:exIndex+nExamplesFile]):

        # Define Example features
        feature = {
            'image': _bytes_feature(image.tostring()),
            'label': _int64_feature(int(label))
        }

        # Create Example out of features
        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Write example to TFRecord file
        writer.write(tf_example.SerializeToString())

    exIndex += nExamplesFile

    print('Saved {:d} examples, last file {}'.format(exIndex, fileName))
