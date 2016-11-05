# Function for handling the data

import struct
import numpy as np
import cv2
from PIL import Image
import os.path

import tensorflow as tf

def read_int(f):
    return struct.unpack('i', f.read(4))[0]

def read_ushort(f):
    return struct.unpack('H', f.read(2))[0]

def get_depth_image(binary_file_name):

    with open(binary_file_name, 'rb') as f:
        no_rows = read_int(f)
        no_cols = read_int(f)

        depth_image = np.zeros((no_rows, no_cols))
        for row in range(no_rows):
            for col in range(no_cols):
                depth_image[row,col] = read_ushort(f)

        return depth_image

def get_color_image(image_file_name):
    return cv2.imread(image_file_name)

#
# Here the datapath would be: /Users/colegulino/Desktop/ape/data/
# And the number is the number of the image
#
def get_rgb_image(data_path, number):
    binary_file_name = data_path + "depth" + str(number) + ".dpt"
    color_file_name = data_path + "color" + str(number) + ".jpg"
    color_image = get_color_image(color_file_name)
    depth_image = get_depth_image(binary_file_name)
    depth_image = np.reshape(depth_image, (depth_image.shape[0], depth_image.shape[1], 1))

    return np.concatenate((color_image, depth_image), axis=2)

#
# Returns a 3x4 transform matrix
#
def get_transform(data_path, number):
    rot_file = data_path + "rot" + str(number) + ".rot"
    tra_file = data_path + "tra" + str(number) + ".tra"

    transform = np.zeros((3,4))

    with open(rot_file) as f:
        content = f.readlines()
        array_size = content[0][:-1].split(' ', 2)

        rot = np.zeros((int(array_size[0]), int(array_size[1])))
        for i in range(1, len(content)):
            row = content[i][:-1].split(' ', 2)
            for col in range(len(row)):
                rot[i-1,col] = float(row[col])

        transform[:3, :3] = rot

    with open(tra_file) as f:
        content = f.readlines()
        array_size = content[0][:-1].split(' ', 2)

        tra = np.zeros((int(array_size[1]), int(array_size[0])))
        for i in range(1, len(content)):
            row = content[i][:-2].split(' ', 1)
            for col in range(len(row)):
                tra[i-1,col] = float(row[col])

        transform[:, 3] = np.reshape(tra, tra.shape[0])

    return transform

#
# Simple example showing how to create a class for the ape
#
class data_set:
    def __init__(self, data_path, data_range):
        self.data_path = data_path
        self.no_examples = data_range

        self.images = self.get_rgb_images()
        self.labels = self.get_transforms()

    def color_image_path(self, number):
        return self.data_path + "color" + str(number) + ".jpg"

    def depth_image_path(self, number):
        return self.data_path + "depth" + str(number) + ".dpt"

    def rotation_image_path(self, number):
        return self.data_path + "rot" + str(number) + ".rot"

    def translation_image_path(self, number):
        return self.data_path + "tra" + str(number) + ".tra"

    def get_rgb_image(self, number):
        return get_rgb_image(self.data_path, number)

    def get_transform(self, number):
        return get_transform(self.data_path, number)

    def get_rgb_images(self):
        images = dict()
        for i in range(self.no_examples[0], self.no_examples[1]):
            images[i] = self.get_rgb_image(i)

        return images

    def get_transforms(self):
        transforms = dict()
        for i in range(self.no_examples[0], self.no_examples[1]):
            transforms[i] = self.get_transform(i)

        return transforms

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#
# Function to convert a dataset into a TFRecord
#
def convert_to_TFRecord(data_set, name, out_data_path):
    path = data_set.data_path

    images = data_set.images
    translations = data_set.labels

    num_examples = len(translations)

    (rows, cols, channels) = images[0].shape

    filename = os.path.join(out_data_path, name + '.tfrecords')

    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for i in images:
        image_raw = images[i].tostring
        raw_translations = translations[i].tostring
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(channels),
            'label': _bytes_feature(raw_translations),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

def read_and_decode(filename_queue, datashape):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([datashape])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.decode_raw(features['label'], tf.float32)

    return image, label

def inputs(file_name, batch_size, num_epochs, data_shape):
    if not num_epochs: num_epochs = None

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        image, label = read_and_decode(filename_queue, data_shape)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)

        return images, sparse_labels

if __name__ == '__main__':

    # How to write a dataset to a TFRecord
    data_path = "/Users/colegulino/Desktop/ape/data/"
    range_of_data = (0, 1235)
    ape = data_set(data_path, range_of_data)
    out_path = "/Users/colegulino/Desktop/"
    convert_to_TFRecord(ape, "ape", out_path)
