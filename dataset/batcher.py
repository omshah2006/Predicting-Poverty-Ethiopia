import os
import tensorflow as tf
import numpy as np
import random

def get_tfrecord_paths(tfrecord_dir):
    tfrecord_paths = sorted([f for f in os.listdir(tfrecord_dir) if not f.startswith('.')])
    return tfrecord_paths


class Batcher():
    def __init__(self, num_records, batch_size, seed):
        self.num_records = num_records
        self.batch_size = batch_size

    def data_augment(self):
        pass

    def process_tfrecords(seld):
        pass

    def parse_tfr_elem(element):
        parse_dict = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string)
        }
        example_message = tf.io.parse_single_example(element, parse_dict)

        img_raw = example_message['image_raw']
        height = example_message['height']
        width = example_message['width']
        depth = example_message['depth']
        label = example_message['label']

        feature = tf.io.parse_tensor(img_raw, out_type=tf.uint8)
        feature = tf.reshape(feature, shape=[height, width, depth])
        return (feature, label)

    def next_batch(self):
        pass
