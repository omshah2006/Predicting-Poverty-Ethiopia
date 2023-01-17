import os
import tensorflow as tf
import numpy as np
import random

from tensorflow.python.data import AUTOTUNE

SIZES = {
    'LSMS-ethiopia-2018': {'train': 4559, 'val': 1302, 'test': 652, 'all': 6513}
}

LSMS_TFRECORDS_DIR = 'data/lsms_tfrecords/'
BUCKET = 'ppt-central-bucket'
FOLDER = 'lsms_tfrecords'

CONSUMPTION_MEAN = 4.64419991542738
CONSUMPTION_STD = 4.717155116197405


def get_tfrecord_paths(split, bucket=True):
    split_sizes = SIZES['LSMS-ethiopia-2018']
    if bucket:
        glob = 'gs://' + BUCKET + '/' + FOLDER + '/' + '*'
        tfrecords = tf.io.gfile.glob(glob)
    else:
        tfrecords = sorted([f for f in os.listdir(LSMS_TFRECORDS_DIR) if not f.startswith('.')])
        for i, file in enumerate(tfrecords):
            tfrecords[i] = os.path.join('data/lsms_tfrecords', file)

    tfrecord_paths = []
    if split == 'all':
        tfrecord_paths = tfrecords
    else:
        if split == 'train':
            tfrecord_paths = tfrecords[0:split_sizes['train']]
        elif split == 'val':
            tfrecord_paths = tfrecords[split_sizes['train']:split_sizes['train'] + split_sizes['val']]
        elif split == 'test':
            tfrecord_paths = tfrecords[-split_sizes['test']:]

    return tfrecord_paths


class Batcher():
    def __init__(self, shuffle, num_records=None, batch_size=512, split='all'):
        self.tfrecords_paths = get_tfrecord_paths(split=split)
        self.num_records = num_records
        self.bands = ['BLUE', 'GREEN', 'RED']
        self.scalar_keys = ['lat', 'lon', 'consumption']
        self.label = ['consumption']
        self.features = self.bands + self.scalar_keys + self.label
        self.buffer_size = 2048
        self.batch_size = batch_size
        self.shuffle = shuffle

    def create_dataset(self):
        # load cifar10 data
        (train_feature, train_label), (
            test_feature,
            test_label,
        ) = tf.keras.datasets.cifar10.load_data()

        # data preprocessing
        # reshape
        train_feature_vector = tf.image.resize(train_feature, [224, 224], method="nearest")
        test_feature_vector = tf.image.resize(test_feature, [224, 224], method="nearest")

        # feature normalization
        # z-score
        mean = np.mean(train_feature_vector, axis=(0, 1, 2, 3))
        std = np.std(train_feature_vector, axis=(0, 1, 2, 3))
        train_feature_normal = train_feature_vector
        # train_feature_normal = (train_feature_vector - mean) / (std + 1e-7)
        test_feature_normal = test_feature_vector
        # test_feature_normal = (test_feature_vector - mean) / (std + 1e-7)

        # one-hot encoding
        train_label_onehot = tf.keras.utils.to_categorical(train_label)
        test_label_onehot = tf.keras.utils.to_categorical(test_label)

        return train_feature_normal, train_label_onehot, test_feature_normal, test_label_onehot

    def parse_tfrecord(self, example_proto):
        columns = [tf.io.FixedLenFeature(shape=[65025], dtype=tf.float32) for k in self.bands] + [
            tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for k in self.scalar_keys
        ]

        features_dict = dict(zip(self.features, columns))
        # print(features_dict)
        return tf.io.parse_single_example(example_proto, features_dict)

    def to_tuple(self, example):
        inputs_list = []
        for key in self.bands:
            band = example.get(key)
            band = tf.reshape(band, [255, 255])[15:239, 15:239]
            inputs_list.append(band)

        stacked = tf.stack(inputs_list, axis=0)
        # Convert from CHW to HWC
        stacked = tf.transpose(stacked, [1, 2, 0])

        label = (example.get('consumption') - CONSUMPTION_MEAN) / CONSUMPTION_STD

        sample = (stacked, label)
        # print(stacked[:, :, :len(self.bands)], example.get('consumption'))
        # stacked[:, :, len(self.bands):]
        return sample

    def get_dataset(self):
        # print(get_tfrecord_paths(tfrecord_dir='data/lsms_tfrecords/'))
        dataset = tf.data.TFRecordDataset(filenames=self.tfrecords_paths)
        dataset = dataset.map(self.parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.to_tuple, num_parallel_calls=tf.data.AUTOTUNE)

        if self.shuffle:
            dataset = dataset.cache().repeat().shuffle(buffer_size=50000).batch(self.batch_size)
                #dataset.shuffle(self.buffer_size).batch(self.batch_size).repeat().cache()
            dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        else:
            dataset = dataset.cache().repeat().batch(self.batch_size)
                # dataset.batch(self.batch_size).repeat().cache()
            dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        return dataset
