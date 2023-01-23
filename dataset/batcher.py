import os
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import random

random.seed(4)

SIZES = {
    'LSMS-ethiopia-2018': {'train': 4559, 'val': 1302, 'test': 652, 'all': 6513}
}

LSMS_TFRECORDS_DIR = '../data/lsms_tfrecords/'
BUCKET = 'ppt-central-bucket'
FOLDER = 'lsms_tfrecords'

CONSUMPTION_MEAN = 4.64419991542738
CONSUMPTION_STD = 4.717155116197405

BAND_MEANS = {'BLUE': 0.05720699718743952, 'GREEN': 0.09490949383988444, 'RED': 0.11647556706520566, 'NIR': 0.25043694995276194, 'SW_IR1': 0.2392968657712096, 'SW_IR2': 0.17881930908670116, 'TEMP': 309.4823962960872, 'avg_rad': 1.8277193893627437}
BAND_STDS = {'BLUE': 0.02379879403788589, 'GREEN': 0.03264212296594092, 'RED': 0.050468921297598834, 'NIR': 0.04951648377311826, 'SW_IR1': 0.07332469136800321, 'SW_IR2': 0.07090649886221509, 'TEMP': 6.000001012494749, 'avg_rad': 4.328436715534132}

def get_tfrecord_paths(split, bucket=True):
    split_sizes = SIZES['LSMS-ethiopia-2018']
    if bucket:
        glob = 'gs://' + BUCKET + '/' + FOLDER + '/' + '*'
        tfrecords = tf.io.gfile.glob(glob)
    else:
        tfrecords = sorted([f for f in os.listdir(LSMS_TFRECORDS_DIR) if not f.startswith('.')])
        random.shuffle(tfrecords)
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


class Batcher:
    def __init__(self, image_shape=(224, 224, 3), bucket=True, num_records=None, buffer_size=5000, batch_size=512, shuffle=True, split='all'):
        self.image_shape = image_shape
        self.tfrecords_paths = get_tfrecord_paths(split=split, bucket=bucket)
        self.num_records = num_records
        self.bands = ['BLUE', 'GREEN', 'RED']
        self.scalar_keys = ['lat', 'lon', 'consumption']
        self.label = ['consumption']
        self.features = self.bands + self.scalar_keys + self.label
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.shuffle = shuffle

    def create_cifar_dataset(self):
        # load cifar10 data
        (train_feature, train_label), (
            test_feature,
            test_label,
        ) = tf.keras.datasets.cifar10.load_data()

        # data preprocessing
        # reshape
        train_feature_vector = tf.image.resize(train_feature, [self.image_shape[0], self.image_shape[1]], method="nearest")
        test_feature_vector = tf.image.resize(test_feature, [self.image_shape[0], self.image_shape[1]], method="nearest")

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

        return (train_feature_normal, train_label_onehot), (test_feature_normal, test_label_onehot)

    def create_cifar_dataset_test(self):
        train_ds, test_ds = tfds.load('cifar10', split=['train', 'test'], as_supervised=True)

        def normalize_resize(image, label):
            image = tf.cast(image, tf.float32)
            image = tf.divide(image, 255)
            image = tf.image.resize(image, (self.image_shape[0], self.image_shape[1]))
            label = tf.one_hot(label, 10)
            return image, label

        train = train_ds.map(normalize_resize, num_parallel_calls=tf.data.AUTOTUNE)
        test = test_ds.map(normalize_resize, num_parallel_calls=tf.data.AUTOTUNE)

        train = train.cache().repeat().shuffle(buffer_size=self.buffer_size).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        test = test.cache().repeat().batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train, test

    def parse_tfrecord(self, example_proto):
        columns = [tf.io.FixedLenFeature(shape=[65025], dtype=tf.float32) for k in self.bands] + [
            tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for k in self.scalar_keys
        ]

        features_dict = dict(zip(self.features, columns))

        return tf.io.parse_single_example(example_proto, features_dict)

    def to_tuple(self, example):
        inputs_list = []
        for key in self.bands:
            band = example.get(key)
            band = tf.reshape(band, [255, 255])[15:239, 15:239]
            # Standardize band
            band = (band - BAND_MEANS[key]) / BAND_STDS[key]
            inputs_list.append(band)

        stacked = tf.stack(inputs_list, axis=0)
        # Convert from CHW to HWC
        stacked = tf.transpose(stacked, [1, 2, 0])
        # Standardize consumption
        label = (example.get('consumption') - CONSUMPTION_MEAN) / CONSUMPTION_STD

        sample = (stacked, label)

        return sample

    def get_dataset(self):
        dataset = tf.data.TFRecordDataset(filenames=self.tfrecords_paths)
        dataset = dataset.map(self.parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.to_tuple, num_parallel_calls=tf.data.AUTOTUNE)

        if self.shuffle:
            dataset = dataset.cache().repeat().shuffle(buffer_size=self.buffer_size).batch(self.batch_size)
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        else:
            dataset = dataset.cache().repeat().batch(self.batch_size)
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset
