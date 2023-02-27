import os
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import random
import dataset.dataset_constants as dc

random.seed(4)

BUCKET = 'ppt-central-bucket'
FOLDER = 'eth_lsms_tfrecords'


def get_tfrecord_paths(country_year, split, bucket=True):
    split_sizes = dc.SIZES[country_year]
    if bucket:
        glob = 'gs://' + BUCKET + '/' + FOLDER + '/' + '*'
        tfrecords = tf.io.gfile.glob(glob)
    else:
        tfrecords = sorted([f for f in os.listdir(LSMS_TFRECORDS_DIR) if not f.startswith('.')])
        random.shuffle(tfrecords)
        for i, file in enumerate(tfrecords):
            tfrecords[i] = os.path.join(LSMS_TFRECORDS_DIR, file)

    tfrecord_paths = []
    if split == 'all':
        tfrecord_paths = tfrecords[:split_sizes['all']]
    else:
        if split == 'train':
            tfrecord_paths = tfrecords[0:split_sizes['train']]
        elif split == 'val':
            tfrecord_paths = tfrecords[split_sizes['train']:split_sizes['train'] + split_sizes['val']]
        elif split == 'test':
            tfrecord_paths = tfrecords[-split_sizes['test']:]
        elif split == 'custom':
            tfrecord_paths = random.sample(tfrecords, split_sizes['custom'])

    return tfrecord_paths


class Batcher:
    def __init__(self, bands, country_year='ethiopia-2018', image_shape=(224, 224, 3), bucket=True, num_records=None, buffer_size=5000, batch_size=512, repeat=None, shuffle=True, split='all'):
        self.country_year = country_year
        self.LSMS_TFRECORDS_DIRS = {
            'ethiopia-2018': '../data/eth_lsms_tfrecords',
            'nigeria-2018': '../data/nga_lsms_tfrecords',
            'malawi-2016': '../data/mwi_lsms_tfrecords',
            'pollution-2018': '../data/pollution_lsms_tfrecords'
        }
        global LSMS_TFRECORDS_DIR
        LSMS_TFRECORDS_DIR = self.LSMS_TFRECORDS_DIRS[country_year]
        self.bands = bands
        self.image_shape = image_shape
        self.tfrecords_paths = get_tfrecord_paths(country_year=country_year, split=split, bucket=bucket)
        self.num_records = num_records
        self.scalar_keys = ['lat', 'lon', 'consumption']
        self.label = ['consumption']
        self.features = self.bands + self.scalar_keys + self.label
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.split = split

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

            # label = tf.one_hot(label, 10)
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
            # band = tf.divide(band, 255)
            # Standardize band
            band = (band - dc.MEANS_DICT[self.country_year][key]) / dc.STD_DEVS_DICT[self.country_year][key]
            inputs_list.append(band)

        stacked = tf.stack(inputs_list, axis=0)
        # Convert from CHW to HWC
        stacked = tf.transpose(stacked, [1, 2, 0])
        # Standardize consumption
        # label = (example.get('consumption') - dc.CONSUMPTION_MEANS[self.country_year]) / dc.CONSUMPTION_STD_DEVS[self.country_year]
        label = example.get('consumption')

        sample = (stacked, label)

        return sample

    def get_dataset(self):
        dataset = tf.data.TFRecordDataset(filenames=self.tfrecords_paths)
        dataset = dataset.map(self.parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.to_tuple, num_parallel_calls=tf.data.AUTOTUNE)

        if self.shuffle:
            dataset = dataset.cache().repeat(self.repeat).shuffle(buffer_size=self.buffer_size).batch(self.batch_size)
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        else:
            dataset = dataset.cache().repeat(self.repeat).batch(self.batch_size)
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset
