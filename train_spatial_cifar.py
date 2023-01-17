import tensorflow as tf
from models import vgg16
import numpy as np
import time
import matplotlib.pyplot as plt
from dataset import batcher

cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.TPUStrategy(cluster_resolver)


lr = 0.0001
batch_size = 128
EPOCHS = 50


def model_train(train_features, train_labels, val_features, val_labels):
    # training definition
    batch_num = 64
    epoch_num = 20
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, decay=1e-6)
    # datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    #     rotation_range=15,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     horizontal_flip=True,
    # )
    # datagen.fit(train_feature)

    # train
    with strategy.scope():
        # Build your model here
        vgg_model = vgg16.VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3),
            pooling="max",
            classes=10,
            custom_top_classes=10,
            classifier_activation="softmax",
        )

        model = vgg_model
        print(model.summary())

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'], steps_per_execution=32)

    tf.profiler.experimental.server.start(6000)

    history = model.fit(
        x=train_features,
        y=train_labels,
        steps_per_epoch=int(len(train_feature) / batch_num),
        epochs=epoch_num,
        validation_data=(val_features, val_labels),
        batch_size=batch_num,
        verbose=2,
    )

    return model


# train_batcher = batcher.Batcher(shuffle=True, split='train').get_dataset()
# val_batcher = batcher.Batcher(shuffle=False, split='val').get_dataset()
# test_batcher = batcher.Batcher(shuffle=False, split='test').get_dataset()

batcher = batcher.Batcher(shuffle=True)
train_feature, train_label, test_feature, test_label = batcher.create_dataset()

# data preprocessing
# reshape
# train_feature_vector = tf.image.resize(train_feature, [224, 224], method="nearest")
# test_feature_vector = tf.image.resize(test_feature, [224, 224], method="nearest")
#
# # feature normalization
# # z-score
# mean = np.mean(train_feature_vector, axis=(0, 1, 2, 3))
# std = np.std(train_feature_vector, axis=(0, 1, 2, 3))
# train_feature_normal = train_feature_vector
# # train_feature_normal = (train_feature_vector - mean) / (std + 1e-7)
# test_feature_normal = test_feature_vector
# # test_feature_normal = (test_feature_vector - mean) / (std + 1e-7)
#
# # one-hot encoding
# train_label_onehot = tf.keras.utils.to_categorical(train_label)
# test_label_onehot = tf.keras.utils.to_categorical(test_label)
#
# train model
model = model_train(
    train_feature, train_label, test_feature, test_label
)

#accuracy evaluation
# accuracy = model.evaluate(test_batcher, test_batcher)
# print("\n[RootMeanSquaredError] = ", accuracy["RootMeanSquaredError"])
#
# # save model
# filename = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
# # model.save(filename + ".h5")
# del model
