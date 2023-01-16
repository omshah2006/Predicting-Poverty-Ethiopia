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


def model_train(training, validation):
    # training definition
    batch_num = 1024
    epoch_num = 20
    # opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6)
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
            classifier_activation="softmax",
        )

        model = vgg_model
        print(model.summary())

        model.compile(loss='MeanSquaredError', optimizer="SGD", metrics=['RootMeanSquaredError'])

    tf.profiler.experimental.server.start(6000)

    history = model.fit(
        x=training,
        steps_per_epoch=int(4559 / batch_num),
        epochs=epoch_num,
        validation_data=validation,
        validation_steps=1302,
        # batch_size=batch_num,
        verbose=2,
    )

    return model


train_batcher = batcher.Batcher(shuffle=True, split='train').get_dataset()
val_batcher = batcher.Batcher(shuffle=False, split='val').get_dataset()
test_batcher = batcher.Batcher(shuffle=False, split='test').get_dataset()


# train model
model = model_train(
    train_batcher, val_batcher
)

#accuracy evaluation
# accuracy = model.evaluate(test_batcher, test_batcher)
# print("\n[RootMeanSquaredError] = ", accuracy["RootMeanSquaredError"])
#
# # save model
# filename = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
# # model.save(filename + ".h5")
# del model
