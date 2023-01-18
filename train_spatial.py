import tensorflow as tf
from models import vgg16
import matplotlib.pyplot as plt
from google.cloud import storage
from models import regularized_vgg16
import numpy as np
import time
from dataset import batcher


cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.TPUStrategy(cluster_resolver)


def model_train(training, validation):
    # training definition
    batch_num = 512
    epoch_num = 100
    # datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    #     rotation_range=15,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     horizontal_flip=True,
    # )
    # datagen.fit(train_feature)

    # train
    with strategy.scope():
        opt = tf.keras.optimizers.SGD(learning_rate=0.00001)
        # Build your model here
        vgg_model = vgg16.VGG16(
            include_imagenet_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3),
            pooling="max",
            classes=1,
            classifier_activation="linear",
        )

        # vgg_model = regularized_vgg16.regularized_vgg16(1)

        model = vgg_model
        print(model.summary())

        model.compile(loss='MeanSquaredError', optimizer=opt, metrics=['RootMeanSquaredError'], steps_per_execution=32)

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

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    fig_name = 'loss-plot-bnmodel.png'
    plt.savefig('models/plots/' + fig_name)
    upload_to_bucket(fig_name, 'models/plots/' + fig_name)

    return model


train_batcher = batcher.Batcher(shuffle=True, split='train').get_dataset()
val_batcher = batcher.Batcher(shuffle=False, split='val').get_dataset()
test_batcher = batcher.Batcher(shuffle=False, split='test').get_dataset()


# train model
model = model_train(
    train_batcher, val_batcher
)


def upload_to_bucket(blob_name, path_to_file, bucket_name='ppt-central-bucket'):
    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(path_to_file)

    # returns a public url
    return blob.public_url

#accuracy evaluation
# accuracy = model.evaluate(test_batcher, test_batcher)
# print("\n[RootMeanSquaredError] = ", accuracy["RootMeanSquaredError"])
#
# # save model
# filename = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
# # model.save(filename + ".h5")
# del model
