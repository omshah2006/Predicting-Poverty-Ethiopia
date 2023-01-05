import tensorflow as tf
from models import vgg16
import numpy as np
import time
import matplotlib.pyplot as plt


lr = 0.0001
batch_size = 128
EPOCHS = 50

# Build your model here
model = vgg16.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling="max",
    classes=10,
    classifier_activation="softmax",
)
for layer in model.layers:
    layer.trainable = False

x = tf.keras.layers.Flatten(name="flatten")(model.layers[-1].output)
x = tf.keras.layers.Dense(4096, activation="relu", name="fc1")(x)
x = tf.keras.layers.Dense(4096, activation="relu", name="fc2")(x)
x = tf.keras.layers.Dense(1000, activation="relu", name="fc3")(x)

x = tf.keras.layers.Dense(10, activation="softmax", name="predictions")(x)

vgg_model = tf.keras.models.Model(inputs=model.inputs, outputs=x)


def model_train(train_feature, train_label, test_feature, test_label):
    # model definition
    model = vgg_model

    # training definition
    batch_num = 64
    epoch_num = 20
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(train_feature)

    # train
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    history = model.fit(
        datagen.flow(train_feature, train_label, batch_size=batch_num),
        steps_per_epoch=int(len(train_feature) / batch_num),
        epochs=epoch_num,
        validation_data=(test_feature, test_label),
        verbose=2,
    )

    # accuracy evaluation
    accuracy = model.evaluate(test_feature, test_label)
    print("\n[Accuracy] = ", accuracy[1])

    return model


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

# train model
model = model_train(
    train_feature_normal, train_label_onehot, test_feature_normal, test_label_onehot
)
accuracy = model.evaluate(test_feature_normal, test_label_onehot)
print("\n[Accuracy] = ", accuracy)

# save model
filename = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
# model.save(filename + ".h5")
del model
