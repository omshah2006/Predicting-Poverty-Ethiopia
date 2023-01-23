import tensorflow as tf

WEIGHTS_PATH_NO_TOP = (
    "https://storage.googleapis.com/tensorflow/"
    "keras-applications/vgg16/"
    "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
)

layers = tf.keras.layers


def vgg16(
    num_classes=1,
    weights="imagenet",
    use_custom_top=True,
    input_shape=(224, 224, 3),
    fl_activation='linear',
    batch_size=None,
):

    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1_conv1"
    )(inputs)

    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1_conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="block2_conv1"
    )(x)
    x = layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="block2_conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv1"
    )(x)
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv2"
    )(x)
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv3"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    # Block 4
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv1"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv2"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv3"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    # Block 5
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block5_conv1"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block5_conv2"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block5_conv3"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    # Create model.
    base_model = tf.keras.Model(inputs, x, name="vgg16")

    # Load weights.
    if weights == "imagenet":
        weights_path = tf.keras.utils.get_file(
            "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
            WEIGHTS_PATH_NO_TOP,
            cache_subdir="models",
            file_hash="6d6bbae143d832006294945121d1f1fc",
        )
        base_model.load_weights(weights_path, by_name=True)

    if use_custom_top:
        # Set up trainable and non-trainable layers
        for layer in base_model.layers:
            if weights == "imagenet":
                layer.trainable = False

        x = layers.Flatten(name="flatten")(base_model.layers[-1].output)
        x = layers.Dense(512, activation="relu", name="fc1")(x)
        # x = layers.Dense(4096, activation="relu", name="fc2")(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(num_classes, activation=fl_activation, name="predictions")(x)

    model = tf.keras.models.Model(inputs=base_model.inputs, outputs=x, name="vgg16")

    return model
