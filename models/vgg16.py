import tensorflow as tf

# Adapted from https://github.com/keras-team/keras/blob/e6784e4302c7b8cd116b74a784f4b78d60e83c26/keras/applications/vgg16.py

WEIGHTS_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/"
    "vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
)
WEIGHTS_PATH_NO_TOP = (
    "https://storage.googleapis.com/tensorflow/"
    "keras-applications/vgg16/"
    "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
)


def VGG16(
        include_top=False,
        include_custom_top=True,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling=None,
        classes=1000,
        custom_top_classes=1,
        classifier_activation="softmax",
):
    if not (weights in {"imagenet", None} or tf.io.gfile.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded.  Received: "
            f"weights={weights}"
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            'If using `weights` as `"imagenet"` with `include_top` '
            "as true, `classes` should be 1000.  "
            f"Received `classes={classes}`"
        )

    inputs = tf.keras.layers.Input(shape=input_shape)

    # Block 1
    x = tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1_conv1"
    )(inputs)

    x = tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1_conv2"
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="block2_conv1"
    )(x)
    x = tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="block2_conv2"
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = tf.keras.layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv1"
    )(x)
    x = tf.keras.layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv2"
    )(x)
    x = tf.keras.layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv3"
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = tf.keras.layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv1"
    )(x)
    x = tf.keras.layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv2"
    )(x)
    x = tf.keras.layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv3"
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    # Block 5
    x = tf.keras.layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block5_conv1"
    )(x)
    x = tf.keras.layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block5_conv2"
    )(x)
    x = tf.keras.layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block5_conv3"
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

    if include_top:
        # Classification block
        x = tf.keras.layers.Flatten(name="flatten")(x)
        x = tf.keras.layers.Dense(4096, activation="relu", name="fc1")(x)
        x = tf.keras.layers.Dense(4096, activation="relu", name="fc2")(x)
        x = tf.keras.layers.Dense(1000, activation="relu", name="fc2")(x)

        x = tf.keras.layers.Dense(
            classes, activation=classifier_activation, name="predictions"
        )(x)
    else:
        if pooling == "avg":
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = tf.keras.layers.GlobalMaxPooling2D()(x)

    # Create model.
    model = tf.keras.Model(inputs, x, name="vgg16")

    # Load weights.
    if weights == "imagenet":
        if include_top:
            weights_path = tf.keras.utils.get_file(
                "vgg16_weights_tf_dim_ordering_tf_kernels.h5",
                WEIGHTS_PATH,
                cache_subdir="models",
                file_hash="64373286793e3c8b2b4e3219cbf3544b",
            )
        else:
            weights_path = tf.keras.utils.get_file(
                "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
                WEIGHTS_PATH_NO_TOP,
                cache_subdir="models",
                file_hash="6d6bbae143d832006294945121d1f1fc",
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    if include_custom_top:
        # Set up trainable and non-trainable layers
        for layer in model.layers:
            layer.trainable = False

        x = tf.keras.layers.Flatten(name="flatten")(model.layers[-1].output)
        x = tf.keras.layers.Dense(4096, activation="relu", name="fc1")(x)
        x = tf.keras.layers.Dense(4096, activation="relu", name="fc2")(x)
        x = tf.keras.layers.Dense(1000, activation="relu", name="fc3")(x)

        x = tf.keras.layers.Dense(custom_top_classes, activation="sigmoid", name="predictions")(x)

        model = tf.keras.models.Model(inputs=model.inputs, outputs=x)

    return model
