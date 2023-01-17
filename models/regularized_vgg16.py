import tensorflow as tf

WEIGHTS_PATH_NO_TOP = (
    "https://storage.googleapis.com/tensorflow/"
    "keras-applications/vgg16/"
    "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
)

layers = tf.keras.layers


def _gen_l2_regularizer(use_l2_regularizer=True, l2_weight_decay=1e-4):
    return tf.keras.regularizers.L2(l2_weight_decay) if use_l2_regularizer else None


def regularized_vgg16(
    num_classes,
    weights='imagenet',
    use_custom_top=True,
    batch_size=None,
    use_l2_regularizer=True,
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5,
):
    """Instantiates the VGG16 architecture.
    Args:
      num_classes: `int` number of classes for image classification.
      batch_size: Size of the batches for each step.
      use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.
      batch_norm_decay: Moment of batch norm layers.
      batch_norm_epsilon: Epsilon of batch borm layers.
    Returns:
      A Keras model instance.
    """
    input_shape = (224, 224, 3)
    img_input = layers.Input(shape=input_shape, batch_size=batch_size)

    x = img_input

    if tf.keras.backend.image_data_format() == "channels_first":
        x = layers.Permute((3, 1, 2))(x)
        bn_axis = 1
    else:  # channels_last
        bn_axis = 3
    # Block 1
    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name="block1_conv1",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name="bn_conv1",
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name="block1_conv2",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name="bn_conv2",
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name="block2_conv1",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name="bn_conv3",
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name="block2_conv2",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name="bn_conv4",
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name="block3_conv1",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name="bn_conv5",
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name="block3_conv2",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name="bn_conv6",
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name="block3_conv3",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name="bn_conv7",
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = layers.Conv2D(
        512,
        (3, 3),
        padding="same",
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name="block4_conv1",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name="bn_conv8",
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        512,
        (3, 3),
        padding="same",
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name="block4_conv2",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name="bn_conv9",
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        512,
        (3, 3),
        padding="same",
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name="block4_conv3",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name="bn_conv10",
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    # Block 5
    x = layers.Conv2D(
        512,
        (3, 3),
        padding="same",
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name="block5_conv1",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name="bn_conv11",
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        512,
        (3, 3),
        padding="same",
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name="block5_conv2",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name="bn_conv12",
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        512,
        (3, 3),
        padding="same",
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name="block5_conv3",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name="bn_conv13",
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

    # Create model.
    base_model = tf.keras.Model(img_input, x, name="vgg16_base")

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
        for layer in base_model.layers:
            layer.trainable = False

        x = layers.Flatten(name="flatten")(base_model.layers[-1].output)
        x = layers.Dense(
            4096, kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer), name="fc1"
        )(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(
            4096, kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer), name="fc2"
        )(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(
            num_classes,
            kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
            name="fc1000",
        )(x)

        x = layers.Activation("linear", dtype="float32")(x)

        model = tf.keras.models.Model(img_input, x, name="vgg16_full")
    else:
        model = tf.keras.models.Model(img_input, x, name="vgg16_headless")


    return model
