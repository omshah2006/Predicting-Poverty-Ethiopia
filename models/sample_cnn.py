import tensorflow as tf

layers = tf.keras.layers


def sample_cnn(num_classes, input_shape, fl_activation):
    model = tf.keras.Sequential()

    # model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', input_shape=input_shape))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='gelu'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.MaxPooling2D())
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='gelu', padding='same', kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='gelu', padding='same', kernel_initializer='he_uniform'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='gelu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='gelu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='gelu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='gelu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.4))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=128, kernel_initializer='he_uniform', activation='gelu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=num_classes, kernel_initializer='he_uniform', activation=fl_activation))

    return model


def sample_vgg(num_classes, input_shape, fl_activation):
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation='gelu', padding='same', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='gelu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3), activation='gelu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='gelu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(256, (3, 3), activation='gelu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='gelu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='gelu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(512, (3, 3), activation='gelu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='gelu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='gelu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(512, (3, 3), activation='gelu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='gelu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='gelu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='gelu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation=fl_activation))

    return model
