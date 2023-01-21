import tensorflow as tf

layers = tf.keras.layers


def sample_cnn(num_classes, input_shape):
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.4))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=128, kernel_initializer='he_uniform', activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=num_classes, kernel_initializer='he_uniform', activation='softmax'))

    return model
