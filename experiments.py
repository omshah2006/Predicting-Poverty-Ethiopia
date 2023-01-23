from train_spatial import train_model

# Experiment name: 'dataset_model'
def run_cifar_vgg16():
    trained_model = train_model(
        experiment_name='cifar_sample_vgg_10_epoch_224_224_run',
        platform="cloud",
        strategy="tpu",
        model_name="vgg16",
        dataset="cifar",
        optimizer="adam",
        lr_rate=1e-3,
        momentum=0.9,
        weight_decay=1e-4,
        num_classes=10,
        weights=None,
        use_custom_top=True,
        input_shape=(224, 224, 3),
        fl_activation="softmax",
        batch_size=128,
        use_l2_regularizer=True,
        batch_norm_decay=0.9,
        batch_norm_epsilon=1e-5,
        loss_func="categorical_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=32,
        num_epochs=100,
        train_steps=int(50000 / 128),
        val_steps=10000,
        verbose=2,
    )

def run_imagery_vgg16():
    trained_model = train_model(
        experiment_name='cifar_sample_vgg_10_epoch_224_224_run',
        platform="cloud",
        strategy="tpu",
        model_name="sample_vgg",
        dataset="imagery",
        optimizer="sgd",
        lr_rate=1e-3,
        momentum=0.9,
        weight_decay=1e-4,
        num_classes=1,
        weights=None,
        use_custom_top=True,
        input_shape=(224, 224, 3),
        fl_activation="linear",
        batch_size=128,
        use_l2_regularizer=True,
        batch_norm_decay=0.9,
        batch_norm_epsilon=1e-5,
        loss_func="MeanSquaredError",
        metrics=["RootMeanSquaredError"],
        steps_per_execution=32,
        num_epochs=10,
        train_steps=int(4559 / 128),
        val_steps=1302,
        verbose=2,
    )

def run_local():
    trained_model = train_model(
        experiment_name='cifar_sample_cnn_local',
        platform="local",
        strategy="mirrored",
        model_name="vgg16",
        dataset="cifar",
        optimizer="adam",
        lr_rate=0.00001,
        momentum=0.9,
        weight_decay=1e-4,
        num_classes=10,
        weights="imagenet",
        use_custom_top=True,
        input_shape=(32, 32, 3),
        fl_activation="relu",
        batch_size=64,
        use_l2_regularizer=True,
        batch_norm_decay=0.9,
        batch_norm_epsilon=1e-5,
        loss_func="categorical_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=None,
        num_epochs=100,
        train_steps=int(50000 / 64),
        val_steps=10000,
        verbose=2,
    )


if __name__ == '__main__':
    run_cifar_vgg16()
