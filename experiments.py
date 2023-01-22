from train_spatial import train_model

# Experiment name: 'dataset_model'
def run_cifar_vgg16():
    trained_model = train_model(
        experiment_name='cifar_vgg16',
        platform="cloud",
        strategy="tpu",
        model_name="vgg16",
        dataset="cifar",
        optimizer="adam",
        lr_rate=0.00001,
        num_classes=10,
        weights="imagenet",
        use_custom_top=True,
        input_shape=(32, 32, 3),
        fl_activation="relu",
        batch_size=512,
        use_l2_regularizer=True,
        batch_norm_decay=0.9,
        batch_norm_epsilon=1e-5,
        loss_func="categorical_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=32,
        num_epochs=100,
        train_steps=int(50000 / 512),
        val_steps=10000,
        verbose=2,
    )


def run_cifar_vgg16_bn():
    trained_model = train_model(
        experiment_name='cifar_vgg16',
        platform="cloud",
        strategy="tpu",
        model_name="vgg16_bn",
        dataset="cifar",
        optimizer="adam",
        lr_rate=0.00001,
        num_classes=10,
        weights="imagenet",
        use_custom_top=True,
        input_shape=(32, 32, 3),
        fl_activation="relu",
        batch_size=256,
        use_l2_regularizer=True,
        batch_norm_decay=0.9,
        batch_norm_epsilon=1e-5,
        loss_func="categorical_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=32,
        num_epochs=100,
        train_steps=int(50000 / 256),
        val_steps=10000,
        verbose=2,
    )


def run_cifar_sample_cnn():
    trained_model = train_model(
        experiment_name='cifar_sample_cnn',
        platform="cloud",
        strategy="tpu",
        model_name="sample_cnn",
        dataset="cifar",
        optimizer="adam",
        lr_rate=0.00001,
        num_classes=10,
        weights="imagenet",
        use_custom_top=True,
        input_shape=(224, 224, 3),
        fl_activation="relu",
        batch_size=512,
        use_l2_regularizer=True,
        batch_norm_decay=0.9,
        batch_norm_epsilon=1e-5,
        loss_func="categorical_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=32,
        num_epochs=100,
        train_steps=int(50000 / 512),
        val_steps=10000,
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
    run_cifar_vgg16_bn()
