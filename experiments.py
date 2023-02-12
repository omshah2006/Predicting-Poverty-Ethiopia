import os
from train_spatial import train_model

# Experiment name: 'dataset_model'
def run_cifar_vgg16():
    trained_model = train_model(
        experiment_name='cifar_sample_vgg_10_epoch_224_224_run',
        platform="cloud",
        strategy="tpu",
        model_name="sample_vgg",
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
        num_epochs=2,
        train_steps=int(50000 / 128),
        val_steps=10000,
        verbose=2,
    )

def run_cifar_vgg16_regression():
    trained_model = train_model(
        experiment_name='cifar_sample_vgg_regression',
        platform="cloud",
        strategy="tpu",
        model_name="sample_vgg",
        dataset="cifar",
        optimizer="adam",
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
        num_epochs=20,
        train_steps=int(50000 / 128),
        val_steps=10000,
        verbose=2,
    )

def run_imagery_vgg16():
    trained_model = train_model(
        experiment_name='imagery_sample_vgg_regression',
        platform="cloud",
        strategy="tpu",
        model_name="sample_vgg",
        dataset="imagery",
        optimizer="adam",
        lr_rate=1e-3,
        momentum=0.9,
        weight_decay=1e-4,
        num_classes=1,
        weights=None,
        use_custom_top=True,
        input_shape=(224, 224, 8),
        fl_activation="linear",
        batch_size=64,
        use_l2_regularizer=True,
        batch_norm_decay=0.9,
        batch_norm_epsilon=1e-5,
        loss_func="MeanSquaredError",
        metrics=["RootMeanSquaredError"],
        steps_per_execution=32,
        num_epochs=200,
        train_steps=int(4559 / 128),
        val_steps=1302,
        verbose=2,
    )

def run_local():
    trained_model = train_model(
        experiment_name='imagery_sample_vgg_regression',
        platform="local",
        strategy="mirrored",
        model_name="sample_vgg",
        dataset="imagery",
        optimizer="adam",
        lr_rate=1e-3,
        momentum=0.9,
        weight_decay=1e-4,
        num_classes=1,
        weights=None,
        use_custom_top=True,
        input_shape=(224, 224, 9),
        fl_activation="linear",
        batch_size=64,
        use_l2_regularizer=True,
        batch_norm_decay=0.9,
        batch_norm_epsilon=1e-5,
        loss_func="MeanSquaredError",
        metrics=["RootMeanSquaredError"],
        steps_per_execution=32,
        num_epochs=200,
        train_steps=int(4559 / 128),
        val_steps=1302,
        verbose=2,
    )


if __name__ == '__main__':
    # Export TPU_NAME before run
    export_tpu_name = "export TPU_NAME=local"
    os.system(export_tpu_name)

    # Run experiment
    run_imagery_vgg16()
