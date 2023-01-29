import tensorflow as tf
from models import vgg16
import matplotlib.pyplot as plt
from google.cloud import storage
from models import vgg16_bn
from models import sample_cnn
from dataset import batcher


def use_distributed_strategy(strategy_type):
    if strategy_type == "tpu":
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)
    elif strategy_type == "mirrored":
        strategy = tf.distribute.MirroredStrategy()
    else:
        raise ValueError("Strategy not found.")

    # Returns the distribution strategy for specified architecture
    return strategy


def create_metrics_plots(platform, history, fig_name):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig("models/plots/" + fig_name + "loss_plot")

    plt.clf()

    # plt.plot(history.history["accuracy"])
    # plt.plot(history.history["val_accuracy"])
    plt.plot(history.history["root_mean_squared_error"])
    plt.plot(history.history["val_root_mean_squared_error"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig("models/plots/" + fig_name + "accuracy_plot")

    if platform == "cloud":
        upload_to_bucket(fig_name + "loss_plot", "models/plots/" + fig_name + "loss_plot" + ".png")
        upload_to_bucket(fig_name + "accuracy_plot", "models/plots/" + fig_name + "accuracy_plot" + ".png")


def upload_to_bucket(blob_name, path_to_file, bucket_name="ppt-central-bucket"):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(path_to_file)

    # Returns a public url
    return blob.public_url


def train_model(
    experiment_name,
    platform,
    strategy,
    model_name,
    dataset,
    optimizer,
    lr_rate,
    momentum,
    weight_decay,
    num_classes,
    weights,
    use_custom_top,
    input_shape,
    fl_activation,
    batch_size,
    use_l2_regularizer,
    batch_norm_decay,
    batch_norm_epsilon,
    loss_func,
    metrics,
    steps_per_execution,
    num_epochs,
    train_steps,
    val_steps,
    verbose,
):
    # Define training strategy
    strategy = use_distributed_strategy(strategy)

    # Create dataset batches
    if platform == "local":
        bucket = False
    elif platform == "cloud":
        bucket = True
    else:
        raise ValueError("Platform must be either 'local' or 'cloud'.")

    if dataset == 'imagery':
        train_batcher = batcher.Batcher(
            image_shape=input_shape, bucket=bucket, batch_size=batch_size, shuffle=False, split="train"
        ).get_dataset()
        val_batcher = batcher.Batcher(
            image_shape=input_shape, bucket=bucket, batch_size=batch_size, shuffle=False, split="val"
        ).get_dataset()
    elif dataset == 'cifar':
        train_batcher, val_batcher = batcher.Batcher(image_shape=input_shape, bucket=bucket, batch_size=batch_size).create_cifar_dataset_test()
    else:
        raise ValueError("Dataset must be either 'imagery' or 'cifar'.")

    # augmentation code
    # datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    #     rotation_range=15,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     horizontal_flip=True,
    # )
    # datagen.fit(train_feature)

    # Compile model
    with strategy.scope():
        if optimizer == "sgd":
            opt = tf.keras.optimizers.SGD(learning_rate=lr_rate, momentum=momentum, nesterov=True)
        elif optimizer == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=lr_rate)

        # Instantiate model
        if model_name == "vgg16":
            model = vgg16.vgg16(
                num_classes, weights, use_custom_top, input_shape, fl_activation, batch_size
            )
        elif model_name == "vgg16_bn":
            model = vgg16_bn.vgg16_bn(
                num_classes,
                weights,
                use_custom_top,
                input_shape,
                fl_activation,
                batch_size,
                use_l2_regularizer,
                batch_norm_decay,
                batch_norm_epsilon,
            )
        elif model_name == "sample_cnn":
            model = sample_cnn.sample_cnn(num_classes, input_shape, fl_activation)
        elif model_name == "sample_vgg":
            model = sample_cnn.sample_vgg(num_classes, input_shape, fl_activation)
        else:
            raise ValueError("Model not found.")

        print(model.summary())

        model.compile(
            loss=loss_func,
            optimizer=opt,
            metrics=metrics,
            steps_per_execution=steps_per_execution,
        )

    def lr_scheduler(epoch):
        return lr_rate * (0.5 ** (epoch // 30))

    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    # Train model
    history = model.fit(
        x=train_batcher,
        steps_per_epoch=train_steps,
        epochs=num_epochs,
        validation_data=val_batcher,
        validation_steps=val_steps,
        # batch_size=batch_num,
        callbacks=[reduce_lr],
        verbose=verbose,
    )

    create_metrics_plots(platform=platform, history=history, fig_name=experiment_name)

    # Save model
    filename = experiment_name + '.h5'
    save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    model.save('saved_models/' + filename, options=save_locally)
    upload_to_bucket('saved_models/' + filename, 'saved_models/' + filename)
    del model

    return None


# if __name__ == '__main__':
    # train model

    # test_batcher = batcher.Batcher(shuffle=False, split='test').get_dataset()

    # accuracy evaluation
    # accuracy = model.evaluate(test_batcher, test_batcher)
    # print("\n[RootMeanSquaredError] = ", accuracy["RootMeanSquaredError"])