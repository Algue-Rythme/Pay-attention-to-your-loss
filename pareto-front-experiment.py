"""
TRAINING VGG LIPSCHITZ NETWORKS ON CIFAR-10 DATASET

Usage:
python train.py --model_type lipschitz -bs 512 -e 3 -lr 1e-3 \
    -is 1 --loss HKR --alphaHKR 100 --min_margin 2.3 --save
"""
import argparse

import tensorflow as tf
import dlt.data.loader as loader
import dlt.data.pipeline as pipeline
import dlt.infrastructure.distributed_training as distributed
from deel.lip.layers import (
    ScaledL2NormPooling2D,
    SpectralConv2D,
    SpectralDense,
    ScaledGlobalL2NormPooling2D,
)
from deel.lip.losses import (
    MulticlassHinge,
    MulticlassHKR,
    MulticlassKR,
    TauCategoricalCrossentropy,
)
import wandb


def load_and_preprocess_data(input_scaling, batch_size, strategy):
    """Loads, preprocesses and augments CIFAR-10 dataset

    Args:
        input_scaling (float): scaling factor to preprocess inputs. Set to 1 to
            get 0-255 images. Set to 1/255 to get 0-1 images.
        batch_size (int): batch size for tf.Dataset.batch()
        strategy (tf.distribute.Strategy): distribution strategy

    Returns:
        training set (with data augmentation), test set and dataset info
    """
    # Load CIFAR-10 dataset
    ds_train, ds_test, ds_info = loader.get_cifar10()

    # Process dataset
    def augment_image(x):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, 51 * input_scaling)
        x = tf.image.random_contrast(x, 1 / 1.3, 1.3)
        x = tf.image.random_hue(x, 0.1)
        x = tf.image.random_saturation(x, 1 / 1.2, 1.2)

        random_height = tf.random.uniform((), minval=32, maxval=40, dtype=tf.int32)
        random_width = tf.random.uniform((), minval=32, maxval=40, dtype=tf.int32)
        x = tf.image.resize(x, (random_height, random_width))
        x = tf.image.random_crop(x, (32, 32, 3))
        return x

    ds_train, ds_test = pipeline.prepare_data(
        ds_train,
        ds_test,
        lambda image: tf.cast(image, tf.float32) * input_scaling,
        lambda label: tf.one_hot(label, ds_info["nb_classes"]),
        augment_image,
        batch_size=batch_size * strategy.num_replicas_in_sync,
    )

    return ds_train, ds_test, ds_info


def create_model(model_type, input_shape, num_classes):
    """Creates a tiny VGG-like model, either regular or Lipschitz.

    Args:
        model_type (str): "regular" or "lipschitz"
        input_shape (tuple): 3-D shape of input images
        num_classes (int): number of output classes

    Returns:
        tf.keras.Model: a Keras VGG-like model
    """
    if model_type == "regular":
        CONV = tf.keras.layers.Conv2D
        DENSE = tf.keras.layers.Dense
        ACT = "relu"
        POOL = tf.keras.layers.MaxPool2D
        KERNEL_INIT = "glorot_uniform"
        GAP = tf.keras.layers.GlobalAvgPool2D
    elif model_type == "lipschitz":
        CONV = SpectralConv2D
        DENSE = SpectralDense
        ACT = "deel-lip>GroupSort2"
        POOL = ScaledL2NormPooling2D
        KERNEL_INIT = "orthogonal"
        GAP = ScaledGlobalL2NormPooling2D
    else:
        raise ValueError(
            "model_type was not recognized. Supported values are 'regular' or "
            "'lipschitz'."
        )

    # VGG config
    conv_sizes = (
        (32, 32),
        (64, 64, 64),
        (128, 128, 128),
    )
    dense_sizes = (128,)

    model_input = x = tf.keras.Input(input_shape)
    conv_kwargs = dict(
        kernel_size=3,
        padding="same",
        activation=ACT,
        kernel_initializer=KERNEL_INIT,
    )
    for i, block in enumerate(conv_sizes):
        for filters in block:
            x = CONV(filters, **conv_kwargs)(x)
        if i < len(conv_sizes) - 1:
            x = POOL()(x)
    x = GAP()(x)
    for units in dense_sizes:
        x = DENSE(units, activation=ACT, kernel_initializer=KERNEL_INIT)(x)
    model_output = DENSE(num_classes)(x)

    return tf.keras.Model(model_input, model_output)


def compile_model(model, loss, learning_rate, strategy):
    """Compiles the model with given loss and learning_rate

    Args:
        model (tf.keras.Model): model to compile
        loss (tf.keras.losses.Loss): loss
        learning_rate (float): learning rate
        strategy (tf.distribute.Strategy): distribution strategy
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    metrics = ["accuracy", MulticlassKR()]
    if type(loss) == MulticlassHKR:
        metrics += [MulticlassHinge(loss.min_margin)]
    with strategy.scope():
        model.compile(optimizer, loss, metrics)


def train_model(model, ds_train, ds_test, epochs, wandb_cb=False):
    """Trains model.

    Args:
        model (tf.keras.Model): model to train
        ds_train (tf.dataset): training set
        ds_test (tf.dataset): test set
        epochs (int): number of epochs
        wandb_cb (bool): whether to save training to W&B

    Returns:
        tf.keras.losses.Loss: a Keras loss
    """
    callbacks = []
    callbacks = [wandb.keras.WandbCallback()]
    model.fit(ds_train, epochs=epochs, validation_data=ds_test, callbacks=callbacks)


def parse_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, help="Model type")
    parser.add_argument("-bs", "--batch_size", type=int, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs")
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate")
    parser.add_argument("-is", "--input_scaling", type=float, help="Input scaling")
    parser.add_argument("--loss", type=str, default="HKR", help="Loss")
    parser.add_argument("--tauCCE", type=float, help="tau for CCE")
    parser.add_argument("--alphaHKR", type=float, help="alpha for HKR, -1 for inf")
    parser.add_argument("--min_margin", type=float, help="min_margin for HKR")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb")
    parser.add_argument("--save", action="store_true", help="Save results")

    args = parser.parse_args()
    if args.alphaHKR == -1:
        args.alphaHKR = float("inf")
    print("-------- Training arguments --------")
    for arg in vars(args):
        print(f"- {arg}: {getattr(args, arg)}")
    return args


def main():

    args = parse_training_args()
    wandb_active, save_active = args.wandb, args.save
    del args.wandb, args.save

    # Handle wandb run
    wandb.init(entity="anonymized", config=args)

    # 1. Load and preprocess CIFAR-10 dataset
    strategy = distributed.get_distributed_strategy()
    ds_train, ds_test, ds_info = load_and_preprocess_data(
        args.input_scaling, args.batch_size, strategy
    )

    # 2. Create and compile VGG model
    model = create_model(args.model_type, ds_info["input_shape"], ds_info["nb_classes"])
    if args.loss == "crossentropy":
        loss = TauCategoricalCrossentropy(args.tau)
    if args.loss == "HKR":
        loss = MulticlassHKR(alpha=args.alphaHKR, min_margin=args.min_margin)
    if args.loss == "hinge":
        loss = MulticlassHinge(min_margin=args.min_margin)

    steps_per_epoch = ds_info["nb_samples_train"] // args.batch_size + 1
    lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=steps_per_epoch * args.epochs,
        alpha=1e-2,
    )
    # lr = args.learning_rate
    compile_model(model, loss, lr, strategy)
    model.summary()

    # 3. Train model
    train_model(model, ds_train, ds_test, args.epochs, wandb_active)


if __name__ == "__main__":
    main()
